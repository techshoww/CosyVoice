# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Bofan Zhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
from typing import Generator
import torch
import numpy as np
import threading
import time
from torch.nn import functional as F
from contextlib import nullcontext
import uuid
from cosyvoice.utils.common import fade_in_out
from cosyvoice.utils.file_utils import convert_onnx_to_trt, export_cosyvoice2_vllm
from cosyvoice.utils.common import TrtContextWrapper
import onnxruntime as ort
from transformers import AutoConfig
class CosyVoiceModel:

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        if self.fp16 is True:
            self.llm.half()
            self.flow.half()
        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 20
        # mel fade in out
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.mel_overlap_dict = {}
        self.flow_cache_dict = {}
        self.hift_cache_dict = {}

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device), strict=True)
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device), strict=False)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_jit(self, llm_text_encoder_model, llm_llm_model, flow_encoder_model):
        llm_text_encoder = torch.jit.load(llm_text_encoder_model, map_location=self.device)
        self.llm.text_encoder = llm_text_encoder
        llm_llm = torch.jit.load(llm_llm_model, map_location=self.device)
        self.llm.llm = llm_llm
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, trt_concurrent, fp16):
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        if not os.path.exists(flow_decoder_estimator_model) or os.path.getsize(flow_decoder_estimator_model) == 0:
            convert_onnx_to_trt(flow_decoder_estimator_model, self.get_trt_kwargs(), flow_decoder_onnx_model, fp16)
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        assert estimator_engine is not None, 'failed to load trt {}'.format(flow_decoder_estimator_model)
        self.flow.decoder.estimator = TrtContextWrapper(estimator_engine, trt_concurrent=trt_concurrent, device=self.device)

    def get_trt_kwargs(self):
        min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
        opt_shape = [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)]
        max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]
        input_names = ["x", "mask", "mu", "cond"]
        return {'min_shape': min_shape, 'opt_shape': opt_shape, 'max_shape': max_shape, 'input_names': input_names}

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        with self.llm_context, torch.cuda.amp.autocast(self.fp16 is True and hasattr(self.llm, 'vllm') is False):
            if isinstance(text, Generator):
                assert isinstance(self, CosyVoice2Model) and not hasattr(self.llm, 'vllm'), 'streaming input text is only implemented for CosyVoice2 and do not support vllm!'
                for i in self.llm.inference_bistream(text=text,
                                                     prompt_text=prompt_text.to(self.device),
                                                     prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                                     prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                     prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                                     embedding=llm_embedding.to(self.device)):
                    self.tts_speech_token_dict[uuid].append(i)
            else:
                for i in self.llm.inference(text=text.to(self.device),
                                            text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                            prompt_text=prompt_text.to(self.device),
                                            prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                            prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                            prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                            embedding=llm_embedding.to(self.device),
                                            uuid=uuid):
                    self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True

    def vc_job(self, source_speech_token, uuid):
        self.tts_speech_token_dict[uuid] = source_speech_token.flatten().tolist()
        self.llm_end_dict[uuid] = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False, speed=1.0):
        with torch.cuda.amp.autocast(self.fp16):
            tts_mel, self.flow_cache_dict[uuid] = self.flow.inference(token=token.to(self.device),
                                                                      token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                                                      prompt_token=prompt_token.to(self.device),
                                                                      prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                                                      prompt_feat=prompt_feat.to(self.device),
                                                                      prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                                                      embedding=embedding.to(self.device),
                                                                      flow_cache=self.flow_cache_dict[uuid])

        # mel overlap fade in out
        if self.mel_overlap_dict[uuid].shape[2] != 0:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech

    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32), stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
            self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
            self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
        if source_speech_token.shape[1] == 0:
            p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        else:
            p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
        p.start()
        if stream is True:
            token_hop_len = self.token_min_hop_len
            while True:
                time.sleep(0.1)
                if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len]) \
                        .unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     finalize=False)
                    yield {'tts_speech': this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
            self.flow_cache_dict.pop(this_uuid)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.current_stream().synchronize()


class CosyVoice2Model(CosyVoiceModel):

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool = False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.infer_axmodel = eval(os.getenv("infer_axmodel", "False"))
        self.infer_onnx = eval(os.getenv("infer_onnx", "False"))
        self.export_onnx = eval(os.getenv("export_onnx", "False"))

        assert not (self.infer_axmodel and self.infer_onnx)
        assert not (self.infer_axmodel and self.export_onnx)
        assert not (self.infer_onnx and self.export_onnx)

        if self.infer_axmodel:
            sys.path.append("npu_infer")
            from utils_axinfer import AxModelInfer
            from utils_lm import Qwen2LM_AXInfer
            cfg = AutoConfig.from_pretrained(
                "pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN", trust_remote_code=True
            )
            self.llm = Qwen2LM_AXInfer(cfg, "CosyVoice-BlankEN-Ax650-prefill_512", "qwen2", prefill_len=512, lastN=1023, chunk_len=128)
        else:
            self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        if self.fp16 is True and not self.infer_axmodel:
            self.llm.half()
            self.flow.half()
        # NOTE must matching training static_chunk_size
        self.token_hop_len = 25
        # hift cache
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}

        self.max_infer_chunk_num = 3                # 用于固定shape 推理

        print("------------------------infer_axmodel",self.infer_axmodel)
        print("------------------------infer_onnx",self.infer_onnx)
        print("------------------------export_onnx",self.export_onnx)
        
        self.flow.init_mask()

        if self.infer_onnx:
            self.flow_input_embed = np.load("flow.input_embedding.npy")
            self.flow_28 = ort.InferenceSession("flow_28.onnx")
            self.flow_53 = ort.InferenceSession("flow_53.onnx")
            self.flow_78 = ort.InferenceSession("flow_78.onnx")
            self.flow_50_final = ort.InferenceSession("flow_50_final.onnx")

            self.hift_50_first = ort.InferenceSession("hift_50_first.onnx")
            self.hift_58 = ort.InferenceSession("hift_58.onnx")
        elif self.infer_axmodel:
            self.flow_input_embed = np.load("token2wav-axmodels/flow.input_embedding.npy")
            self.flow_encoder_28 = AxModelInfer("token2wav-axmodels/flow_encoder_28.axmodel")
            self.flow_encoder_53 = AxModelInfer("token2wav-axmodels/flow_encoder_53.axmodel")
            self.flow_encoder_78 = AxModelInfer("token2wav-axmodels/flow_encoder_78.axmodel")
            self.flow_encoder_50_final = AxModelInfer("token2wav-axmodels/flow_encoder_50_final.axmodel")

            self.flow_estimator_200 = AxModelInfer("token2wav-axmodels/flow_estimator_200.axmodel")
            self.flow_estimator_250 = AxModelInfer("token2wav-axmodels/flow_estimator_250.axmodel")
            self.flow_estimator_300 = AxModelInfer("token2wav-axmodels/flow_estimator_300.axmodel")

            # self.hift_50_first = AxModelInfer("token2wav-axmodels/hift_50_first.axmodel")
            # self.hift_58 = AxModelInfer("token2wav-axmodels/hift_58.axmodel")
            self.hift_50_first_p1 = AxModelInfer("hift_p1_50_first.onnx")
            self.hift_50_first_p2 = AxModelInfer("token2wav-axmodels/hift_p2_50_first.axmodel")
            self.hift_58_p1 = AxModelInfer("hift_p1_58.onnx")
            self.hift_58_p2 = AxModelInfer("token2wav-axmodels/hift_p2_58.axmodel")


    def load(self, llm_model, flow_model, hift_model):
        if not self.infer_onnx and not self.infer_axmodel:
            self.llm.load_state_dict(torch.load(llm_model, map_location=self.device), strict=True)
            self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device), strict=False)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()
        

    def load_jit(self, flow_encoder_model):
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_vllm(self, model_dir):
        export_cosyvoice2_vllm(self.llm, model_dir, self.device)
        from vllm import EngineArgs, LLMEngine
        engine_args = EngineArgs(model=model_dir,
                                 skip_tokenizer_init=True,
                                 enable_prompt_embeds=True,
                                 gpu_memory_utilization=0.2)
        self.llm.vllm = LLMEngine.from_engine_args(engine_args)
        self.llm.lock = threading.Lock()
        del self.llm.llm.model.model.layers

    def flow_onnx(self, token_embedding,  prompt_feat, embedding, token_len, finalize):
        
        mu,  spks, cond = self.flow_encoder_onnx(token_embedding,  prompt_feat, embedding, token_len, finalize)

        # mask = self.flow.get_buffer(f"mask_{mu.shape[2]}").to(mu)
        # mask = mask.unsqueeze(1)
        # print("mask",mask)
        mask = torch.ones((mu.shape[2]),  device=mu.device).unsqueeze(0).unsqueeze(0)
        feat = self.flow_decoder_onnx(mu, mask, spks, cond)
        mel_len1, mel_len2 = prompt_feat.shape[1], mu.shape[2] - prompt_feat.shape[1]

        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float()

    def flow_encoder_onnx(self, token_embedding,  prompt_feat, embedding, token_len, finalize):
        if not finalize:    
            if token_len == 28:
                sess = self.flow_encoder_28
            elif token_len == 53:
                sess = self.flow_encoder_53
            elif token_len == 78:
                sess = self.flow_encoder_78 
            else:
                raise NotImplementedError(f"token_len:{token_len}") 
        elif finalize:
            if token_len == 50:
                sess = self.flow_encoder_50_final
            else:
                raise NotImplementedError(f"finalize:{finalize},token_len:{token_len}") 
        
        inputs = {"token_embedding":token_embedding,
                    "prompt_feat":prompt_feat.cpu().numpy(),
                    "embedding":embedding.cpu().numpy()}
        mu,  spks, cond = sess.run(None, inputs)

        device = prompt_feat.device
        mu = torch.from_numpy(mu).to(device)
        spks = torch.from_numpy(spks).to(device)
        cond = torch.from_numpy(cond).to(device)
        return mu,  spks, cond

    def flow_decoder_estimator_onnx(self, x, mask, mu, t, spks, cond):
        x_len = x.shape[2]
        if x_len == 200:
            sess = self.flow_estimator_200
        elif x_len == 250:
            sess = self.flow_estimator_250
        elif x_len == 300:
            sess = self.flow_estimator_300
        else:
            raise NotImplementedError(f"x_len:{x_len}") 

        inputs = {"x":x.cpu().numpy(), "mask":mask.cpu().numpy(), "mu":mu.cpu().numpy(),  "t":t.cpu().numpy(), "spks":spks.cpu().numpy(), "cond":cond.cpu().numpy()}
        output = sess.run(None, inputs)[0]
        output = torch.from_numpy(output).to(x.device)
        return output


    def flow_decoder_solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        streaming = True
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        # sol = []

        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        # x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        # mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=x.dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=x.dtype)
        spks_in = torch.zeros([2, 80], device=x.device, dtype=x.dtype)
        cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox
            # x_in[:] = x
            x_in = torch.cat([x,x],dim=0)
            # mask_in[:] = mask
            mask_in = torch.cat([mask, mask],dim=0)
            # mu_in[0] = mu
            mu_in = torch.cat([mu, mu_in[1:]], dim=0)
            # t_in[:] = t.unsqueeze(0)
            t_in = torch.cat([t, t],dim=0)
            # spks_in[0] = spks
            spks_in = torch.cat([spks, spks_in[1:]], dim=0)
            # cond_in[0] = cond
            cond_in = torch.cat([cond, cond_in[1:]], dim=0)
            # dphi_dt = self.forward_estimator(
            #     x_in, mask_in,
            #     mu_in, t_in,
            #     spks_in,
            #     cond_in,
            #     streaming
            # )
            dphi_dt = self.flow_decoder_estimator_onnx(
                x_in, mask_in,
                mu_in, t_in,
                spks_in,
                cond_in,
            )
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = ((1.0 + self.flow.decoder.inference_cfg_rate) * dphi_dt - self.flow.decoder.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            # sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            else:
                sol = x

        # return sol[-1].float()
        return sol.float()

    def flow_decoder_onnx(self, mu, mask, spks, cond, n_timesteps=10, temperature=1.0):

        z = self.flow.decoder.rand_noise[:, :, :mu.size(2)].to(mu.device).to(mu.dtype) * temperature
        # fix prompt and overlap part mu and z
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.flow.decoder.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.flow_decoder_solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

    def hift_onnx(self, tts_mel, cache_source):
        mel_len = tts_mel.shape[2]

        if cache_source.shape[2] == 0:
            if mel_len == 50:
                # sess_hift = self.hift_50_first
                sess_hift_p1 = self.hift_50_first_p1
                sess_hift_p2 = self.hift_50_first_p2
            else:
                raise NotImplementedError 
        else:
            if mel_len == 58:
                # sess_hift = self.hift_58
                sess_hift_p1 = self.hift_58_p1
                sess_hift_p2 = self.hift_58_p2
            else:
                raise NotImplementedError 
       
        # if cache_source.shape[2] == 0:
        #     inputs = {"mel":tts_mel.cpu().numpy()}
        # else:
        #     inputs = {"mel":tts_mel.cpu().numpy(),
        #                 "hift_cache_source":cache_source.cpu().numpy()}
        # tts_speech, tts_source = sess_hift.run(None, inputs)

        t1 = time.time()
        inputs = {"mel":tts_mel.cpu().numpy()}
        s = sess_hift_p1.run(None, inputs)[0]
        t2 = time.time()
        print(f"hift part1 use time:{t2-t1} s", )

        if cache_source.shape[2] == 0:
            inputs = {"mel":tts_mel.cpu().numpy(), "s":s}
        else:
            inputs = {"mel":tts_mel.cpu().numpy(), "s":s,
                        "hift_cache_source":cache_source.cpu().numpy()}
            
        tts_speech, tts_source = sess_hift_p2.run(None, inputs)

        tts_speech = torch.from_numpy(tts_speech).to(tts_mel.device)
        tts_source = torch.from_numpy(tts_source).to(tts_mel.device)

        return tts_speech, tts_source

    def axllm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        
        for i in self.llm.inference(text_token_ids=text.cpu().numpy(),
                                    prompt_token_ids=prompt_text.cpu().numpy(),
                                    prompt_speech_token=llm_prompt_speech_token.cpu().numpy()):
            self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True

    def flow_embed_tokens(self, input_ids):
        assert len(input_ids.shape)==2, f"not support shape:{input_ids.shape}"
        ret = []
        for ids in input_ids:
            emb =  np.take(self.flow_input_embed, ids, axis=0)
            ret.append(emb)
        
        ret = np.stack(ret)
        return ret

    def token2wav(self, token, prompt_token, prompt_feat, embedding, token_offset, uuid, stream=False, finalize=False, speed=1.0):
        t1 = time.time()
        if self.infer_onnx or self.infer_axmodel:
            token_embedding = torch.concat([prompt_token.to(self.device), token.to(self.device)], dim=1) 
            # token_embedding = self.flow.input_embedding(token_embedding)
            token_embedding = self.flow_embed_tokens(token_embedding.detach().cpu().numpy())

            # embedding = F.normalize(embedding.to(self.device), dim=1)
            token_len = token.shape[1]
            tts_mel = self.flow_onnx(token_embedding,  prompt_feat, embedding, token_len, finalize)
        else:
            with torch.cuda.amp.autocast(self.fp16):
                print("token.shape",token.shape)
                print("token,min,max",token.min(), token.max())
                print("prompt_token,min,max",prompt_token.min(), prompt_token.max())

                if not self.export_onnx and not self.infer_onnx and not self.infer_axmodel:
                    tts_mel, _ = self.flow.inference(token=token.to(self.device),
                                                    token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                                    prompt_token=prompt_token.to(self.device),
                                                    prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                                    prompt_feat=prompt_feat.to(self.device),
                                                    prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                                    embedding=embedding.to(self.device),
                                                    streaming=stream,
                                                    finalize=finalize)
                elif self.export_onnx:
                    token_embedding = torch.concat([prompt_token.to(self.device), token.to(self.device)], dim=1) 
                    token_embedding = self.flow.input_embedding(token_embedding)
                    # embedding = F.normalize(embedding.to(self.device), dim=1)
                    if not finalize:
                        tts_mel = self.flow.inference_export(token_embedding=token_embedding.to(self.device),
                                                        prompt_feat=prompt_feat.to(self.device),
                                                        embedding=embedding)
                    else:
                        tts_mel = self.flow.inference_export_final(token_embedding=token_embedding.to(self.device),
                                                        prompt_feat=prompt_feat.to(self.device),
                                                        embedding=embedding.to(self.device))
                else:
                    raise NotImplementedError
        t2 = time.time()

        dt_flow = t2 - t1
        print("tts_mel.shape",tts_mel.shape)
        # tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
        
        if finalize:
            # tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
            neg_offset = token_offset * self.flow.token_mel_ratio - tts_mel.shape[2]        # 用来截取有效 tts_speech
            tts_mel = tts_mel[:,:,-self.token_hop_len*self.flow.token_mel_ratio:]           # 保留最小chunk_size 
        else:
            start =  min( token_offset // self.token_hop_len, self.max_infer_chunk_num-1) * self.token_hop_len
            tts_mel = tts_mel[:, :, start * self.flow.token_mel_ratio:]
        print("token_offset", token_offset)
        print("tts_mel.shape",tts_mel.shape)
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        print("318 tts_mel.shape",tts_mel.shape)
        t1 = time.time()
        if finalize is False:
            if not (self.infer_onnx or self.infer_axmodel):
                tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            else:
                tts_speech, tts_source = self.hift_onnx(tts_mel, hift_cache_source)
            print("tts_speech",tts_speech.shape)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            if not (self.infer_onnx or self.infer_axmodel):
                tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            else:
                tts_speech, tts_source = self.hift_onnx(tts_mel, hift_cache_source)
            tts_speech = tts_speech[:, neg_offset*480:]
            # tts_source = tts_source[:,:, neg_offset*480:]
            print("tts_speech",tts_speech.shape)
            
            if self.hift_cache_dict[uuid] is not None:
                print("self.hift_cache_dict[uuid]['speech'].shape",self.hift_cache_dict[uuid]['speech'].shape)
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        t2 = time.time()
        dt_hift = t2 - t1 

        dt = dt_flow + dt_hift 
        rtf = dt/(tts_speech.shape[1]/24000)
        print("---------------rft",rtf)
        return tts_speech

    def tts(self, text=torch.zeros(1, 0, dtype=torch.int32), flow_embedding=torch.zeros(0, 192), llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), source_speech_token=torch.zeros(1, 0, dtype=torch.int32), stream=False, speed=1.0, **kwargs):
        print("model.tts:")
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
        if source_speech_token.shape[1] == 0:
            if not self.infer_axmodel:
                p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
            else:
                p = threading.Thread(target=self.axllm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        else:
            p = threading.Thread(target=self.vc_job, args=(source_speech_token, this_uuid))
        p.start()
        if stream is True:
            token_offset = 0
            print("flow_prompt_speech_token.shape",flow_prompt_speech_token.shape)
            prompt_token_len = flow_prompt_speech_token.shape[1]
            prompt_token_align_len = (prompt_token_len//self.token_hop_len) * self.token_hop_len
            flow_prompt_speech_token = flow_prompt_speech_token[:, 0:prompt_token_align_len]
            prompt_speech_feat = prompt_speech_feat[:, 0:prompt_token_align_len*2]
            print("prompt_token_align_len",prompt_token_align_len)
            print("prompt_speech_feat",prompt_speech_feat.shape)
            print("flow_prompt_speech_token.shape",flow_prompt_speech_token.shape)

            prompt_token_pad = int(np.ceil(flow_prompt_speech_token.shape[1] / self.token_hop_len) * self.token_hop_len - flow_prompt_speech_token.shape[1])
            while True:
                time.sleep(0.1)
                this_token_hop_len = self.token_hop_len + prompt_token_pad if token_offset == 0 else self.token_hop_len
                
                if len(self.tts_speech_token_dict[this_uuid]) - token_offset >= this_token_hop_len + self.flow.pre_lookahead_len:
                    # this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_offset + this_token_hop_len + self.flow.pre_lookahead_len]).unsqueeze(dim=0)
                    start = token_offset -  min( token_offset // self.token_hop_len, self.max_infer_chunk_num-1) * self.token_hop_len
                    end = token_offset + this_token_hop_len + self.flow.pre_lookahead_len
                    print("start ,end:",start, end)
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][ start : token_offset + this_token_hop_len + self.flow.pre_lookahead_len]).unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     token_offset=token_offset,
                                                     uuid=this_uuid,
                                                     stream=stream,
                                                     finalize=False)
                    token_offset += this_token_hop_len
                    yield {'tts_speech': this_tts_speech.cpu()}
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) - token_offset < this_token_hop_len + self.flow.pre_lookahead_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            print("total speech token size:",len(self.tts_speech_token_dict[this_uuid]))
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            start = this_tts_speech_token.shape[1] -  min( this_tts_speech_token.shape[1] // self.token_hop_len, self.max_infer_chunk_num-1) * self.token_hop_len
            this_tts_speech_token = this_tts_speech_token[:, start:]
            print("final this_tts_speech_token",this_tts_speech_token.shape)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             token_offset=token_offset - start,
                                             uuid=this_uuid,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             token_offset=0,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.current_stream().synchronize()
