import os

import numpy as np
import onnxruntime as ort
from ml_dtypes import bfloat16
# from scipy.special import softmax
from transformers import AutoTokenizer
import gc
import dill 
import math
import torch
from torch import nn
from scipy.special import log_softmax
from typing import Dict, Optional, Callable, List, Generator
from utils_axinfer import AxModelInfer, AxLMInfer
import time 

def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    prob, indices = [], []
    cum_prob = 0.0
    sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
    for i in range(len(sorted_idx)):
        # sampling both top-p and numbers.
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += sorted_value[i]
            prob.append(sorted_value[i])
            indices.append(sorted_idx[i])
        else:
            break
    prob = torch.tensor(prob).to(weighted_scores)
    indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
    top_ids = indices[prob.multinomial(1, replacement=True)]
    return top_ids

def random_sampling(weighted_scores, decoded_tokens, sampling):
    top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True)
    return top_ids

def ras_sampling(weighted_scores, decoded_tokens, sampling, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids).sum().item()
    if rep_num >= win_size * tau_r:
        top_ids = random_sampling(weighted_scores, decoded_tokens, sampling)
    return top_ids

class Qwen2LM_AXInfer(AxLMInfer):
    def __init__(
        self, cfg, model_dir, model_name, prefill_len, lastN, chunk_len=-1, run_dynamic=False, lazy_load=False, provider_options=None
    ):

        super().__init__(cfg, model_dir, model_name, prefill_len, lastN, chunk_len, run_dynamic, lazy_load, provider_options=provider_options)

        self.embeds = np.load(f"{model_dir}/model.embed_tokens.weight.npy")
        self.llm_embedding = np.load(f"{model_dir}/llm.llm_embedding.npy")
        self.speech_embedding = np.load(f"{model_dir}/llm.speech_embedding.npy")
        self.total_prefill = prefill_len
        self.chunk_len = chunk_len

        kv_dim = (
            cfg.hidden_size
            // cfg.num_attention_heads
            * cfg.num_key_value_heads
        )
        self.k_caches = [
            np.zeros((1, lastN, kv_dim), dtype=bfloat16)
            for _ in range(cfg.num_hidden_layers)
        ]
        self.v_caches = [
            np.zeros((1, lastN, kv_dim), dtype=bfloat16)
            for _ in range(cfg.num_hidden_layers)
        ]

        self.llm_decoder = AxModelInfer(f"{model_dir}/llm_decoder.onnx")
        self.speech_token_size = 6561

    def embed_tokens(self, input_ids):
        assert len(input_ids.shape)==2, f"not support shape:{input_ids.shape}"
        ret = []
        for ids in input_ids:
            emb =  np.take(self.embeds, ids, axis=0)
            ret.append(emb)
        
        ret = np.stack(ret)
        return ret


    def speech_embed_tokens(self, input_ids):
        assert len(input_ids.shape)==2, f"not support shape:{input_ids.shape}"
        ret = []
        for ids in input_ids:
            emb =  np.take(self.speech_embedding, ids, axis=0)
            ret.append(emb)
        
        ret = np.stack(ret)
        return ret

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            sampling: int,
            ignore_eos: bool = True,
    ):
        num_trials, max_trials = 0, 100
        while True:
            top_ids = ras_sampling(weighted_scores, decoded_tokens, sampling)
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError('sampling reaches max_trials {} and still get eos when ignore_eos is True, check your input!'.format(max_trials))
        return top_ids

    def forward_prefill(self, input_embeds, position_ids):
        token_len = input_embeds.shape[1]
        indices = np.zeros((1, self.prefill_len), dtype=np.uint32)
        indices[:, 0:token_len] = position_ids.astype(np.uint32)
        mask = np.zeros((1, self.prefill_len, self.prefill_len)) - 65536
        data = np.zeros((1, self.prefill_len, self.hidden_size)).astype(bfloat16)
        data[:, 0:token_len] = input_embeds
        for i in range(token_len):
            mask[:, i, : i + 1] = 0

        mask = mask.astype(bfloat16)
        chunk_num = math.ceil(token_len / self.chunk_len)
        for i in range(self.num_hidden_layers):
            if self.chunk_len <= 0:
                input_feed = {
                    "K_cache": np.zeros((1, 1, self.hidden_size), dtype=bfloat16),
                    "V_cache": np.zeros((1, 1, self.hidden_size), dtype=bfloat16),
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }
                outputs = self.prefill_decoder_sessins[i](input_feed, shape_group=1)

                self.k_caches[i][:, :token_len, :] = outputs[0][:, :token_len, :]
                self.v_caches[i][:, :token_len, :] = outputs[1][:, :token_len, :]
                data[:, 0:token_len] = outputs[2][:, :token_len, :]
            else:
                layer_output = []
                for ck in range(chunk_num):
                    
                    gid = ck + 1
                    if ck==0:
                        input_feed = {
                            "K_cache": np.zeros((1, 1, self.hidden_size), dtype=bfloat16),
                            "V_cache": np.zeros((1, 1, self.hidden_size), dtype=bfloat16),
                            "indices": indices[:, 0:self.chunk_len],
                            "input": data[:, 0:self.chunk_len],
                            "mask": mask[:, 0:self.chunk_len, 0:self.chunk_len],
                        }
                        outputs = self.prefill_decoder_sessins[i](input_feed, shape_group=gid)
                        self.k_caches[i][:, :self.chunk_len, :] = outputs[0][:, :self.chunk_len, :]
                        self.v_caches[i][:, :self.chunk_len, :] = outputs[1][:, :self.chunk_len, :]

                    else:
                        input_feed = {
                            "K_cache": self.k_caches[i][:, :ck*self.chunk_len, :],
                            "V_cache": self.v_caches[i][:, :ck*self.chunk_len, :],
                            "indices": indices[:, ck*self.chunk_len:(ck+1)*self.chunk_len],
                            "input": data[:, ck*self.chunk_len:(ck+1)*self.chunk_len],
                            "mask": mask[:, ck*self.chunk_len:(ck+1)*self.chunk_len, 0:(ck+1)*self.chunk_len],
                        }
                        outputs = self.prefill_decoder_sessins[i](input_feed, shape_group=gid)
                        self.k_caches[i][:, ck*self.chunk_len:(ck+1)*self.chunk_len, :] = outputs[0][:, :self.chunk_len, :]
                        self.v_caches[i][:, ck*self.chunk_len:(ck+1)*self.chunk_len, :] = outputs[1][:, :self.chunk_len, :]

                    layer_output.append(outputs[2][:, :self.chunk_len, :])
                
                data = np.concatenate(layer_output, axis=1)

        post_out = self.post_process_session(
            {"input": data[:, token_len - 1 : token_len, :]}
        )[1]
        return post_out

    def forward_decode(self, input_embeds, position_id):

        # start_ids = np.max(indices) + 1
        
        mask = np.zeros((1, 1, self.lastN + 1), dtype=np.float32).astype(bfloat16)
        mask[:, :, : self.lastN] -= 65536
        mask[:, :, :position_id] = 0
        # for start_indice in range(np.max(indices) + 1, self.lastN + 1):

        # if self.prefill_len > 0 and start_indice < token_len:
        #     continue
        indices = np.array([position_id], np.uint32).reshape((1, 1))
        # start_ids += 1

        data = input_embeds.astype(bfloat16)
        
        for i in range(self.cfg.num_hidden_layers):
            input_feed = {
                "K_cache": self.k_caches[i],
                "V_cache": self.v_caches[i],
                "indices": indices,
                "input": data,
                "mask": mask,
            }

            outputs = self.prefill_decoder_sessins[i](input_feed, shape_group=0)

            self.k_caches[i][:, position_id, :] = outputs[0][:, :, :]
            self.v_caches[i][:, position_id, :] = outputs[1][:, :, :]
            data = outputs[2]
        
        post_out = self.post_process_session({"input": data})[1]
    
        return post_out
        

    def inference(
        self,
        text_token_ids,
        prompt_token_ids,
        prompt_speech_token=None,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
    ):

        text_token_ids = np.concatenate([prompt_token_ids, text_token_ids], axis=1)
        prompt_text_len = prompt_token_ids.shape[1]
        text_len = text_token_ids.shape[1]

        text_embeddings =  self.embed_tokens(text_token_ids)
        
        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding[0].reshape(1,1,-1)
        task_id_emb = self.llm_embedding[1].reshape(1,1,-1)

        if prompt_speech_token is not None:
            prompt_speech_token_emb = self.speech_embed_tokens(prompt_speech_token)
        else:
            prompt_speech_token_emb = np.zeros((1,0,self.cfg.hidden_size)).astype(bfloat16)

        lm_input = np.concatenate([sos_eos_emb, text_embeddings, task_id_emb, prompt_speech_token_emb], axis=1)

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # infer
        out_tokens = []
        t1 = time.time()
        for i in range(max_len):
            if i==0:
                position_ids = np.arange(lm_input.shape[1]).reshape(1,-1)
                y_pred = self.forward_prefill(lm_input, position_ids)
                position_id = lm_input.shape[1]
            else:
                y_pred = self.forward_decode(lm_input, position_id)
                position_id += 1
            logp = self.llm_decoder({"x":y_pred[:, -1].astype(np.float32)})[0]
            # logp = log_softmax(logp, axis=-1)
            logp = torch.from_numpy(logp)
            logp = logp.log_softmax(dim=-1)
            top_ids = self.sampling_ids(logp.squeeze(dim=0), out_tokens, sampling, ignore_eos=True if i < min_len else False).item()
            if top_ids == self.speech_token_size:
                break
            if top_ids > self.speech_token_size:
                continue
            # in stream mode, yield token one by one
            yield top_ids
            out_tokens.append(top_ids)
            lm_input = self.speech_embedding[top_ids].reshape(1, 1, -1)
            t2 = time.time()
            # print("len out_tokens", len(out_tokens))
            # print("llm time(s):",t2-t1)

