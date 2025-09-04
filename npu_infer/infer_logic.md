
# 推理逻辑
## frontend_zero_shot

QwenTokenizer 将 prompt text 和 input text 转为 text token 输入llm  
speech_tokenizer_model onnx  将 prompt audio 转换为 prompt_speech_token  
prompt audio 经过采样后使用`feat_extractor: !name:matcha.utils.audio.mel_spectrogram`提取 speech_feat  
_extract_spk_embedding  campplus onnx 将 将 prompt audio 转换为 固定尺寸的 flow_embedding
```python
def frontend_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, resample_rate, zero_shot_spk_id):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        if zero_shot_spk_id == '':
            prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
            prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k)
            speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
            speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
            if resample_rate == 24000:
                # cosyvoice2, force speech_feat % speech_token = 2
                token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
                speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len
                speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
            embedding = self._extract_spk_embedding(prompt_speech_16k)
            model_input = {'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
                           'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
                           'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                           'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                           'llm_embedding': embedding, 'flow_embedding': embedding}
        else:
            model_input = self.spk2info[zero_shot_spk_id]
        model_input['text'] = tts_text_token
        model_input['text_len'] = tts_text_token_len
        return model_input
```

## tts
llm 异步输出speech token
等待只有当剩余的speech token 数 >= this_token_hop_len + self.flow.pre_lookahead_len 时才执行token2wav
输入token2wav 的 speech token 需要做长度对齐
prompt token 也做了长度对齐(这里又bug，输入llm的speech token没有做对齐,可能是音质下降的原因)
需要取一些前面计算过多 speech token 做长度对齐

长度太短的话音质会下降

## llm  
输入：  
```python
text: torch.Tensor,
            text_len: torch.Tensor,
            prompt_text: torch.Tensor,
            prompt_text_len: torch.Tensor,
            prompt_speech_token: torch.Tensor,
            prompt_speech_token_len: torch.Tensor,
            embedding: torch.Tensor,
            sampling: int = 25,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            uuid: str = '',
```
qwen2 model.embed_tokens 将 text_token转为 embedding   
speech_embedding 将 prompt_speech_token 转为 prompt_speech_token_emb  
llm_embedding.sos_eos_emb llm_embedding.task_id_emb  text embeding prompt_speech_token_emb 拼接  
llm prefill 推理  
llm_decoder  
llm 逐token推理 + llm_decoder + sample top id + speech_embedding 编码 循环  



## token2wav

输入:  
```python
this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     token_offset=token_offset,
                                                     uuid=this_uuid,
                                                     stream=stream,
                                                     finalize=False)
```


通过 flow.input_embedding 将 prompt speech token 和 speech token转为 token_embedding
flow_embedding 过 F.normalize

调用 flow模型
flow模型输出 tts_mel ，取出 tts_mel 中有效的长度
hift_cache_mel 拼接 tts_mel 
调用 hift模型
调用 fade_in_out 做拼接
取出最后的 mel_cache_len 长度留作下次 hift_cache_mel 拼接 tts_mel
### flow 

### hift


## 注意  
llm 和 token2wav 输入的 prompt speech token 相同，但是之后转 embedding的处理不同：  
llm中是使用 speech_embedding 将 prompt_speech_token 转为 prompt_speech_token_emb   维度是 896  
token2wav中是使用 flow.input_embedding 将 prompt speech token转为 token_embedding  维度是 512  