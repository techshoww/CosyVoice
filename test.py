import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch
import os 

os.environ["export_onnx"] = "True"
os.environ["save_calib"] = "True"
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
prompt_text = '希望你以后能够做的比我还好呦。'
text = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
# text = '恭喜发财，恭喜发财'
output = []
for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech_16k, stream=True)):
    print("j['tts_speech'].shape",j['tts_speech'].shape)
    torchaudio.save('zero_shot_{}_stream_test.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    output.append(j['tts_speech'])

output = torch.cat(output, dim=1)
torchaudio.save("zero_shot_output_test.wav", output, cosyvoice.sample_rate)

# # save zero_shot spk for future usage
# assert cosyvoice.add_zero_shot_spk('希望你以后能够做的比我还好呦。', prompt_speech_16k, 'my_zero_shot_spk') is True
# for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '', '', zero_shot_spk_id='my_zero_shot_spk', stream=False)):
#     torchaudio.save('zero_shot_{}_1.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
# cosyvoice.save_spkinfo()

# # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
# for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
#     torchaudio.save('fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# # instruct usage
# output = []
# text = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
# text = '恭喜发财，恭喜发财'
# for i, j in enumerate(cosyvoice.inference_instruct2(text, '用四川话说这句话', prompt_speech_16k, stream=True)):
#     torchaudio.save('instruct_{}_test_sichuan.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
#     output.append(j['tts_speech'])
# output = torch.cat(output, dim=1)
# torchaudio.save("zero_shot_output_test_sichuan.wav", output, cosyvoice.sample_rate)

# bistream usage, you can use generator as input, this is useful when using text llm model as input
# NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length
# output = []

# def text_generator():
#     yield '收到好友从远方寄来的生日礼物，'
#     yield '那份意外的惊喜与深深的祝福'
#     yield '让我心中充满了甜蜜的快乐，'
#     yield '笑容如花儿般绽放。'
# for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}_gtest.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

#     output.append(j['tts_speech'])

# output = torch.cat(output, dim=1)
# torchaudio.save("zero_shot_output_gtest.wav", output, cosyvoice.sample_rate)
