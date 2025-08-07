import torch
from cosyvoice.utils.file_utils import load_wav
import torch
import torchaudio
import numpy as np
import librosa
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import pesq
from pystoi import stoi
# from torchmetrics import SignalNoiseRatio, SignalDistortionRatio # 可选的信号级指标
import os

# --- 配置 ---
# 假设音频是语音，采样率为 16kHz
SAMPLE_RATE = 16000

# --- 辅助函数 ---

def load_and_resample(audio_path, target_sr=SAMPLE_RATE):
    """加载音频并重采样到目标采样率"""
    waveform, original_sr = torchaudio.load(audio_path)
    if original_sr != target_sr:
        # 使用 torchaudio 的 resample
        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
        waveform = resampler(waveform)
    # 确保是单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform.squeeze().numpy(), target_sr # 返回 numpy array 和 sr

def compute_mel_spectrogram(waveform, sr=SAMPLE_RATE):
    """计算梅尔频谱图"""
    # 使用 librosa 计算 Mel 频谱图
    # 参数可以根据需要调整
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, sr=sr,
        n_fft=2048, hop_length=512, n_mels=128 # 常见参数
    )
    # 转换为对数刻度 (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

def compute_mfcc(waveform, sr=SAMPLE_RATE):
    """计算 MFCC 特征"""
    # 使用 librosa 计算 MFCC
    mfccs = librosa.feature.mfcc(
        y=waveform, sr=sr,
        n_mfcc=13, # 通常取前13个系数
        n_fft=2048, hop_length=512
    )
    return mfccs

# --- 评估方法 ---

def compare_mel_spectrogram_mse(ref_audio_path, gen_audio_path):
    """
    比较两个音频的梅尔频谱图 MSE
    """
    print("--- Comparing Mel-Spectrogram MSE ---")
    try:
        ref_waveform, sr1 = load_and_resample(ref_audio_path, SAMPLE_RATE)
        gen_waveform, sr2 = load_and_resample(gen_audio_path, SAMPLE_RATE)

        if sr1 != sr2:
             print(f"Warning: Sample rates differ after resampling. sr1={sr1}, sr2={sr2}. Using {SAMPLE_RATE}.")

        ref_mel = compute_mel_spectrogram(ref_waveform, sr1)
        gen_mel = compute_mel_spectrogram(gen_waveform, sr2)

        # 简单的 MSE 计算 (要求频谱图尺寸完全一致)
        # 如果长度不同，需要先对齐或截断/填充
        min_len = min(ref_mel.shape[1], gen_mel.shape[1])
        ref_mel_trunc = ref_mel[:, :min_len]
        gen_mel_trunc = gen_mel[:, :min_len]

        mse = np.mean((ref_mel_trunc - gen_mel_trunc) ** 2)
        print(f"Mel-Spectrogram MSE: {mse:.4f}")
        # 通常 MSE 越小越好，但没有绝对的“好”阈值
        return mse
    except Exception as e:
        print(f"Error in Mel-Spectrogram MSE comparison: {e}")
        return None

def compare_mfcc_dtw(ref_audio_path, gen_audio_path):
    """
    比较两个音频的 MFCC 特征 DTW 距离
    """
    print("--- Comparing MFCC DTW Distance ---")
    try:
        ref_waveform, sr1 = load_and_resample(ref_audio_path, SAMPLE_RATE)
        gen_waveform, sr2 = load_and_resample(gen_audio_path, SAMPLE_RATE)

        if sr1 != sr2:
             print(f"Warning: Sample rates differ after resampling. sr1={sr1}, sr2={sr2}. Using {SAMPLE_RATE}.")

        ref_mfcc = compute_mfcc(ref_waveform, sr1) # Shape: (n_mfcc, time_steps)
        gen_mfcc = compute_mfcc(gen_waveform, sr2)

        # DTW 需要 (time_steps, features) 的形状
        ref_mfcc_transposed = ref_mfcc.T
        gen_mfcc_transposed = gen_mfcc.T

        # 使用 fastdtw 计算距离和路径
        distance, path = fastdtw(ref_mfcc_transposed, gen_mfcc_transposed, dist=euclidean)
        # Normalize by path length to make it more comparable
        normalized_distance = distance / len(path) if len(path) > 0 else float('inf')
        print(f"MFCC DTW Distance: {distance:.2f}")
        print(f"MFCC DTW Normalized Distance: {normalized_distance:.4f}")
        # 距离越小，相似度越高
        return distance, normalized_distance
    except Exception as e:
        print(f"Error in MFCC DTW comparison: {e}")
        return None, None

def compare_pesq(ref_audio_path, gen_audio_path):
    """
    比较两个语音音频的 PESQ 分数 (窄带模式)
    注意：PESQ 对采样率和音频类型有要求
    """
    print("--- Comparing PESQ Score ---")
    target_sr_pesq = 16000 # PESQ 通常使用 16kHz 或 8kHz
    try:
        # PESQ 通常用于窄带语音 (8kHz)，但宽带 (16kHz) 也支持 (PESQ 4.0.6+)
        # 这里使用 16kHz 宽带模式 ('wb')
        ref_waveform, sr1 = load_and_resample(ref_audio_path, target_sr_pesq)
        gen_waveform, sr2 = load_and_resample(gen_audio_path, target_sr_pesq)

        if sr1 != sr2 or sr1 != target_sr_pesq:
             print(f"Warning: Sample rates after resampling for PESQ. sr1={sr1}, sr2={sr2}. Expected {target_sr_pesq}.")

        # 确保是 numpy 数组且是 float32
        ref_wav = ref_waveform.astype(np.float32)
        gen_wav = gen_waveform.astype(np.float32)

        # 检查长度，PESQ 可能对极短音频有要求
        if len(ref_wav) == 0 or len(gen_wav) == 0:
            raise ValueError("Audio is empty after loading/resampling.")

        # 调用 pesq 库计算分数 (窄带 'nb'=8kHz, 宽带 'wb'=16kHz)
        # PESQ 分数范围通常在 -0.5 到 4.5 之间，4.5 表示最好
        pesq_score = pesq.pesq(target_sr_pesq, ref_wav, gen_wav, 'wb') # 使用宽带模式
        print(f"PESQ Score (wideband @ {target_sr_pesq}Hz): {pesq_score:.4f}")
        # 分数越高，感知质量越好
        return pesq_score
    except Exception as e:
        # PESQ 对输入非常敏感，容易出错
        print(f"Error in PESQ comparison (might be due to audio characteristics or PESQ library): {e}")
        # 可以尝试窄带模式作为备选
        try:
            sr_nb = 8000
            ref_waveform_nb, _ = load_and_resample(ref_audio_path, sr_nb)
            gen_waveform_nb, _ = load_and_resample(gen_audio_path, sr_nb)
            ref_wav_nb = ref_waveform_nb.astype(np.float32)
            gen_wav_nb = gen_waveform_nb.astype(np.float32)
            if len(ref_wav_nb) > 0 and len(gen_wav_nb) > 0:
                 pesq_score_nb = pesq.pesq(sr_nb, ref_wav_nb, gen_wav_nb, 'nb')
                 print(f"  Fallback - PESQ Score (narrowband @ {sr_nb}Hz): {pesq_score_nb:.4f}")
                 return pesq_score_nb
            else:
                 print("  Fallback failed: Audio too short for narrowband PESQ.")
        except Exception as e_nb:
             print(f"  Fallback PESQ also failed: {e_nb}")
        return None

# --- 主函数 ---
def main():
    # 请替换为你的参考音频和生成音频的实际路径
    reference_audio_file = "zero_shot_output.wav"
    generated_audio_file = "zero_shot_output_test.wav"

    if not os.path.exists(reference_audio_file):
        print(f"Error: Reference audio file '{reference_audio_file}' not found.")
        return
    if not os.path.exists(generated_audio_file):
        print(f"Error: Generated audio file '{generated_audio_file}' not found.")
        return

    print(f"Comparing '{reference_audio_file}' and '{generated_audio_file}'\n")

    # 1. 梅尔频谱图 MSE
    mse_score = compare_mel_spectrogram_mse(reference_audio_file, generated_audio_file)

    # 2. MFCC DTW 距离
    dtw_dist, dtw_norm_dist = compare_mfcc_dtw(reference_audio_file, generated_audio_file)

    # 3. PESQ 分数 (适用于语音)
    pesq_score = compare_pesq(reference_audio_file, generated_audio_file)

    print("\n--- Summary of Results ---")
    if mse_score is not None:
        print(f"Mel-Spectrogram MSE: {mse_score:.4f} (Lower is better)")
    if dtw_dist is not None:
        print(f"MFCC DTW Distance: {dtw_dist:.2f} (Lower is better)")
        print(f"MFCC DTW Normalized Distance: {dtw_norm_dist:.4f} (Lower is better)")
    if pesq_score is not None:
        print(f"PESQ Score (wb): {pesq_score:.4f} (Higher is better, up to 4.5)")
    print("--------------------------")

if __name__ == "__main__":
    main()