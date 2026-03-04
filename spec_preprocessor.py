import os
import random
import numpy as np
import librosa
import librosa.display
import pywt
import torch
import time
import sys
import cv2
import warnings
from pathlib import Path

from pyexpat import features

warnings.filterwarnings('ignore')

# ====================== 1. 全局配置与固定随机种子 ======================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BRANCH_TWO_DIR = os.path.join(BASE_DIR, "Branch Two")
os.makedirs(BRANCH_TWO_DIR,exist_ok=True)
SPEC_INFO = os.path.join(BRANCH_TWO_DIR, "preprocess_info.npz")
test_info = os.path.join(BASE_DIR, "test_info.npz")

# 音频参数
SAMPLE_RATE = 44100
DURATION = 1  # 统一音频时长为1秒
N_FFT = 2048  # 与4444.py保持一致
HOP_LENGTH = 512  # 与4444.py保持一致
WIN_LENGTH = 1024  # 与4444.py保持一致
N_MELS = 128  # Mel滤波器组数
WAVELET_TYPE = 'morl'  # 小波类型


# ======================  多频谱提取与融合（核心修改部分） ======================
def extract_fourier_spectrum(y, sr):
    """傅里叶频谱：使用librosa.stft，未归一化"""
    stft = librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window='hamming')
    fourier_spec = np.abs(stft)
    fourier_spec = librosa.amplitude_to_db(fourier_spec, ref=np.max)  # 未归一化
    return fourier_spec


def get_cwt_scale(signal, sample_rate):
    """动态计算小波变换尺度（基于信号能量分布的5%-95%频率范围）"""
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1 / sample_rate)
    magnitude = np.abs(fft)
    # 计算能量分布
    energy = magnitude ** 2
    cumulative_energy = np.cumsum(energy) / (np.sum(energy) + (1e-10))
    idx_min = np.argmax(cumulative_energy > 0.05)  # 5%能量处
    idx_max = np.argmax(cumulative_energy > 0.95)  # 95%能量处
    f_min = max(freqs[idx_min], 20)  # 至少20Hz
    f_max = min(freqs[idx_max], sample_rate / 2)  # 不超过奈奎斯特频率
    center_freq = pywt.central_frequency(WAVELET_TYPE)  # 小波中心频率
    # 从频率计算尺度
    scale_max = center_freq * sample_rate / f_min
    scale_min = center_freq * sample_rate / f_max
    scale = np.logspace(np.log10(scale_min), np.log10(scale_max), 128)  # 生成128个尺度
    return scale[::-1]


def extract_wavelet_spectrum(y, sr):
    """小波频谱：动态尺度，取对数幅值再转dB"""
    scale = get_cwt_scale(y, sr)
    cwtmat, _ = pywt.cwt(y, scales=scale, wavelet=WAVELET_TYPE, sampling_period=1 / sr)
    cwt_abs = np.abs(cwtmat)
    cwt_log = np.log1p(cwt_abs)  # 取对数幅值
    wavelet_spec = librosa.amplitude_to_db(cwt_log, ref=np.max)  # 转换为dB
    return wavelet_spec


def gauss_fbank(sr, n_filters, sigma_coef, frequencies):
    """自定义高斯滤波器组"""
    fmin, fmax = 0, sr // 2
    # 转换到Mel尺度
    mel_min = librosa.hz_to_mel(fmin)
    mel_max = librosa.hz_to_mel(fmax)
    # 计算中心频率
    mel_centers = np.linspace(mel_min, mel_max, n_filters)
    centers = librosa.mel_to_hz(mel_centers)
    # 计算标准差
    sigma_spacing = (mel_max - mel_min) / (n_filters - 1) if n_filters > 1 else (mel_max - mel_min)
    hz_sigmas = []
    for mel_center in mel_centers:
        hz_low = librosa.mel_to_hz(mel_center - sigma_spacing / 2)
        hz_high = librosa.mel_to_hz(mel_center + sigma_spacing / 2)
        sigma_hz = (hz_high - hz_low) * sigma_coef
        hz_sigmas.append(sigma_hz)
    sigmas = np.array(hz_sigmas)

    # 生成高斯滤波器
    def gaussian_filterbank(frequencies):
        frequencies = np.asarray(frequencies)
        filters = np.zeros((n_filters, len(frequencies)))
        for i in range(n_filters):
            filters[i] = np.exp(-0.5 * ((frequencies - centers[i]) / sigmas[i]) ** 2)
        return filters

    filters = gaussian_filterbank(frequencies)
    return filters, centers, sigmas


def extract_mel_spectrum(y, sr):
    """Mel频谱：自定义高斯滤波器组"""
    # 先计算STFT幅值
    stft = librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window='hamming')
    fft_spec = np.abs(stft)
    # 生成频率轴
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    # 获取高斯滤波器组
    filters, centers, sigmas = gauss_fbank(sr, N_MELS, 0.5, frequencies)
    # 应用滤波器组
    mel_spec = np.dot(filters, fft_spec)
    # # 计算二阶差分
    # mel_delta = librosa.feature.delta(mel_spec)
    # mel_delta2 = librosa.feature.delta(mel_delta)  # 二阶差分
    # 转换为dB
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def get_spec(audio_path, labels_, target_size):
    spec=[]
    labels=[]
    try:
        for path, label in zip(audio_path, labels_):
            audio, sr = librosa.load(path, sr = SAMPLE_RATE)
            f = extract_fourier_spectrum(audio, SAMPLE_RATE)
            w = extract_wavelet_spectrum(audio, SAMPLE_RATE)
            m = extract_mel_spectrum(audio,SAMPLE_RATE)

            f_ = cv2.resize(f, target_size, interpolation=cv2.INTER_LINEAR)
            w_ = cv2.resize(w, target_size, interpolation=cv2.INTER_LINEAR)
            m_ = cv2.resize(m, target_size, interpolation=cv2.INTER_LINEAR)
            merged = np.stack([f_,w_,m_], axis=0).astype(np.float32)

            spec.append(merged)
            labels.append(label)

            p = Path(path)
            print(f'已处理音频:{p.stem}')

    except Exception as e:
        print(f" 处理失败: {audio_path} | {e}")
    return spec, labels

def get_test_spec(audios,labels,target_size):
    spec=[]
    label_=[]
    try:
        for audio,label in zip(audios,labels):
            f = extract_fourier_spectrum(audio, SAMPLE_RATE)
            w = extract_wavelet_spectrum(audio, SAMPLE_RATE)
            m = extract_mel_spectrum(audio, SAMPLE_RATE)

            f_ = cv2.resize(f, target_size, interpolation=cv2.INTER_LINEAR)
            w_ = cv2.resize(w, target_size, interpolation=cv2.INTER_LINEAR)
            m_ = cv2.resize(m, target_size, interpolation=cv2.INTER_LINEAR)
            merged = np.stack([f_, w_, m_], axis=0).astype(np.float32)

            spec.append(merged)
            label_.append(label)
    except Exception as e:
        print(f" 处理失败:  {e}")
    return spec, label_



# ====================== 主程序执行 ======================
if __name__ == "__main__":
    print("===== 数据预处理程序 =====")
    print(f"程序启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 训练用数据
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir,'Branch One','audio_info.npz')
    data = np.load(data_path)
    audio_paths = data['audio_paths']
    labels_ = data['label']

    #测试用数据
    test_data = np.load(os.path.join(base_dir, 'test_data.npz'))
    test_audio = test_data['features']
    test_paths = test_data['paths']
    test_labels= test_data['labels']

    # 频谱提取与融合
    istrain = False
    if istrain:
        spec, label = get_spec(audio_paths, labels_, target_size=(128, 128))
        if len(spec) == 0:
            print("错误：频谱处理失败，程序终止！")
            sys.exit(1)
        np.savez(
            SPEC_INFO,
            audio_paths=audio_paths,
            spec_array=spec,
            labels=label)
        print(f"\n预处理结果已保存至 {SPEC_INFO}")

    else:
        spec, label = get_test_spec(test_audio, test_labels, target_size=(128, 128))
        if len(spec) == 0:
            print("错误：频谱处理失败，程序终止！")
            sys.exit(1)
        np.savez(
            test_info,
            test_paths=test_paths,
            audio_array=test_audio,
            spec_array=spec,
            labels=label)
        print(f"\n预处理结果已保存至 {test_info}")

    print(f"程序结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
