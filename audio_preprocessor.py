import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import random


class AudioPreprocessor:
    """音频预处理与数据增强类"""

    def __init__(self, hop_length=512, win_length=1024, n_fft=2048, n_mel=128,
                 sample_rate=22050, duration=1):
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.n_mel = n_mel
        self.sample_rate = sample_rate
        self.duration = duration
        self.label_encoder = None  # 保存标签编码器


    '-------------STFT-------------'

    def STFT(self, y):
        y_ = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length, window='hamming')
        return y_  # 返回矩阵

    '-------------预加重--------------'

    def pre_emphasis(self, y):
        y = librosa.effects.preemphasis(y)
        return librosa.effects.percussive(y)

    '--------------去噪---------------'

    def denoising(self, y):
        # 防止输入信号全为零
        if np.allclose(y, 0):
            return y

        # Step 1. 带通滤波
        def bandpass(y, sr, low=200, high=5000):
            from scipy.signal import butter, filtfilt
            nyq = 0.5 * sr
            low = max(low, 1)  # 避免0频率
            high = min(high, nyq * 0.99)  # 略低于奈奎斯特频率
            b, a = butter(4, [low / nyq, high / nyq], btype='band')
            return filtfilt(b, a, y)

        y1 = bandpass(y, self.sample_rate)

        # Step 2. 小波去噪
        import pywt
        coeffs = pywt.wavedec(y1, 'db8', level=4)
        for i in range(1, len(coeffs)):
            max_coeff = max(abs(coeffs[i]))
            threshold = 0.04 * max_coeff if max_coeff > 1e-10 else 1e-10
            coeffs[i] = pywt.threshold(coeffs[i], threshold)
        y2 = pywt.waverec(coeffs, 'db8')

        return y2

    '-------------静音消除-------------'

    def remove_silence(self, y):
        if np.max(np.abs(y)) < 1e-10:
            return y
        y_, _ = librosa.effects.trim(y, ref=np.max)
        intervals = librosa.effects.split(y_, top_db=30, ref=np.max)
        y_after_split = librosa.effects.remix(np.array(y_), np.array(intervals))
        return y_after_split

    '------------响度增强------------'

    def louder(self, y):
        max_abs = np.max(np.abs(y))
        if max_abs < 1e-10:
            return y
        y = y / max_abs
        target_rms = 0.05
        rms = np.sqrt(np.mean(y ** 2)) + 1e-10
        gain = target_rms / rms
        y = y * gain
        y = np.clip(y, -1.0, 1.0)
        return y

    '------------音频数据增强------------'

    def augment_audio(self, y, AUGMENT_FACTOR):
        augmented = []
        augmented.append(y)

        # 1. 音量调整
        gain_factor = np.random.uniform(0.7, 1.3)
        y_vol = y * gain_factor
        y_vol = np.clip(y_vol, -1.0, 1.0).astype(np.float32)
        augmented.append(y_vol)

        # # 2. 音调偏移
        # pitch_shift = np.random.uniform(-1, 1)
        # y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift).astype(np.float32)
        # augmented.append(y_pitch)

        # 3. 时间拉伸
        rate = np.random.uniform(0.9, 1.1)
        y_stretch = librosa.effects.time_stretch(y, rate=rate).astype(np.float32)
        augmented.append(y_stretch)

        # 4. 添加轻微噪声
        noise_amp = 0.05 * np.random.uniform() * np.max(y)
        y_noise = y + noise_amp * np.random.normal(size=y.shape[0]).astype(np.float32)
        y_noise = np.clip(y_noise, -1.0, 1.0)
        augmented.append(y_noise)

        # 去重并限制数量
        unique_augmented = []
        seen = set()
        for item in augmented:
            item_tuple = tuple(item)
            if item_tuple not in seen:
                seen.add(item_tuple)
                unique_augmented.append(item)

        if len(unique_augmented) < AUGMENT_FACTOR:
            while len(unique_augmented) < AUGMENT_FACTOR:
                unique_augmented.append(random.choice(unique_augmented))
        else:
            unique_augmented = random.sample(unique_augmented, AUGMENT_FACTOR)

        return unique_augmented

    '------------统一音频长度------------'

    def normalize_length(self, y, sr):
        target_length = self.sample_rate * self.duration
        if len(y) < target_length:
            pad_amount = target_length - len(y)
            pad_before = pad_amount // 2
            pad_after = pad_amount - pad_before
            y_processed = np.pad(y, (pad_before, pad_after), mode='constant')
        else:
            start = (len(y) - target_length) // 2
            y_processed = y[start:start + target_length]

        if self.sample_rate != self.sample_rate:
            y_processed = librosa.resample(
                y_processed,
                orig_sr=self.sample_rate,
                target_sr=sr)

        return y_processed, sr

    '------------完整音频处理流程------------'

    def process_audio(self, y, sr):
        y_pre_emp = self.pre_emphasis(y)
        y_denoised = self.denoising(y_pre_emp)
        y_remove_silence = self.remove_silence(y_denoised)
        y_louder = self.louder(y_remove_silence)
        y_nor, sr = self.normalize_length(y_louder, sr)
        return y_nor.astype(np.float32), sr

    '------------加载、处理音频并保存------------'

    def load_process_save_and_extract(self, branch_one_file, data_dirs, save_data=True):
        # 创建保存目录（在processed_data下）
        processed_full_dir = os.path.join(branch_one_file, 'processed_data')
        os.makedirs(processed_full_dir, exist_ok=True)
        for label in data_dirs.keys():
            label_dir = os.path.join(processed_full_dir, label)
            os.makedirs(label_dir, exist_ok=True)

        features = []
        labels = []
        audio_path = []
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']

        for label, dir_path in data_dirs.items():  # 键值对，标签与路径
            print(f"加载并处理 {dir_path} 目录下的数据...")
            # glob获取该路径下所有文件；suffix:文件后缀名；lower转为小写
            audio_files = [f for f in Path(dir_path).glob('*') if f.suffix.lower() in audio_extensions]
            # 保存原始处理音频（在processed_data下）
            processed_label_dir = os.path.join(processed_full_dir, label)
            os.makedirs(processed_label_dir,exist_ok=True)

            for file in audio_files:
                try:
                    audio, sr = librosa.load(file, sr=self.sample_rate)
                    processed_audio, sr = self.process_audio(audio, sr)
                    augmented_audios = self.augment_audio(processed_audio, 4)

                    # 保存增强音频（在processed_data下）
                    for i, aug_audio in enumerate(augmented_audios):  # 0代表原音频
                        aug_audio_normalized, sr = self.normalize_length(aug_audio, sr)
                        aug_output_filename = f"processed_{file.stem}_processed_{i}.wav"
                        aug_output_path = os.path.join(processed_label_dir, aug_output_filename)
                        sf.write(aug_output_path, aug_audio_normalized, sr)
                        features.append(aug_audio_normalized)
                        labels.append(label)
                        audio_path.append(aug_output_path)
                        print(f"已处理并保存增强音频: {aug_output_path}")

                except Exception as e:
                    print(f"处理 {file.name} 时出错: {e}")

        # 标签编码
        X = np.array(features)
        y = np.array(labels)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.label_encoder = le

        # 保存路径与标签文件
        audio_info = os.path.join(branch_one_file, 'audio_info.npz')
        np.savez(audio_info, audio_paths=audio_path, label=y)

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42)  # random_state只是一种数据洗牌的模式，可换成不同数字

        # 保存处理后的数据和标签编码器（在本文件目录下）
        if save_data:
            data_path = os.path.join(branch_one_file, 'processed_data.npz')
            encoder_path = os.path.join(branch_one_file, 'label_encoder.pkl')
            np.savez(data_path, X_train=X_train, X_val=X_val,
                     y_train=y_train, y_val=y_val)
            with open(encoder_path, 'wb') as f:
                pickle.dump(le, f)
            print(f"处理后的数据已保存为 {data_path}")
            print(f"标签编码器已保存为 {encoder_path}")

        return (X_train, X_val, y_train, y_val), le

    def test_process(self, save_path, data_dirs):
        features = []
        labels = []
        paths = []
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']


        for label, dir_path in data_dirs.items():  # 键值对，标签与路径
            print(f"加载并处理 {dir_path} 目录下的数据...")
            # glob获取该路径下所有文件；suffix:文件后缀名；lower转为小写
            audio_files = [f for f in Path(dir_path).glob('*') if f.suffix.lower() in audio_extensions]

            for file in audio_files:
                try:
                    path = os.path.abspath(file)
                    audio, sr = librosa.load(file, sr=self.sample_rate)
                    processed_audio, sr = self.process_audio(audio, sr)
                    features.append(processed_audio)
                    labels.append(label)
                    paths.append(path)
                    print(f"已处理并保存音频: {os.path.basename(file)}")
                except Exception as e:
                    print(f"处理 {file.name} 时出错: {e}")

        np.savez(save_path, features=features, labels=labels, paths=paths)
        return


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    is_train = False

    if is_train:
        hollow_path = os.path.join(base_dir,'data','hollow')
        not_hollow_path = os.path.join(base_dir, 'data', 'not_hollow')
        branch_one_file = os.path.join(base_dir, 'Branch One')
        os.makedirs(branch_one_file, exist_ok=True)
        # 定义原始数据目录
        data_dirs = {
            'hollow': hollow_path,
            'not_hollow': not_hollow_path}  # 请替换为实际数据目录路径
        # 初始化处理器
        preprocessor = AudioPreprocessor(sample_rate=44100, duration=1)
        # 加载、处理数据并保存
        _, _ = preprocessor.load_process_save_and_extract(branch_one_file, data_dirs)

    else:
        hollow_path = os.path.join(base_dir, 'data', 'test_data', 'hollow')
        not_hollow_path = os.path.join(base_dir, 'data', 'test_data', 'not_hollow')
        # test_dir = os.path.join(base_dir,'test')
        # os.makedirs(test_dir)
        save_path = os.path.join(base_dir, 'test_data.npz')

        # 定义原始数据目录
        data_dirs = {
            'hollow': hollow_path,
            'not_hollow': not_hollow_path}  # 请替换为实际数据目录路径

        # 初始化处理器
        preprocessor = AudioPreprocessor(sample_rate=44100, duration=1)

        # 加载、处理数据并保存
        preprocessor.test_process(save_path, data_dirs)


    # npz=np.load('processed_data/processed_data.npz')
    # for key,value in npz.items():
    #     print(f"{key}: {value.shape}")
    #     print(value)
    # print(pickle.load(open('processed_data/label_encoder.pkl', 'rb')))
