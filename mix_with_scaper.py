import scaper
import os
import random
import soundfile as sf
import numpy as np
import librosa

# 参数
sr = 44100
duration = 6.0
bg_path = r'E:/Waveformer-main/my_dataset/background'
fg_path = r'E:/Waveformer-main/my_dataset/foreground'
outfolder = r'E:/Waveformer-main/my_dataset/mixmic'

# 确保临时背景文件目录存在
os.makedirs(os.path.join(bg_path, 'UrbanSound8K'), exist_ok=True)
os.makedirs(outfolder, exist_ok=True)

# 收集所有背景音频文件（不论长短）
bg_files = []
for root, dirs, files in os.walk(bg_path):
    for file in files:
        if file.endswith('.wav'):
            bg_files.append(os.path.join(root, file))

# 获取前景标签列表
foreground_root = r'E:/Waveformer-main/my_dataset/foreground'
foreground_labels = [d for d in os.listdir(foreground_root) if os.path.isdir(os.path.join(foreground_root, d))]

# 收集每个前景类别下可用的音频文件（长度>=3秒）
fg_files_dict = {}
for label in foreground_labels:
    label_dir = os.path.join(fg_path, label)
    files = []
    for file in os.listdir(label_dir):
        if file.endswith('.wav'):
            file_path = os.path.join(label_dir, file)
            try:
                f = sf.SoundFile(file_path)
                duration_sec = len(f) / f.samplerate
                if duration_sec >= 3.0:
                    files.append(file_path)
            except:
                pass
    if files:
        fg_files_dict[label] = files

# 只保留有可用音频的前景类别
foreground_labels = list(fg_files_dict.keys())

num_samples = 60000  # 生成100小时混合音频

def loop_pad_to_duration(wav_path, target_duration, sr=44100):
    audio, file_sr = sf.read(wav_path)
    # 若不是目标采样率，自动重采样
    if file_sr != sr:
        # librosa要求float32格式
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        # 单通道/多通道兼容
        if audio.ndim == 1:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        else:
            # 多通道分别重采样
            audio = np.stack([librosa.resample(audio[:, ch], orig_sr=file_sr, target_sr=sr) for ch in range(audio.shape[1])], axis=1)
    cur_len = len(audio)
    target_len = int(target_duration * sr)
    if cur_len >= target_len:
        return audio[:target_len]
    n_repeat = int(np.ceil(target_len / cur_len))
    if audio.ndim == 1:
        audio_long = np.tile(audio, n_repeat)
    else:
        audio_long = np.tile(audio, (n_repeat, 1))
    return audio_long[:target_len]

for i in range(num_samples):
    sc = scaper.Scaper(duration, fg_path, bg_path)
    sc.sr = sr

    # 随机选一个背景音频，循环补齐到6秒，保存为临时文件到background/UrbanSound8K/
    temp_bg_path = os.path.join(bg_path, 'UrbanSound8K', f'temp_bg_{i}.wav')
    bg_file = random.choice(bg_files)
    audio = loop_pad_to_duration(bg_file, duration, sr)
    sf.write(temp_bg_path, audio, sr)

    # 添加背景事件
    sc.add_background(
        label=('const', 'UrbanSound8K'),
        source_file=('const', temp_bg_path),
        source_time=('const', 0)
    )

    # 随机选择前景事件数
    n_fg = random.choice([3, 4, 5])
    for _ in range(n_fg):
        label = random.choice(foreground_labels)
        sc.add_event(
            label=('const', label),
            source_file=('choose', fg_files_dict[label]),
            source_time=('const', 0),
            event_time=('uniform', 0, duration),
            event_duration=('uniform', 3.0, 5.0),
            snr=('uniform', 15.0, 25.0),
            pitch_shift=None,
            time_stretch=None
        )

    jams_outfile = os.path.join(outfolder, f'mixture_{i:05d}.jams')
    audio_outfile = os.path.join(outfolder, f'mixture_{i:05d}.wav')

    sc.generate(audio_outfile, jams_outfile,
                allow_repeated_label=True,
                allow_repeated_source=True,
                reverb=None)

    # 删除临时背景文件
    if os.path.exists(temp_bg_path):
        os.remove(temp_bg_path)

print('合成完成！') 