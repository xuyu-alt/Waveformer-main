import scaper
import os
import random
import soundfile as sf
import numpy as np
import librosa
import pandas as pd
import json

print("脚本开始运行...")

# 参数
duration = 6.0
sr = 44100
bg_path = r'E:/Waveformer-main/my_dataset/background'
fg_path = r'E:/Waveformer-main/my_dataset/foreground'
outfolder = r'E:/Waveformer-main/my_dataset/mixmic_test'

print(f"背景路径: {bg_path}")
print(f"前景路径: {fg_path}")
print(f"输出路径: {outfolder}")

os.makedirs(outfolder, exist_ok=True)

# 收集背景音频
def get_wavs(root):
    wavs = []
    for r, ds, fs in os.walk(root):
        for f in fs:
            if f.endswith('.wav'):
                wavs.append(os.path.join(r, f))
    return wavs
bg_files = get_wavs(bg_path)
print(f"找到 {len(bg_files)} 个背景音频文件")

# 收集前景类别和音频
def get_fg_dict(root):
    fg_dict = {}
    for label in os.listdir(root):
        d = os.path.join(root, label)
        if os.path.isdir(d):
            files = []
            for f in os.listdir(d):
                if f.endswith('.wav'):
                    path = os.path.join(d, f)
                    try:
                        with sf.SoundFile(path) as sfh:
                            if len(sfh) / sfh.samplerate >= 3.0:
                                files.append(path)
                    except:
                        pass
            if files:
                fg_dict[label] = files
    return fg_dict
fg_files_dict = get_fg_dict(fg_path)
foreground_labels = list(fg_files_dict.keys())
print(f"有可用音频的前景标签数: {len(foreground_labels)}")

# 音频循环补齐
def loop_pad_to_duration(wav_path, target_duration, sr=44100):
    audio, file_sr = sf.read(wav_path)
    if file_sr != sr:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim == 1:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        else:
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

num_samples = 3000  # 5小时 = 3000*6秒

eval_data = []
print(f"开始生成{num_samples}个测试样本...")
for i in range(num_samples):
    if i % 500 == 0:
        print(f"进度: {i}/{num_samples}")
    sc = scaper.Scaper(duration, fg_path, bg_path)
    sc.sr = sr
    temp_bg_path = os.path.join(bg_path, 'UrbanSound8K', f'temp_bg_test_{i}.wav')
    bg_file = random.choice(bg_files)
    audio = loop_pad_to_duration(bg_file, duration, sr)
    sf.write(temp_bg_path, audio, sr)
    sc.add_background(
        label=('const', 'UrbanSound8K'),
        source_file=('const', temp_bg_path),
        source_time=('const', 0)
    )
    n_fg = random.choice([3, 4, 5])
    foreground_events = []
    for _ in range(n_fg):
        label = random.choice(foreground_labels)
        event_time = random.uniform(0, duration - 3.0)
        event_duration = random.uniform(3.0, min(5.0, duration - event_time))
        snr = random.uniform(15.0, 25.0)
        sc.add_event(
            label=('const', label),
            source_file=('choose', fg_files_dict[label]),
            source_time=('const', 0),
            event_time=('const', event_time),
            event_duration=('const', event_duration),
            snr=('const', snr),
            pitch_shift=None,
            time_stretch=None
        )
        foreground_events.append({
            'label': label,
            'start_time': event_time,
            'duration': event_duration,
            'snr': snr
        })
    jams_outfile = os.path.join(outfolder, f'mixture_{i:05d}.jams')
    audio_outfile = os.path.join(outfolder, f'mixture_{i:05d}.wav')
    sc.generate(audio_outfile, jams_outfile,
                allow_repeated_label=True,
                allow_repeated_source=True,
                reverb=None)
    if os.path.exists(temp_bg_path):
        os.remove(temp_bg_path)
    sample_info = {
        'sample_id': f'mixture_{i:05d}',
        'audio_file': f'mixture_{i:05d}.wav',
        'jams_file': f'mixture_{i:05d}.jams',
        'num_foreground_events': n_fg,
        'foreground_events': foreground_events,
        'background_label': 'UrbanSound8K',
        'duration': duration,
        'sample_rate': sr
    }
    eval_data.append(sample_info)
print(f"测试数据集生成完成！共生成{num_samples}个样本")
csv_data = []
for sample in eval_data:
    for event in sample['foreground_events']:
        csv_data.append({
            'sample_id': sample['sample_id'],
            'audio_file': sample['audio_file'],
            'jams_file': sample['jams_file'],
            'foreground_label': event['label'],
            'start_time': event['start_time'],
            'duration': event['duration'],
            'snr': event['snr'],
            'background_label': sample['background_label'],
            'total_duration': sample['duration'],
            'sample_rate': sample['sample_rate']
        })
csv_file = os.path.join(outfolder, 'test_dataset_info.csv')
pd.DataFrame(csv_data).to_csv(csv_file, index=False, encoding='utf-8-sig')
json_file = os.path.join(outfolder, 'test_dataset_info.json')
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(eval_data, f, indent=2, ensure_ascii=False)
print(f"CSV表格: {csv_file}")
print(f"JSON信息: {json_file}") 