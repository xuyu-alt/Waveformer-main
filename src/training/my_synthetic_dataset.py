import os
import random
from pathlib import Path
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as AT

class MySoundScapesDataset(torch.utils.data.Dataset):
    """
    适配你自定义数据集的Dataset类，不依赖jams，直接动态采样混合。
    """
    _labels = [
        "Crying_and_sobbing", "Explosion", "Gunshot", "Human_voice",
        "Music", "Screaming", "Speech", "Yell"
    ]

    def __init__(self, input_dir, sr=44100, resample_rate=None, n_samples=10000, max_num_targets=3):
        self.input_dir = input_dir
        self.sr = sr
        self.max_num_targets = max_num_targets
        self.n_samples = n_samples
        self.duration = 6.0  # seconds
        self.fg_dir = os.path.join(input_dir, 'foreground')
        self.bg_dir = os.path.join(input_dir, 'background', 'UrbanSound8K')
        # 收集前景音频
        self.fg_files = {label: list(Path(os.path.join(self.fg_dir, label)).glob('*.wav')) for label in self._labels}
        # 收集所有背景音频
        self.bg_files = list(Path(self.bg_dir).rglob('*.wav'))
        # 重采样器
        if resample_rate is not None:
            self.resampler = AT.Resample(sr, resample_rate)
            self.sr = resample_rate
        else:
            self.resampler = lambda a: a

    def __len__(self):
        return self.n_samples

    def _get_label_vector(self, labels):
        vector = torch.zeros(len(self._labels))
        for label in labels:
            idx = self._labels.index(label)
            vector[idx] = 1
        return vector

    def __getitem__(self, idx):
        # 1. 随机选一个背景音
        bg_path = random.choice(self.bg_files)
        bg_wave, bg_sr = torchaudio.load(str(bg_path))
        if bg_wave.shape[0] > 1:
            bg_wave = bg_wave.mean(dim=0, keepdim=True)  # 转单声道
        if bg_sr != self.sr:
            bg_wave = torchaudio.functional.resample(bg_wave, bg_sr, self.sr)
        # 截取或补零到6秒
        target_len = int(self.sr * self.duration)
        if bg_wave.shape[1] < target_len:
            pad = target_len - bg_wave.shape[1]
            bg_wave = torch.nn.functional.pad(bg_wave, (0, pad))
        else:
            bg_wave = bg_wave[:, :target_len]
        # 2. 随机选3~5个前景事件
        n_fg = random.choice([3, 4, 5])
        fg_labels = random.sample(self._labels, n_fg)
        label_vector = self._get_label_vector(fg_labels)
        mixture = bg_wave.clone()
        gt = torch.zeros_like(mixture)
        for label in fg_labels:
            # 随机选一个前景音频
            fg_file = random.choice(self.fg_files[label])
            fg_wave, fg_sr = torchaudio.load(str(fg_file))
            if fg_wave.shape[0] > 1:
                fg_wave = fg_wave.mean(dim=0, keepdim=True)
            if fg_sr != self.sr:
                fg_wave = torchaudio.functional.resample(fg_wave, fg_sr, self.sr)
            # 随机采样事件持续时间和起始时间
            event_duration = random.uniform(3.0, 5.0)
            event_time = random.uniform(0.0, self.duration - event_duration)
            event_len = int(event_duration * self.sr)
            event_start = int(event_time * self.sr)
            # 截取或补零前景音
            if fg_wave.shape[1] < event_len:
                fg_wave = torch.nn.functional.pad(fg_wave, (0, event_len - fg_wave.shape[1]))
            else:
                fg_wave = fg_wave[:, :event_len]
            # 随机采样SNR
            snr_db = random.uniform(15.0, 25.0)
            # 计算缩放系数
            bg_segment = bg_wave[:, event_start:event_start+event_len]
            fg_rms = fg_wave.pow(2).mean().sqrt()
            bg_rms = bg_segment.pow(2).mean().sqrt() + 1e-8
            desired_fg_rms = bg_rms * (10 ** (snr_db / 20.0))
            if fg_rms > 0:
                fg_wave = fg_wave * (desired_fg_rms / fg_rms)
            # 混合到背景
            mixture[:, event_start:event_start+event_len] += fg_wave
            gt[:, event_start:event_start+event_len] += fg_wave
        # 防止溢出，归一化到[-1, 1]
        mixture = torch.clamp(mixture, -1.0, 1.0)
        gt = torch.clamp(gt, -1.0, 1.0)
        return mixture, label_vector, gt 