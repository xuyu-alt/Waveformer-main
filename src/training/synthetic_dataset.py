"""
Torch dataset object for synthetically rendered spatial data.
"""

import os
import json
import random
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scaper
import torch
import torchaudio
import torchaudio.transforms as AT
from random import randrange

class FSDSoundScapesDataset(torch.utils.data.Dataset):  # type: ignore
    """
    FSD Sound Scapes数据集的自定义Dataset类。
    该类用于动态合成带有目标声音和背景声音的混合音频样本，
    并生成相应的标签和目标音频，供神经网络训练和评估使用。
    """

    # 所有支持的目标声音标签
    _labels = [
    "Acoustic_guitar", "Applause", "Bark", "Bass_drum",
    "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet",
    "Computer_keyboard", "Cough", "Cowbell", "Double_bass",
    "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping",
    "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire",
    "Harmonica", "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow",
    "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter",
    "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone",
    "Trumpet", "Violin_or_fiddle", "Writing"]

    def __init__(self, input_dir, dset='', sr=None,
                 resample_rate=None, max_num_targets=1):
        """
        构造函数，初始化数据集。
        input_dir: 数据根目录
        dset: 数据集类型（'train', 'val', 'test'）
        sr: 采样率
        resample_rate: 是否重采样到指定采样率
        max_num_targets: 每个样本最多包含的目标声音数
        """
        assert dset in ['train', 'val', 'test'], \
            "`dset` must be one of ['train', 'val', 'test']"
        self.dset = dset
        self.max_num_targets = max_num_targets
        # 前景声音目录（目标声音）
        self.fg_dir = os.path.join(input_dir, 'FSDKaggle2018/%s' % dset)
        # 背景声音目录（环境噪音）
        if dset in ['train', 'val']:
            self.bg_dir = os.path.join(
                input_dir,
                'TAU-acoustic-sounds/'
                'TAU-urban-acoustic-scenes-2019-development')
        else:
            self.bg_dir = os.path.join(
                input_dir,
                'TAU-acoustic-sounds/'
                'TAU-urban-acoustic-scenes-2019-evaluation')
        logging.info("Loading %s dataset: fg_dir=%s bg_dir=%s" %
                     (dset, self.fg_dir, self.bg_dir))

        # 获取所有样本的路径（每个样本一个文件夹，包含jams配置）
        self.samples = sorted(list(
            Path(os.path.join(input_dir, 'jams', dset)).glob('[0-9]*')))

        # 检查采样率是否匹配
        jamsfile = os.path.join(self.samples[0], 'mixture.jams')
        _, jams, _, _ = scaper.generate_from_jams(
            jamsfile, fg_path=self.fg_dir, bg_path=self.bg_dir)
        _sr = jams['annotations'][0]['sandbox']['scaper']['sr']
        assert _sr == sr, "Sampling rate provided does not match the data"

        # 是否需要重采样
        if resample_rate is not None:
            self.resampler = AT.Resample(sr, resample_rate)
            self.sr = resample_rate
        else:
            self.resampler = lambda a: a
            self.sr = sr

    def _get_label_vector(self, labels):
        """
        根据标签列表生成多热(one-hot)向量。
        labels: 当前样本包含的目标声音标签列表
        返回: 多热向量（长度为类别数）
        """
        vector = torch.zeros(len(FSDSoundScapesDataset._labels))

        for label in labels:
            idx = FSDSoundScapesDataset._labels.index(label)
            assert vector[idx] == 0, "Repeated labels"
            vector[idx] = 1 

        return vector

    def __len__(self):
        """
        返回数据集的样本数量。
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取第idx个样本，动态合成混合音频、标签和目标音频。
        返回：(混合音频, 标签向量, 目标音频)
        """
        sample_path = self.samples[idx]
        jamsfile = os.path.join(sample_path, 'mixture.jams')

        # 使用Scaper根据jams配置动态合成混合音频
        mixture, jams, ann_list, event_audio_list = scaper.generate_from_jams(
            jamsfile, fg_path=self.fg_dir, bg_path=self.bg_dir)
        # ann_list: 每个事件的注释信息，event_audio_list: 每个事件的音频
        # 0号事件为背景，1号及以后为前景事件
        isolated_events = {}
        for e, a in zip(ann_list, event_audio_list[1:]):
            # e[2]为事件标签
            isolated_events[e[2]] = a
        # 读取真实目标事件标签
        gt_events = list(pd.read_csv(
            os.path.join(sample_path, 'gt_events.csv'), sep='\t')['label'])

        # 转为torch格式并调整通道顺序
        mixture = torch.from_numpy(mixture).permute(1, 0)
        mixture = self.resampler(mixture.to(torch.float))

        # 根据数据集类型选择目标标签
        if self.dset == 'train':
            # 训练集：随机选1~max_num_targets个目标
            labels = random.sample(gt_events, randrange(1,self.max_num_targets+1))
        elif self.dset == 'val':
            # 验证集：顺序选取目标
            labels = gt_events[:idx%self.max_num_targets+1]
        elif self.dset == 'test':
            # 测试集：选前max_num_targets个目标
            labels = gt_events[:self.max_num_targets]
        label_vector = self._get_label_vector(labels)

        # 合成目标音频（所有目标事件的音频相加）
        gt = torch.zeros_like(
            torch.from_numpy(event_audio_list[1]).permute(1, 0))
        for l in labels:
            gt = gt + torch.from_numpy(isolated_events[l]).permute(1, 0)
        gt = self.resampler(gt.to(torch.float))

        return mixture, label_vector, gt #, jams

def tensorboard_add_sample(writer, tag, sample, step, params):
    """
    将一个样本（混合、标签、目标、输出）添加到tensorboard，便于可视化。
    """
    if params['resample_rate'] is not None:
        sr = params['resample_rate']
    else:
        sr = params['sr']
    resample_rate = 16000 if sr > 16000 else sr

    m, l, gt, o = sample
    m, gt, o = (
        torchaudio.functional.resample(_, sr, resample_rate).cpu()
        for _ in (m, gt, o))

    def _add_audio(a, audio_tag, axis, plt_title):
        for i, ch in enumerate(a):
            axis.plot(ch, label='mic %d' % i)
            writer.add_audio(
                '%s/mic %d' % (audio_tag, i), ch.unsqueeze(0), step, resample_rate)
        axis.set_title(plt_title)
        axis.legend()

    # 动态获取标签列表
    from pathlib import Path
    fg_root = Path('my_dataset/foreground')
    labels_87 = [d.name for d in fg_root.iterdir() if d.is_dir()]
    labels_87.sort()

    for b in range(m.shape[0]):
        label = []
        for i in range(len(l[b, :])):
            if l[b, i] == 1 and i < len(labels_87):
                label.append(labels_87[i])

        # 添加波形图到tensorboard
        rows = 3 # input, output, gt
        fig = plt.figure(figsize=(10, 2 * rows))
        axes = fig.subplots(rows, 1, sharex=True)
        _add_audio(m[b], '%s/sample_%d/0_input' % (tag, b), axes[0], "Mixed")
        _add_audio(o[b], '%s/sample_%d/1_output' % (tag, b), axes[1], "Output (%s)" % label)
        _add_audio(gt[b], '%s/sample_%d/2_gt' % (tag, b), axes[2], "GT (%s)" % label)
        writer.add_figure('%s/sample_%d/waveform' % (tag, b), fig, step)

def tensorboard_add_metrics(writer, tag, metrics, label, step):
    """
    将评估指标添加到tensorboard。
    """
    # Be robust to different metric key names
    possible_keys = [
        'scale_invariant_signal_noise_ratio',
        'signal_noise_ratio',
        'si_snr',
        'snr'
    ]
    key = None
    for k in possible_keys:
        if k in metrics:
            key = k
            break
    if key is None:
        return

    vals = np.asarray(metrics[key])

    # 检查数据是否为空
    if len(vals) > 0:
        writer.add_histogram('%s/%s' % (tag, 'SI-SNRi'), vals, step)
    else:
        # 如果数据为空，添加一个默认值
        writer.add_histogram('%s/%s' % (tag, 'SI-SNRi'), [0.0], step)

    # 动态获取标签列表
    from pathlib import Path
    fg_root = Path('my_dataset/foreground')
    labels_87 = [d.name for d in fg_root.iterdir() if d.is_dir()]
    labels_87.sort()

    label_names = []
    for _ in label:
        idx = torch.argmax(_).item()
        if idx < len(labels_87):
            label_names.append(labels_87[idx])
        else:
            label_names.append(f"unknown_{idx}")
    
    for l, v in zip(label_names, vals):
        writer.add_histogram('%s/%s' % (tag, l), v, step)
