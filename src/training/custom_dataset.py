"""
自定义数据集类，支持FSD50K、MAD和UrbanSound8K数据集
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

class CustomAudioDataset(torch.utils.data.Dataset):
    """
    自定义音频数据集类。
    支持FSD50K（人声和音乐）、MAD（爆炸声）作为前景声音，
    UrbanSound8K和MAD作为背景声音。
    """

    # 自定义标签列表 - 根据你的需求调整
    _labels = [
        # FSD50K 人声和音乐类别
        "Human_voice", "Speech", "Singing", "Male_voice", "Female_voice",
        "Guitar", "Piano", "Violin", "Drums", "Bass", "Saxophone", "Trumpet",
        "Flute", "Clarinet", "Harmonica", "Accordion", "Organ", "Synthesizer",
        
        # MAD 爆炸声类别
        "Explosion", "Gunshot", "Artillery", "Mortar", "Rocket", "Grenade",
        "Bomb", "Detonation", "Blast", "Impact", "Shockwave",
        
        # 其他军事相关声音
        "Helicopter", "Jet_engine", "Tank", "Vehicle", "Radio_communication",
        "Alarm", "Siren", "Warning_signal"
    ]

    def __init__(self, input_dir, dset='', sr=None,
                 resample_rate=None, max_num_targets=1,
                 fg_source='FSD50K_MAD', bg_source='UrbanSound8K_MAD'):
        """
        构造函数，初始化数据集。
        input_dir: 数据根目录
        dset: 数据集类型（'train', 'val', 'test'）
        sr: 采样率
        resample_rate: 是否重采样到指定采样率
        max_num_targets: 每个样本最多包含的目标声音数
        fg_source: 前景声音源 ('FSD50K_MAD', 'FSD50K', 'MAD')
        bg_source: 背景声音源 ('UrbanSound8K_MAD', 'UrbanSound8K', 'MAD')
        """
        assert dset in ['train', 'val', 'test'], \
            "`dset` must be one of ['train', 'val', 'test']"
        self.dset = dset
        self.max_num_targets = max_num_targets
        self.fg_source = fg_source
        self.bg_source = bg_source
        
        # 前景声音目录（目标声音）
        if fg_source == 'FSD50K_MAD':
            # 支持多个前景声音源
            self.fg_dirs = [
                os.path.join(input_dir, 'FSD50K', dset),
                os.path.join(input_dir, 'MAD', dset)
            ]
        elif fg_source == 'FSD50K':
            self.fg_dirs = [os.path.join(input_dir, 'FSD50K', dset)]
        elif fg_source == 'MAD':
            self.fg_dirs = [os.path.join(input_dir, 'MAD', dset)]
        else:
            raise ValueError(f"Unsupported fg_source: {fg_source}")
        
        # 背景声音目录（环境噪音）
        if bg_source == 'UrbanSound8K_MAD':
            # 支持多个背景声音源
            self.bg_dirs = [
                os.path.join(input_dir, 'UrbanSound8K', dset),
                os.path.join(input_dir, 'MAD', 'background')
            ]
        elif bg_source == 'UrbanSound8K':
            self.bg_dirs = [os.path.join(input_dir, 'UrbanSound8K', dset)]
        elif bg_source == 'MAD':
            self.bg_dirs = [os.path.join(input_dir, 'MAD', 'background')]
        else:
            raise ValueError(f"Unsupported bg_source: {bg_source}")
        
        logging.info("Loading %s dataset: fg_dirs=%s bg_dirs=%s" %
                     (dset, self.fg_dirs, self.bg_dirs))

        # 获取所有样本的路径（每个样本一个文件夹，包含jams配置）
        self.samples = sorted(list(
            Path(os.path.join(input_dir, 'jams', dset)).glob('[0-9]*')))

        # 检查采样率是否匹配
        jamsfile = os.path.join(self.samples[0], 'mixture.jams')
        _, jams, _, _ = scaper.generate_from_jams(
            jamsfile, fg_path=self.fg_dirs[0], bg_path=self.bg_dirs[0])
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
        vector = torch.zeros(len(CustomAudioDataset._labels))

        for label in labels:
            idx = CustomAudioDataset._labels.index(label)
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
        # 注意：这里简化处理，实际可能需要修改jams文件以支持多源
        mixture, jams, ann_list, event_audio_list = scaper.generate_from_jams(
            jamsfile, fg_path=self.fg_dirs[0], bg_path=self.bg_dirs[0])
        
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

        # 添加音频归一化，防止振幅超出范围
        def normalize_audio(audio, target_max=0.9):
            """归一化音频到指定范围"""
            max_amp = torch.max(torch.abs(audio))
            if max_amp > 0:
                # 缩放到目标范围，留一些余量防止裁剪
                scale_factor = target_max / max_amp
                audio = audio * scale_factor
            return audio

        # 归一化混合音频和目标音频
        mixture = normalize_audio(mixture)
        gt = normalize_audio(gt)

        return mixture, label_vector, gt

def create_custom_jams_specification():
    """
    创建自定义的jams规范，用于生成新的混合配置
    """
    # 这里可以定义新的混合规范
    # 例如：如何组合FSD50K和MAD的前景声音
    pass

def prepare_custom_dataset():
    """
    准备自定义数据集的辅助函数
    """
    # 1. 下载和整理FSD50K数据
    # 2. 下载和整理MAD数据  
    # 3. 下载和整理UrbanSound8K数据
    # 4. 生成新的jams配置文件
    pass 