import os
import json
import logging
from pathlib import Path
from typing import List
import numpy as np
import torch
import scaper
import torchaudio
import torchaudio.transforms as AT
import tempfile
import random
import soundfile as sf
import warnings
from scaper.core import ScaperWarning

# Silence Scaper clipping warnings to avoid noisy logs
warnings.filterwarnings("ignore", category=ScaperWarning)

class FolderJAMSDataset(torch.utils.data.Dataset):
    """
    从一个目录中读取 Scaper 生成的 mixture_*.jams（及对应wav），
    按给定的 label_list 过滤前景事件，构造 label 向量与目标音频 gt。
    会修复jams中的临时背景文件路径，替换为一个真实可用的背景音频文件，
    用前景白名单替换无效/不可读的前景源，失败时重试若干次。
    """
    def __init__(self, root_dir: str, label_list: List[str], sr: int = 44100, resample_rate: int = None,
                 bg_root: str = 'my_dataset/background', fg_root: str = 'my_dataset/foreground', max_retries: int = 3,
                 duration: float = 6.0):
        self.root_dir = root_dir
        self.label_list = label_list
        self.sr = sr
        self.duration = duration
        self.resample_rate = resample_rate
        self.jams_files = sorted(str(p) for p in Path(root_dir).glob('mixture_*.jams'))
        assert len(self.jams_files) > 0, f"No jams files found in {root_dir}"
        
        # 验证数据目录
        if not Path(bg_root).exists():
            logging.warning(f"Background directory {bg_root} does not exist")
        if not Path(fg_root).exists():
            logging.warning(f"Foreground directory {fg_root} does not exist")
        if resample_rate is not None:
            self.resampler = AT.Resample(sr, resample_rate)
            self.out_sr = resample_rate
        else:
            self.resampler = None
            self.out_sr = sr
        # 预收集背景文件
        self.bg_files = [str(p) for p in Path(bg_root).rglob('*.wav')]
        assert len(self.bg_files) > 0, f"No background wavs found in {bg_root}"
        self.max_retries = max_retries
        # 建立前景可读文件白名单（按标签）
        self.fg_valid = {}
        fg_root_path = Path(fg_root)
        for label in self.label_list:
            label_dir = fg_root_path / label
            files = []
            if label_dir.is_dir():
                for p in label_dir.glob('*.wav'):
                    try:
                        with sf.SoundFile(str(p)):
                            files.append(str(p))
                    except Exception:
                        continue
            self.fg_valid[label] = files

    def __len__(self):
        return len(self.jams_files)

    def _get_label_vector(self, labels: List[str]) -> torch.Tensor:
        vec = torch.zeros(len(self.label_list))
        for l in labels:
            if l in self.label_list:
                idx = self.label_list.index(l)
                vec[idx] = 1
        return vec

    def _render_from_jams_fixed(self, jamsfile: str):
        # 读取并修正 jams
        with open(jamsfile, 'r', encoding='utf-8') as f:
            jams_obj = json.load(f)
        scaper_box = jams_obj['annotations'][0]['sandbox']['scaper']
        
        # 降低背景音量
        if 'bg_spec' in scaper_box:
            for bg in scaper_box['bg_spec']:
                if len(bg) > 3:  # 确保有音量参数
                    bg[3] = 0.3  # 降低背景音量到0.3
        
        # 降低前景音量
        if 'fg_spec' in scaper_box:
            for fg in scaper_box['fg_spec']:
                if len(fg) > 4:  # 确保有音量参数
                    fg[4] = 0.2  # 降低前景音量到0.2
        
        # 背景替换为真实文件
        bg_path = random.choice(self.bg_files)
        try:
            # 确保路径格式正确
            bg_path = bg_path.replace('\\', '/')
            # 替换背景规格中的文件路径
            scaper_box['bg_spec'][0][1][1] = bg_path
            # 替换data数组中的背景文件路径
            for item in jams_obj['annotations'][0]['data']:
                if item['value']['role'] == 'background':
                    item['value']['source_file'] = bg_path
                    break
        except Exception as e:
            pass
        
        # 用前景白名单替换每个事件的 source_file 列表
        if 'fg_spec' in scaper_box:
            for ev in scaper_box['fg_spec']:
                try:
                    ev_label = None
                    if isinstance(ev[0], list) and len(ev[0]) >= 2 and ev[0][0] == 'const':
                        ev_label = ev[0][1]
                    if ev_label in self.fg_valid and len(self.fg_valid[ev_label]) > 0:
                        ev[1] = ['choose', list(self.fg_valid[ev_label])]
                    else:
                        ev[1] = ['const', bg_path]
                        if isinstance(ev[4], list) and len(ev[4]) >= 3 and ev[4][0] == 'uniform':
                            ev[4][1] = 0.01
                            ev[4][2] = 0.02
                except Exception:
                    continue
        
        # 写入临时文件渲染
        with tempfile.NamedTemporaryFile('w', suffix='.jams', delete=False) as tf:
            json.dump(jams_obj, tf)
            temp_jams_path = tf.name
        try:
            # 直接调用，不使用 validate 参数
            return scaper.generate_from_jams(temp_jams_path)
        except Exception as e:
            raise
        finally:
            try:
                os.remove(temp_jams_path)
            except Exception:
                pass

    def _dummy_sample(self):
        T = int(self.duration * (self.out_sr or self.sr))
        mixture_t = torch.zeros(T, 1).permute(1, 0)  # [C,T]
        gt = torch.zeros_like(mixture_t)
        label_vec = torch.zeros(len(self.label_list))
        return mixture_t, label_vec, gt

    def __getitem__(self, idx: int):
        jamsfile = self.jams_files[idx]
        last_err = None
        for _ in range(self.max_retries):
            try:
                mixture, jams, ann_list, event_audio_list = self._render_from_jams_fixed(jamsfile)
                # 解析标签：仅依据实际前景事件 ann_list，避免使用 sandbox['fg_labels'] 造成全1
                fg_labels = []
                try:
                    for e in ann_list[1:]:  # ann_list[0] 通常为背景
                        try:
                            # 修复：正确解析JAMS文件中的标签
                            if isinstance(e, dict):
                                lbl = e.get('value', {}).get('label', None)
                            else:
                                lbl = e[2]
                            if isinstance(lbl, str):
                                fg_labels.append(lbl)
                        except Exception:
                            continue
                except Exception:
                    fg_labels = []
                sel_labels = [l for l in set(fg_labels) if l in self.label_list]
                label_vec = self._get_label_vector(sel_labels)
                # 异常标签监控
                if label_vec.sum().item() == 0:
                    logging.warning(f"No valid foreground labels parsed for {jamsfile}; parsed={fg_labels[:5]}... total={len(fg_labels)}")
                if label_vec.sum().item() == len(self.label_list):
                    logging.warning(f"All labels set to 1 for {jamsfile}; check JAMS parsing and label_list mapping")
                # mixture -> torch [C,T]
                mixture_t = torch.from_numpy(mixture).permute(1, 0).to(torch.float)
                if self.resampler is not None:
                    mixture_t = self.resampler(mixture_t)
                # 构造gt
                if len(event_audio_list) > 1:
                    gt = torch.zeros_like(mixture_t)
                    for e, a in zip(ann_list[1:], event_audio_list[1:]):
                        try:
                            if isinstance(e, dict):
                                lbl = e.get('value', {}).get('label', None)
                            else:
                                lbl = e[2]
                        except Exception:
                            lbl = None
                        if isinstance(lbl, str) and lbl in self.label_list:
                            a_t = torch.from_numpy(a).permute(1, 0).to(torch.float)
                            if self.resampler is not None:
                                a_t = self.resampler(a_t)
                            gt += a_t
                else:
                    gt = torch.zeros_like(mixture_t)
                
                # 添加音频归一化，防止振幅超出范围
                def normalize_audio(audio, target_max=0.7):
                    """归一化音频到指定范围"""
                    max_amp = torch.max(torch.abs(audio))
                    if max_amp > 0:
                        # 缩放到目标范围，留一些余量防止裁剪
                        scale_factor = target_max / max_amp
                        audio = audio * scale_factor
                    return audio

                # 若无有效前景或gt接近静音，则跳过本样本重试
                try:
                    if label_vec.sum().item() <= 0:
                        raise RuntimeError("empty_label_after_filter")
                    if gt.abs().max().item() < 1e-6 or (gt.pow(2).mean().item() < 1e-8):
                        raise RuntimeError("silent_gt_after_filter")
                except Exception:
                    raise
                # 归一化混合音频和目标音频
                mixture_t = normalize_audio(mixture_t)
                gt = normalize_audio(gt)
                
                # 检查数据是否有效
                if (gt == 0).all():
                    print(f'WARNING: Ground truth is all zeros for {jamsfile}')
                if (mixture_t == 0).all():
                    print(f'WARNING: Mixed audio is all zeros for {jamsfile}')
                
                return mixture_t, label_vec, gt
            except Exception as err:
                last_err = err
                continue
        # 多次失败后，尝试直接加载同名 mixture wav 作为回退
        logging.error(f"Failed to render from JAMS for index {idx} after {self.max_retries} retries: {last_err}")
        logging.error(f"JAMS file: {jamsfile}")
        wav_path = jamsfile.replace('.jams', '.wav')
        try:
            if os.path.exists(wav_path):
                audio, sr = torchaudio.load(wav_path)
                if audio.dim() == 2 and audio.size(0) > 1:
                    audio = torch.mean(audio, dim=0, keepdim=True)
                elif audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                # 重采样
                if self.resampler is not None and sr is not None and sr != self.out_sr:
                    audio = self.resampler(audio)
                # 归一化到约0.7
                max_amp = torch.max(torch.abs(audio))
                if max_amp > 0:
                    audio = audio * (0.7 / max_amp)
                # 标签置为依据 ann_list 的多热（若解析也失败，则全零），gt 置零
                try:
                    with open(jamsfile, 'r', encoding='utf-8') as f:
                        jams_obj = json.load(f)
                    ann_list = jams_obj['annotations'][0]['data']
                    fg_labels = []
                    for e in ann_list[1:]:
                        try:
                            lbl = e['value']['label'] if isinstance(e, dict) else e[2]
                            if isinstance(lbl, str):
                                fg_labels.append(lbl)
                        except Exception:
                            continue
                    sel_labels = [l for l in set(fg_labels) if l in self.label_list]
                    label_vec = self._get_label_vector(sel_labels)
                except Exception:
                    label_vec = torch.zeros(len(self.label_list))
                gt = torch.zeros_like(audio)
                return audio, label_vec, gt
        except Exception as _fallback_err:
            logging.error(f"Fallback load wav failed: {wav_path} err={_fallback_err}")
        # 最终回退到哑样本
        logging.warning(f"Returning dummy sample for index {idx} - this will cause loss=0!")
        return self._dummy_sample() 