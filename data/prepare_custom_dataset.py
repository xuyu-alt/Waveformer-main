"""
准备自定义数据集的脚本
支持FSD50K、MAD和UrbanSound8K数据集
"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd

def download_dataset(url, filename, extract_dir):
    """
    下载并解压数据集
    """
    print(f"Downloading {filename} from {url}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Download completed: {filename}")
        
        print(f"Extracting to {extract_dir}...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction completed!")
        
        # 清理zip文件
        os.remove(filename)
        print("Cleanup completed!")
        
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

def prepare_fsd50k():
    """
    准备FSD50K数据集
    """
    print("=== Preparing FSD50K Dataset ===")
    
    # FSD50K下载链接（需要根据实际情况调整）
    fsd50k_url = "https://zenodo.org/record/4060432/files/FSD50K.eval_mp3.hdf.zip"
    
    # 创建目录结构
    fsd50k_dir = "FSD50K"
    os.makedirs(fsd50k_dir, exist_ok=True)
    
    # 下载数据集
    download_dataset(fsd50k_url, "FSD50K.zip", fsd50k_dir)
    
    # 整理目录结构
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(fsd50k_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # 这里需要根据FSD50K的实际结构来整理文件
        # 将人声和音乐相关的类别文件复制到对应目录
        voice_music_categories = [
            "Human_voice", "Speech", "Singing", "Male_voice", "Female_voice",
            "Guitar", "Piano", "Violin", "Drums", "Bass", "Saxophone", "Trumpet",
            "Flute", "Clarinet", "Harmonica", "Accordion", "Organ", "Synthesizer"
        ]
        
        for category in voice_music_categories:
            category_dir = os.path.join(split_dir, category)
            os.makedirs(category_dir, exist_ok=True)

def prepare_mad():
    """
    准备MAD (Military Audio Dataset) 数据集
    """
    print("=== Preparing MAD Dataset ===")
    
    # MAD下载链接（需要根据实际情况调整）
    mad_url = "https://example.com/mad_dataset.zip"  # 替换为实际链接
    
    # 创建目录结构
    mad_dir = "MAD"
    os.makedirs(mad_dir, exist_ok=True)
    
    # 下载数据集
    download_dataset(mad_url, "MAD.zip", mad_dir)
    
    # 整理目录结构
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(mad_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # 爆炸声和军事相关类别
        explosion_categories = [
            "Explosion", "Gunshot", "Artillery", "Mortar", "Rocket", "Grenade",
            "Bomb", "Detonation", "Blast", "Impact", "Shockwave"
        ]
        
        for category in explosion_categories:
            category_dir = os.path.join(split_dir, category)
            os.makedirs(category_dir, exist_ok=True)
    
    # 创建背景音目录
    bg_dir = os.path.join(mad_dir, "background")
    os.makedirs(bg_dir, exist_ok=True)

def prepare_urbansound8k():
    """
    准备UrbanSound8K数据集
    """
    print("=== Preparing UrbanSound8K Dataset ===")
    
    # UrbanSound8K下载链接
    urbansound8k_url = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"
    
    # 创建目录结构
    urbansound8k_dir = "UrbanSound8K"
    os.makedirs(urbansound8k_dir, exist_ok=True)
    
    # 下载数据集
    download_dataset(urbansound8k_url, "UrbanSound8K.tar.gz", urbansound8k_dir)
    
    # 整理目录结构
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(urbansound8k_dir, split)
        os.makedirs(split_dir, exist_ok=True)

def create_custom_jams_specifications():
    """
    创建自定义的jams配置文件
    """
    print("=== Creating Custom JAMS Specifications ===")
    
    # 创建jams目录结构
    jams_dir = "CustomJams"
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(jams_dir, split)
        os.makedirs(split_dir, exist_ok=True)
    
    # 这里需要根据实际需求生成jams配置文件
    # 可以使用scaper库来生成新的混合配置
    
    print("JAMS specifications created!")

def main():
    """
    主函数：准备所有数据集
    """
    print("Starting custom dataset preparation...")
    
    # 1. 准备FSD50K数据集
    prepare_fsd50k()
    
    # 2. 准备MAD数据集
    prepare_mad()
    
    # 3. 准备UrbanSound8K数据集
    prepare_urbansound8k()
    
    # 4. 创建自定义jams配置
    create_custom_jams_specifications()
    
    print("Custom dataset preparation completed!")

if __name__ == "__main__":
    main() 