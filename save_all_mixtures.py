import os
import torchaudio
from src.training.my_synthetic_dataset import MySoundScapesDataset

# 配置参数
save_dir = r'E:/Waveformer-main/my_dataset/mixmic'
os.makedirs(save_dir, exist_ok=True)

dataset = MySoundScapesDataset(input_dir='E:/Waveformer-main/my_dataset')

print(f'准备保存 {len(dataset)} 个混合音频样本到 {save_dir} ...')

for i in range(len(dataset)):
    mixture, label_vector, gt = dataset[i]
    mix_path = os.path.join(save_dir, f'mixture_{i:05d}.wav')
    gt_path = os.path.join(save_dir, f'gt_{i:05d}.wav')
    torchaudio.save(mix_path, mixture, dataset.sr)
    torchaudio.save(gt_path, gt, dataset.sr)
    if i % 100 == 0:
        print(f'已保存 {i} 个样本...')

print('全部混合音频保存完成！') 