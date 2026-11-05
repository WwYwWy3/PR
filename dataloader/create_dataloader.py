import os
import sys
from typing import Dict

from torch.utils.data import DataLoader

from .augmentations import SEGAUG
from .segmentation import SEGDATASET

def create_dataloader(
    task: str,
    mode: str,
    dataset_config: Dict, 
    augmentation_config: Dict
) -> DataLoader:
    config = dataset_config['DATASET']
    mode, task = mode.upper(), task.upper()
    if task == 'SEGMENTATION':
        dataset = SEGDATASET(config, SEGAUG(augmentation_config), mode)

    batch_size = config['BATCH_SIZE']
    if config.get('NAME') == 'BTCV' and mode != 'TRAIN':
        batch_size = 1  # 3D 评估要按 volume 拼接切片，必须逐片送入
    num_workers = config['NUM_WORKERS']
    shuf = True if mode == 'TRAIN' else False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuf, num_workers=num_workers)

def YAML2DICT(yaml_file):
    import yaml
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    DL = create_dataloader(
        task = "segmentation",
        mode = 'train',
        dataset_config = YAML2DICT('/mnt/f/PatternRecognition/codes/config/datasets/ISIC2018.yaml'),
        augmentation_config = YAML2DICT('/mnt/f/PatternRecognition/codes/config/datasets/augmentation.yaml')
    )

    for data in DL:
        image = data['image']
        mask = data['mask']

        print(f"image shape: {image.shape}; mask shape: {mask.shape}")