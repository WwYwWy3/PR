import os
import sys
from typing import Dict, Union

from PIL import Image
import numpy as np
import torch

from .BASEDS import BASEDATASET
from .augmentations import SEGAUG

class SEGDATASET(BASEDATASET):
    def __init__(self, dataset_config: Dict, aug: SEGAUG, mode: str = 'train'):
        super().__init__(dataset_config, mode)
        self.augmentation = aug
        image_info, mask_info = self.config.get("IMAGE_INFO", None), self.config.get("MASK_INFO", None)

        self.image_type, self.mask_type = image_info.get("TYPE", None), mask_info.get("TYPE", None)
        self.image_mode, self.mask_mode = image_info.get("MODE", None), mask_info.get("MODE", None)

        if self.image_type == 'npz':
            self.image_key = image_info.get("KEY", None)
        if self.mask_type == 'npz':
            self.mask_key = mask_info.get("KEY", None)

        self.label_mapping = self.config.get("LABEL_MAPPING")
        if self.dataset_name == 'BTCV':
            self.data_list = self.__preprocess_BTCV__(self.data_list)

    def __BTCV_windowing__(self, frame: np.ndarray):
        # window op for CT image
        pass

    def __preprocess_BTCV__(self, data_list):
        image_mask_pair = []
        for idx, sample in enumerate(data_list):
            image_dir = sample['image']
            label_dir = sample['label']
            # 按切片编号排序并按文件名配对，避免 image/label 顺序错位
            for name in sorted(os.listdir(image_dir), key=lambda item: int(os.path.splitext(item)[0])):
                image_mask_pair.append({
                    'image': os.path.join(image_dir, name),
                    'label': os.path.join(label_dir, name),
                    'idx': idx,
                })
        return image_mask_pair

    def __load_image__(self, image_path, is_mask: bool = True):
        if is_mask:
            if self.mask_type == 'npz': 
                image_np = np.load(image_path)[self.mask_key].astype(np.int8)
            else:
                image_np = np.array(Image.open(image_path).convert(self.mask_mode), dtype = np.int8)
        else:
            if self.image_type == 'npz':
                image_np = np.load(image_path)[self.image_key].astype(np.float32)
            else:
                image_np = np.array(Image.open(image_path).convert(self.image_mode), dtype = np.float32)
        return image_np

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index):
        image_mask_pair = self.data_list[index]
        image_np = self.__load_image__(image_mask_pair['image'], is_mask = False)
        mask_np = self.__load_image__(image_mask_pair['label'], is_mask = True)
        image_aug, mask_aug = self.augmentation.APPLY(image_np, mask_np)

        return {'image': image_aug, 'mask': mask_aug, 'idx': image_mask_pair.get('idx', 0)}

def YAML2DICT(yaml_file):
    import yaml
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == '__main__':
    dataset_config = YAML2DICT('/mnt/f/PatternRecognition/codes/config/datasets/BTCV.yaml')['DATASET']
    # aug = SEGAUG(YAML2DICT('/mnt/f/PatternRecognition/codes/config/datasets/augmentation.yaml'))

    DS = SEGDATASET(dataset_config, None, 'train')
