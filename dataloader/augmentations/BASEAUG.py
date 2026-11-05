import os
os.environ["ALBUMENTATIONS_OFFLINE"] = "1"
import sys

import numpy as np
from typing import Dict, List, Union, Any
import albumentations as A

__OPERATIONS_NAME__ = {
    "Resize": A.Resize,
    "RandomHorizontalFlip": A.HorizontalFlip,
    "RandomVerticalFlip": A.VerticalFlip,
    "GaussianBlur": A.GaussianBlur,
    "GaussNoise": A.GaussNoise,
    "Rotate": A.Rotate,
    "RandomBrightnessContrast": A.RandomBrightnessContrast
}

class BASEAUGMENTATION:
    def __init__(self, CONFIG: Dict):
        self.config: Dict = CONFIG
        self.operation: Dict = CONFIG.get("OPERATIONS", {})
        self.compose: Dict = CONFIG.get("COMPOSE", {})
        self.task: Dict = CONFIG.get("TASK", "CLASSIFICATIONS")

        op_list = []
        for op_name, op_params in self.operation.items():
            if op_name == "Resize":
                op_list.append(A.Resize(height = op_params['height'], width = op_params['width']))
                continue
            op_list.append(__OPERATIONS_NAME__[op_name](**op_params))
        op_list.append(A.Normalize(**CONFIG.get("NORMALIZE", None)))
        op_list.append(A.ToTensorV2(**CONFIG.get("TOTENSORV2", None)))

        self.augmentation = A.Compose(
            transforms=op_list, 
            bbox_params=self.compose.get("bbox_params", None) if self.task == 'DETECTION' else None,
            additional_targets=self.compose.get("additional_targets", None)
        )

    def APPLY(self):
        raise NotImplementedError("APPLY method must be implemented")

