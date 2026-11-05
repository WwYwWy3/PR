import os
import sys
from typing import Dict

import numpy as np

from .BASEAUG import BASEAUGMENTATION

class SEGAUG(BASEAUGMENTATION):
    def __init__(self, CONFIG: Dict):
        super().__init__(CONFIG)

    def APPLY(self, image: np.ndarray, mask: np.ndarray):
        AUGMENT_RES = self.augmentation(image = image, mask = mask)
        image_aug, mask_aug = AUGMENT_RES['image'], AUGMENT_RES['mask']

        return image_aug, mask_aug
