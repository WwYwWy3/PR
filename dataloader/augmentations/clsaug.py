import os
import sys
from typing import Dict

import numpy as np

from .BASEAUG import BASEAUGMENTATION


class CLSAUG(BASEAUGMENTATION):
    def __init__(self, CONFIG: Dict):
        super().__init__(CONFIG)
        
    def APPLY(self, image: np.ndarray):
        return self.augmentation(image = image, mask = None)

    