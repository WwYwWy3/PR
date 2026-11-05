import json
import os
import sys
from typing import Dict

from torch.utils.data import Dataset
import yaml

class BASEDATASET(Dataset):
    def __init__(self, dataset_config: Dict, mode: str = 'train'):
        super().__init__()
        self.dataset_name = dataset_config.get("NAME", None)
        if self.dataset_name is None:
            raise ValueError("NAME must be specified in dataset config")
        self.config = dataset_config # overall config
        self.data_list_format = dataset_config.get("FORMAT", None) # way of dataset organized json or yaml
        self.data_list = self.__preprocess__(mode.upper()) # preprocess data list

    def __process_data_list__(self, data_list):
        if self.data_list_format == 'json':
            with open(data_list, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
        elif self.data_list_format == 'yaml':
            with open(data_list, 'r', encoding='utf-8') as f:
                data_list = yaml.safe_load(f)
                # print(data_list[0])
        else:
            raise ValueError(f"{self.data_list_format} not in ['json', 'yaml']")

        if not isinstance(data_list, list):
            raise ValueError(f"{data_list} should contain a list of samples")
        return data_list

    def __preprocess__(self, mode: str):
        data_list = self.config.get(mode, None) # get split file json or yaml
        if not os.path.exists(data_list) and mode == 'VALIDATION':
            data_list = self.config.get('TEST', None)
        # print(data_list)
        return self.__process_data_list__(data_list)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index):
        raise NotImplementedError("__getitem__ must be Implemented")

