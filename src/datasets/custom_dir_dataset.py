import json
import logging
import os
import shutil
from pathlib import Path
from glob import glob
from typing import Dict, List, Optional, Union, Any

import numpy as np
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

class CustomDirDataset(BaseDataset):
    """Dataset for creating and managing Librispeech mixture data."""

    def __init__(
        self, 
        dataset_folder: Path,
        num_workers: int = 2,
        *args, 
        **kwargs
    ):

        self.dataset_folder = dataset_folder
        self.num_workers  = num_workers
        self.speaker_to_id = {}
        index = self._create_index()

        super().__init__(index, *args, **kwargs)


    def __getitem__(self, ind: int) -> Dict[str, Any]:
        data_dict = self._index[ind]
        reference = None
        if data_dict["reference"] is not None:
            reference = self.load_object(data_dict["reference"])

        target = None
        if data_dict["target"] is not None:
            target = self.load_object(data_dict["target"])
        
        return {
            "reference": reference,
            "mix": self.load_object(data_dict["mix"]),
            "target": target,
            "speaker_id": ind
            }

    def _create_index(self) -> List[Dict[str, Any]]:

        mixes = np.array(sorted(glob(str(self.dataset_folder / Path('audios') / Path('mix') / Path('FirstSpeakerID*')))), dtype=object)
  
        index = []
        for i in range(len(mixes)):
            fileName = 'FirstSpeakerID' + str(i + 1) + '_*'
            reference = str(mixes[i])
            target = str(mixes[i])
            if len(glob(str(self.dataset_folder / Path('audios') / Path('s1') / fileName))) > 0:
                reference = glob(str(self.dataset_folder / Path('audios') / Path('s1') / fileName))[0]
            
            if len(glob(str(self.dataset_folder / Path('audios') / Path('s2') / fileName))) > 0:
                print(glob(str(self.dataset_folder / Path('audios') / Path('s2') / fileName)))
                target = glob(str(self.dataset_folder / Path('audios') / Path('s2') / fileName))[0]
            entry = {

                "reference": reference,
                "mix": str(mixes[i]),
                "target": target,
                "speaker_id": i
            }
            index.append(entry)

        return index 
    
# CustomDirDataset(dataset_folder = "/Users/molutan/Downloads/DirectoryWithUtterances")