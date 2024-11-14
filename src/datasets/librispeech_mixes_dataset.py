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
from src.utils import ROOT_PATH
from src.datasets.mixer import MixtureGenerator, LibriSpeechSpeakerFiles

logger = logging.getLogger(__name__)

LIBRISPEECH_URLS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


class LibrispeechMixed(BaseDataset):
    """Dataset for creating and managing Librispeech mixture data."""

    def __init__(
        self, 
        part: str,
        out_folder: Path = ROOT_PATH / "data" / "datasets" / "librispeech" / "mixes",
        data_dir: Optional[Union[str, Path]] = None,
        snr_levels: List[int] = [-5, 5],
        num_workers: int = 2,
        mixer_audio_length: int = 10,
        *args, 
        **kwargs
    ):
        if part not in LIBRISPEECH_URLS and part != 'train_all':
            raise ValueError(f"Invalid dataset part: {part}")

        self.out_folder = out_folder
        self._data_dir = Path(data_dir) if data_dir else ROOT_PATH / "data" / "datasets" / "librispeech"
        self._data_dir.mkdir(exist_ok=True, parents=True)
        self.snr_levels = snr_levels
        self.num_workers  = num_workers
        self.mixer_audio_length = mixer_audio_length
        self.speaker_to_id = {}
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_part(self, part: str) -> None:
        arch_path = self._data_dir / f"{part}.tar.gz"
        
        download_file(LIBRISPEECH_URLS[part], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)

        librispeech_dir = self._data_dir / "LibriSpeech"
        for fpath in librispeech_dir.iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))


    def __getitem__(self, ind: int) -> Dict[str, Any]:
        data_dict = self._index[ind]
        
        return {
            "reference": self.load_object(data_dict["reference"]),
            "mix": self.load_object(data_dict["mix"]),
            "target": self.load_object(data_dict["target"]),
            "speaker_id": data_dict["speaker_id"]
            }

    def _get_or_load_index(self, part: str) -> List[Dict[str, Any]]:
        index_path = Path(self._data_dir / f"{part}-mixed-index.json")
    
        if index_path.exists():
            return json.loads(index_path.read_text())

        index = self._create_index(part)
        index_path.write_text(json.dumps(index, indent=2))
        return index

    def _create_mixes(self, split_dir: Path, out_folder: Path) -> None:
        speakers_files = []
        for speaker_id in os.listdir(split_dir):
            speaker_dir_path = os.path.join(split_dir, speaker_id)
            if os.path.isdir(speaker_dir_path):
                speakers_files.append(LibriSpeechSpeakerFiles(speaker_id, split_dir, "*.flac"))
        
        mixer = MixtureGenerator(
            speakers_files=speakers_files,
            out_folder=out_folder
        )
        mixer.generate_mixes(
            self.snr_levels, self.num_workers, update_steps=100, audioLen=self.mixer_audio_length
        )

    def _create_index(self, part: str) -> List[Dict[str, Any]]:
        mixes_out_folder = Path(self.out_folder)

        if not mixes_out_folder.exists():
            split_dir = self._data_dir / part
            if not split_dir.exists():
                self._load_part(part)
            self._create_mixes(split_dir, mixes_out_folder)

        refs = np.array(sorted(glob(str(mixes_out_folder / '*-ref.wav'))), dtype=object)
        mixes = np.array(sorted(glob(str(mixes_out_folder / '*-mixed.wav'))), dtype=object)
        targets = np.array(sorted(glob(str(mixes_out_folder / '*-target.wav'))), dtype=object)
  
        index = []
        for i in range(len(refs)):
            speaker_id = int(refs[i].split('/')[-1].split('_')[0])
            if speaker_id not in self.speaker_to_id:
                self.speaker_to_id[speaker_id] = len(self.speaker_to_id)
            entry = {
                "reference": str(refs[i]),
                "mix": str(mixes[i]),
                "target": str(targets[i]),
                "speaker_id": self.speaker_to_id[speaker_id] 
            }
            index.append(entry)

        return index 