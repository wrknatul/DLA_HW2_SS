import json
import logging
import os
import shutil
from pathlib import Path
from glob import glob

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm
import numpy as np

from src.datasets.base_dataset import BaseDataset
from src.utils import ROOT_PATH
from src.datasets.mixture_generator import MixtureGenerator, LibriSpeechSpeakerFiles


logger = logging.getLogger(__name__)

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}

def files_collector(audios_dir, audioTemplate="*-norm.wav", with_text=False):
    result = []
    for speaker_id in os.listdir(audios_dir):
        speaker_dir_path = os.path.join(audios_dir, speaker_id)
        if os.path.isdir(speaker_dir_path):
            result.append(LibriSpeechSpeakerFiles(speaker_id, audios_dir, audioTemplate, with_text))
    return result

class LibreSpeechMixer(BaseDataset):
    def __init__(self, part, mixer=None, data_dir=None, *args, **kwargs):
        assert part in URL_LINKS or part == 'train_all'
        return
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
        self._data_dir = Path(data_dir)
        index = self._get_or_load_index(part, mixer)
        super().__init__(index, *args, **kwargs)


    def _load_part(self, part):
        self._data_dir.mkdir(exist_ok=True, parents=True)
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        download_file(URL_LINKS[part], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))


    def _get_or_load_index(self, part, mixer_config):
        index_path = Path(mixer_config.get("index_path", self._data_dir / f"{part}-mixed-index.json"))
        if index_path.exists():
            logger.info(f"Found existing index at {index_path}")
            with index_path.open() as f:
                index = json.load(f)
        else:
            logger.info(f"Creating index at {index_path}")
            index = self._create_index(part, mixer_config)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index


    def _create_mixes(self, split_dir, out_folder, mixer_config):
        if mixer_config is None:
            mixer_config = {}
        speakers_files = files_collector(split_dir, "*.flac")
        
        nfiles = mixer_config.get("nfiles", 5000)
        test = mixer_config.get("test", False)
        mixer = MixtureGenerator(speakers_files, out_folder, nfiles=nfiles, test=test)
        
        mixer.generate_mixes(
            snr_levels=mixer_config.get("snr_levels", [-5, 5]),
            num_workers=mixer_config.get("num_workers", 2),
            update_steps=100,
            trim_db=mixer_config.get("trim_db", None),
            vad_db=mixer_config.get("vad_db", 20),
            audioLen=mixer_config.get("audioLen", 3)
        )


    def _create_index(self, part, mixer_config):
        mixes_out_folder = Path(mixer_config.get("out_folder", self._data_dir / f"{part}-mixed"))
        logger.info(f"mixes folder {mixes_out_folder}")
        if not mixes_out_folder.exists():
            split_dir = self._data_dir / part
            # Download Librispeech if it doesn't exist
            if not split_dir.exists():
                self._load_part(part)
            self._create_mixes(split_dir, mixes_out_folder, mixer_config)
        refs = np.array(sorted(glob(os.path.join(mixes_out_folder, '*-ref.wav'))), dtype=object)
        mixes = np.array(sorted(glob(os.path.join(mixes_out_folder, '*-mixed.wav'))), dtype=object)
        targets = np.array(sorted(glob(os.path.join(mixes_out_folder, '*-target.wav'))), dtype=object)
        speaker_ids = np.array([int(ref.split("/")[-1].split("_")[0]) for ref in refs])
        
        with_text = mixer_config.get("with_text", False)
        if with_text:
            texts = np.array(sorted(glob(os.path.join(mixes_out_folder, '*.txt'))), dtype=object)

        sorted_indices = speaker_ids.argsort()
        refs = refs[sorted_indices]
        mixes = mixes[sorted_indices]
        targets = targets[sorted_indices]
        speaker_ids = speaker_ids[sorted_indices]
        speaker_ids_mapped = [0] + np.cumsum((speaker_ids[1:] > speaker_ids[:-1]).astype(int)).tolist()
        if with_text:
            texts = texts[sorted_indices]
        
        if with_text:
            zipped_data = zip(refs, mixes, targets, speaker_ids_mapped, texts)
        else:
            zipped_data = zip(refs, mixes, targets, speaker_ids_mapped)
        index = []
        for tuple in zipped_data:
            index_row = {
                "reference": tuple[0],
                "mix": tuple[1],
                "target": tuple[2],
                "speaker_id": tuple[3]
            }
            if with_text:
                index_row["text"] = tuple[-1]
            index.append(index_row)
        return index