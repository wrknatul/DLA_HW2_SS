import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    result_batch = {}
    result_batch["reference"] = pad_sequence([item["reference"].squeeze(0)
                                                for item in dataset_items], batch_first=True
                                              ).unsqueeze(1)
    result_batch["reference_length"] = torch.tensor([item["reference"].shape[-1] for item in dataset_items])
    mix = [item["mix"].squeeze(0) for item in dataset_items]
    target = [item["target"].squeeze(0) for item in dataset_items]
    mix_target_padded = pad_sequence(mix + target, batch_first=True).unsqueeze(1)
    result_batch["mix"] = mix_target_padded[:len(mix)]
    result_batch["target"] = mix_target_padded[len(mix):]
    result_batch["speaker_id"] = torch.tensor([item["speaker_id"] for item in dataset_items])
    result_batch["mix_path"] = [item["mix_path"] for item in dataset_items]
    if "text" in dataset_items[0]:
      result_batch["text"] = [item["text"] for item in dataset_items]
    return result_batch
