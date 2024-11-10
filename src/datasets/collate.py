import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
        """

    result_batch = {}

    # Pad sequences and stack them into a batch
    # Assuming data_object contains sequences of varying lengths
    mix_audios = []
    reference_audios = []
    target_audios = []

    for item in dataset_items:

        mix_audios.append(item["mix"].squeeze(0))
        reference_audios.append(item["reference"].squeeze(0))
        target_audios.append(item["target"].squeeze(0))  
    
    return {
        "mix": pad_sequence(mix_audios, batch_first=True),
        "reference": pad_sequence(reference_audios, batch_first=True),
        "target": pad_sequence(target_audios, batch_first=True)
    }
