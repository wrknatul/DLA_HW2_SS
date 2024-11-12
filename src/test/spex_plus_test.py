import torch
from src.model.spex_plus.model import SpexPlusModel

def spex_plus_test():
    model = SpexPlusModel(
        sizes_of_conv_kernels=[20, 80, 160],
        encoder_out_channels=256,
        resnet_in_channels=256,
        resnet_out_channels=512,
        speaker_encoder_out_channels=256,
        tcn_in_channels=256,
        tcn_middle_channels=512,
        tcn_out_channels=256,
        num_tcn_blocks=8,
        num_speakers=100)
    mix = torch.ones((3, 1, 3313))
    reference = torch.ones((3, 1, 600))
    out = model(mix, reference)
    
    