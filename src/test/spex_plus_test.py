import torch
import torchaudio
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
    mix = torchaudio.load("/Users/ivansidor/DLA_HW2_SS/src/test/mix.wav")[0][None, :]
    ref = torchaudio.load("/Users/ivansidor/DLA_HW2_SS/src/test/ref.wav")[0][None, :]
    out = model(mix, ref)
    print(out)
    
    