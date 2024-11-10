import torch
from src.model.spex_plus.model import SpexPlusModel

def spex_plus_test():
    model = SpexPlusModel(
        sizes_of_conv_kernels=[20, 80, 160],
        encoder_out_channels=256,
        tcn_in_channels=256)
    mix = torch.rand((7, 1, 3010))
    reference = torch.ones((7, 1, 600))
    out = model(mix, reference)
    for x in out:
        print(x.shape) 
    mix = torch.rand((7, 1, 3009))
    reference = torch.ones((7, 1, 600))
    out = model(mix, reference)
    for x in out:
        print(x.shape) 