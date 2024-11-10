from torch import Tensor
import torch.nn as nn
from typing import List

class MixSpeechDecoder(nn.Module):
    def __init__(
            self,
            sizes_of_conv_kernels: List[int],
            encoder_out_channels: int):
        super(MixSpeechDecoder, self).__init__()
        sizes_of_conv_kernels.sort()
        min_kernel_size = sizes_of_conv_kernels[0]
        self.deconv_decoders = nn.ModuleList(_create_deconv_decoder(
            kernel_size=conv_kernel_size,
            encoder_out_channels=encoder_out_channels,
            min_kernel_size=min_kernel_size) 
            for conv_kernel_size in sizes_of_conv_kernels)
            
    def forward(self, input) -> List[Tensor]:
        decoded_results = [decoder(input) for decoder in self.deconv_decoders]
        return list(decoded_results)

def _create_deconv_decoder(
        kernel_size: int,
        encoder_out_channels: int,
        min_kernel_size: int) -> nn.ConvTranspose1d:
    return nn.ConvTranspose1d(
        in_channels=encoder_out_channels,
        out_channels=1,
        kernel_size=kernel_size,
        stride=min_kernel_size//2,
        padding=0,
        dilation=1)