import torch
import torch.nn as nn
from typing import Tuple, List

class TwinSpeechEncoder(nn.Module):
    def __init__(
            self,
            sizes_of_conv_kernels: List[int],
            out_channels: int):
        super(TwinSpeechEncoder, self).__init__()
        sizes_of_conv_kernels.sort()
        min_kernel_size = sizes_of_conv_kernels[0]
        self.conv_encoders = nn.ModuleList(_create_conv_encoder(
            kernel_size=conv_kernel_size,
            out_channels=out_channels,
            min_kernel_size=min_kernel_size) 
            for conv_kernel_size in sizes_of_conv_kernels)
            
    def forward(self, input_audio) -> Tuple[torch.Tensor]:
        encoded_inputs = [encoder(input_audio) for encoder in self.conv_encoders]
        return encoded_inputs

def _create_conv_encoder(
        kernel_size: int,
        out_channels: int,
        min_kernel_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConstantPad1d((0, kernel_size - min_kernel_size), 0),
        nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=min_kernel_size//2,
            padding=0,
            dilation=1),
        nn.ReLU())