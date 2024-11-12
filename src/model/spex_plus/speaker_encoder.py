import torch 
import torch.nn as nn
from src.model.spex_plus.resnet_block import ResNetBlock

class SpeakerEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            resnet_in_channels: int,
            resnet_out_channels: int):
        super(SpeakerEncoder, self).__init__()
        self.normalization = nn.LayerNorm(in_channels)
        self.resnet = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=resnet_in_channels,
                kernel_size=1),
            ResNetBlock(
                in_channels=resnet_in_channels,
                out_channels=resnet_in_channels),
            ResNetBlock(
                in_channels=resnet_in_channels,
                out_channels=resnet_out_channels),
            ResNetBlock(
                in_channels=resnet_out_channels,
                out_channels=resnet_out_channels),
            nn.Conv1d(
                in_channels=resnet_out_channels,
                out_channels=out_channels,
                kernel_size=1))
        
    def forward(self, input):
        output = torch.transpose(input, 1, 2)
        output = self.normalization(output)
        output = torch.transpose(output, 1, 2)
        output = self.resnet(output)
        return output