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
        self.conv1d1 =nn.Conv1d(
                in_channels=in_channels,
                out_channels=resnet_in_channels,
                kernel_size=1),
        self.resnet1 = ResNetBlock(
                in_channels=resnet_in_channels,
                out_channels=resnet_in_channels)
        self.resnet2 = ResNetBlock(
                in_channels=resnet_in_channels,
                out_channels=resnet_out_channels)
        self.resnet3 = ResNetBlock(
                in_channels=resnet_out_channels,
                out_channels=resnet_out_channels)
        self.conv1d2 = nn.Conv1d(
                in_channels=resnet_out_channels,
                out_channels=out_channels,
                kernel_size=1)
        
    def forward(self, input):
        output = torch.transpose(self.normalization(torch.transpose(input, 1, 2)), 1, 2)
        output = self.conv1d1(output)
        output = self.resnet1(output)
        output = self.resnet2(output)
        output = self.resnet3(output)
        output = self.conv1d2(output)
        return output