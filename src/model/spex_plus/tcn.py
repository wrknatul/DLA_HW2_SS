import torch
import torch.nn as nn

class TCN(nn.Module):
    def __init__(
            self,
            num_tcn_blocks: int,
            in_channels: int,
            middle_channels: int,
            out_channels: int,
            kernel_size: int):
        super(TCN, self).__init__()
        self.first_block = FirstTCNBlock(
            in_channels=in_channels,
            middle_channels=middle_channels,
            out_channels=out_channels,
            kernel_size=kernel_size)
        self.other_blocks = nn.ModuleList(RegularTCNBlock(
            in_channels=out_channels,
            middle_channels=middle_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation_power=i) for i in range(1, num_tcn_blocks))  

    def forward(self, input, audio_reference):
        output = self.first_block(input, audio_reference)
        for block in self.other_blocks:
            output = block(output)
        return output

class TCNBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            middle_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int):
        super(TCNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=middle_channels,
                kernel_size=1),
            nn.PReLU(),
            GlobalLayerNormalization(middle_channels),
            nn.Conv1d(
                in_channels=middle_channels,
                out_channels=middle_channels,
                kernel_size=kernel_size,
                groups=middle_channels,
                padding=dilation * (kernel_size - 1) // 2,
                dilation=dilation),
            nn.PReLU(),
            GlobalLayerNormalization(middle_channels),
            nn.Conv1d(
                in_channels=middle_channels,
                out_channels=out_channels,
                kernel_size=1))
    
    def forward(self, input):
        return self.block(input)

EPS = 1e-6
        
class GlobalLayerNormalization(nn.Module):
    def __init__(
            self, 
            channels: int):
        super(GlobalLayerNormalization, self).__init__()
        self.channels = channels
        self.gamma = nn.Parameter(torch.ones(channels), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channels), requires_grad=True)

    def forward(self, input):
        dims = list(range(1, len(input.shape)))
        mean = input.mean(dim=dims, keepdim=True)
        var = torch.pow(input - mean, 2).mean(dim=dims, keepdim=True)
        normed_input = (input - mean) / (var + EPS).sqrt()
        return (self.gamma * normed_input.transpose(1, -1) + self.beta).transpose(1, -1)
    
class FirstTCNBlock(TCNBlock):
    def __init__(
            self,
            in_channels: int,
            middle_channels: int,
            out_channels: int,
            kernel_size: int):
        super(FirstTCNBlock, self).__init__(
            in_channels=in_channels,
            middle_channels=middle_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, 
            dilation=1)

    def forward(self, input, audio_reference):
        audio_reference = audio_reference.repeat_interleave(input.shape[-1], dim=-1)
        return super().forward(torch.cat([input, audio_reference], dim=1)) + input

class RegularTCNBlock(TCNBlock):
    def __init__(
            self,
            in_channels: int,
            middle_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation_power: int):
        super(RegularTCNBlock, self).__init__(
            in_channels=in_channels,
            middle_channels=middle_channels,
            out_channels=out_channels,
            kernel_size=kernel_size, 
            dilation=2 ** dilation_power)

    def forward(self, input):
        return super().forward(input)