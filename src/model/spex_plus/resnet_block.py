import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int):
        super(ResNetBlock, self).__init__()
        self.before_skip_connection = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False),
            nn.BatchNorm1d(num_features=out_channels),
            nn.PReLU(),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False),
            nn.BatchNorm1d(num_features=out_channels))
        self.after_skip_connection = nn.Sequential(
            nn.PReLU(),
            nn.MaxPool1d(kernel_size=3))
        if in_channels != out_channels:
            self.skip_conn_preparer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False)
        else:
            self.skip_conn_preparer = nn.Identity()
        
    def forward(self, input):
        output = self.before_skip_connection(input)
        output += self.skip_conn_preparer(input)
        output = self.after_skip_connection(output)
        return output