import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ

from src.metrics.base_metric import BaseMetric


class PESQMetric(BaseMetric):
    def __init__(self, sr: int=16000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sr = sr
        self.pesq = PESQ(self.sr, 'wb')

    def __call__(self, s1: torch.Tensor, target: torch.Tensor, **kwargs):
        self.pesq = self.pesq.to(s1.device)
        pesq = self.pesq(s1 / s1.norm(dim=-1, keepdim=True), target)
        return pesq 