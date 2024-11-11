import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ

from src.metrics.base_metric import BaseMetric


class PESQMetric(BaseMetric):
    def __init__(self, sr: int=16000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sr = sr
        self.pesq = PESQ(self.sr, 'wb')

    def __call__(self, preds: torch.Tensor, target_audio: torch.Tensor, **kwargs):
        self.pesq = self.pesq.to(preds.device)
        pesq = self.pesq(preds / preds.norm(dim=-1, keepdim=True), target_audio)
        return pesq 