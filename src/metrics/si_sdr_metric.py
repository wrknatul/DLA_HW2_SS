import torch
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as SI_SDR

from src.metrics.base_metric import BaseMetric


class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_sdr = SI_SDR()

    def __call__(self, s1: torch.Tensor, target: torch.Tensor, **kwargs):
        self.si_sdr = self.si_sdr.to(s1.device)
        return self.si_sdr(s1, target).item()