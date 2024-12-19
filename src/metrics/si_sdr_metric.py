import torch
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as SI_SDR
import math 

from src.metrics.base_metric import BaseMetric


class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_sdr = SI_SDR()

    def __call__(self, s1: torch.Tensor, target: torch.Tensor, **kwargs):
        self.si_sdr = self.si_sdr.to(s1.device)
        res = self.si_sdr(s1, target).item()
        if math.isinf(res) or math.isnan(res) or -0.1 > res:
            return 0.
        return res
