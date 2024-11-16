import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as si_sdr

class SpexPlusLoss(torch.nn.Module):
    def __init__(
            self, alpha: float=0.1, beta: float=0.1, gamma: float=0.5
        ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce_loss = CrossEntropyLoss(reduction='mean')
        self.dist_ratio_1 = si_sdr()
        self.dist_ratio_2 = si_sdr()
        self.dist_ratio_3 = si_sdr()

    def forward(
            self, 
            s1: Tensor, s2: Tensor, s3: Tensor, target: Tensor,
            speaker_preds: Tensor = None, speaker_id: Tensor = None, **batch
        ) -> Tensor:
        device = speaker_preds.device
        speaker_id = speaker_id.to(device)
        target_audio = target.to(device)

        s1_loss = (1 - self.alpha - self.beta) * self.dist_ratio_1(s1, target_audio)
        s2_loss = self.alpha * self.dist_ratio_2(s2, target_audio)
        s3_loss = self.beta * self.dist_ratio_3(s3, target_audio)
        si_sdr_loss = -(s1_loss + s2_loss + s3_loss)

        ce_loss = 0
        if speaker_preds is not None:
            ce_loss = self.ce_loss(speaker_preds, speaker_id)

        loss = si_sdr_loss + self.gamma * ce_loss
        return loss