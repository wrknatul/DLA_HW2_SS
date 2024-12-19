import torch
import torch.nn as nn
import torch.nn.functional as tfunc
from src.model.spex_plus.validate import validate_params
from typing import List, Tuple
from src.model.spex_plus.twin_speech_encoder import TwinSpeechEncoder
from src.model.spex_plus.mix_speech_decoder import MixSpeechDecoder
from src.model.spex_plus.speaker_encoder import SpeakerEncoder
from src.model.spex_plus.tcn import TCN

class SpexPlusModel(nn.Module):
    def __init__(
            self,
            sizes_of_conv_kernels: List[int],
            encoder_out_channels: int,
            resnet_in_channels: int,
            resnet_out_channels: int,
            speaker_encoder_out_channels: int,
            tcn_in_channels: int,
            tcn_middle_channels: int,
            tcn_out_channels: int,
            num_tcns: int,
            num_tcn_blocks: int,
            num_speakers: int):
        super(SpexPlusModel, self).__init__()
        validate_params(sizes_of_conv_kernels)
        self.encoder = TwinSpeechEncoder(
            sizes_of_conv_kernels=sizes_of_conv_kernels, 
            out_channels=encoder_out_channels)
        self.encoded_mix_concater = nn.Conv1d(len(sizes_of_conv_kernels) * encoder_out_channels, tcn_in_channels, kernel_size=1)
        self.decoder = MixSpeechDecoder(
            sizes_of_conv_kernels=sizes_of_conv_kernels,
            encoder_out_channels=encoder_out_channels)
        self.speaker_encoder = SpeakerEncoder(
            in_channels=len(sizes_of_conv_kernels) * encoder_out_channels,
            out_channels=speaker_encoder_out_channels,
            resnet_in_channels=resnet_in_channels,
            resnet_out_channels=resnet_out_channels)        
        self.tcns = nn.ModuleList(TCN(
            num_tcn_blocks=num_tcn_blocks,
            in_channels=tcn_in_channels+speaker_encoder_out_channels,
            middle_channels=tcn_middle_channels,
            out_channels=tcn_out_channels,
            kernel_size=3) for i in range(num_tcns))
        self.after_encoder_masks = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(resnet_in_channels, encoder_out_channels, kernel_size=1),
                nn.ReLU()
            ) for i in range(len(sizes_of_conv_kernels)))
        self.speaker_head = nn.Linear(speaker_encoder_out_channels, num_speakers)

    def forward(self, mix: torch.Tensor, reference: torch.Tensor, **kwargs) -> dict:
        # t = torch.cuda.get_device_properties(0).total_memory
        # r = torch.cuda.memory_reserved(0)
        # a = torch.cuda.memory_allocated(0)
        # print(t, r, a)
        encoded_mix = self.encoded_mix_concater(torch.cat(self.encoder(mix), 1))
        processed_audio_reference = torch.sum(self._process_reference(reference), -1, True)
        for tcn in self.tcns:
            encoded_mix = tcn(encoded_mix, processed_audio_reference)
        masked_mixes = []
        for mask_layer, mix_after_encoder in zip(self.after_encoder_masks, encoded_mix):
            masked_mixes.append(mix_after_encoder * mask_layer(encoded_mix))
        decoded_mix_parts = self.decoder(encoded_mix)
        decoded_mix_parts[0] = tfunc.pad(decoded_mix_parts[0], (0, mix.shape[-1] - decoded_mix_parts[0].shape[-1]))
        for i in range(1, len(decoded_mix_parts)):
            decoded_mix_parts[i] = decoded_mix_parts[i][:, :, :mix.shape[-1]]
        speaker_preds = self.speaker_head(processed_audio_reference.squeeze())
        if speaker_preds.ndim == 1:
            speaker_preds = torch.unsqueeze(speaker_preds, 0)
        return {
            "s1": decoded_mix_parts[0],
            "s2": decoded_mix_parts[1],
            "s3": decoded_mix_parts[2],
            "speaker_preds": speaker_preds,
        }
    
    def _process_reference(self, audio_reference: torch.Tensor) -> torch.Tensor:
        encoded_reference = torch.cat(self.encoder(audio_reference), 1)
        encoded_reference = self.speaker_encoder(encoded_reference)
        return encoded_reference
        
