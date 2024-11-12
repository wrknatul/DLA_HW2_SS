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
        self.tcn = TCN(
            num_tcn_blocks=num_tcn_blocks,
            in_channels=tcn_in_channels+speaker_encoder_out_channels,
            middle_channels=tcn_middle_channels,
            out_channels=tcn_out_channels,
            kernel_size=3)
        self.after_encoder_masks = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(resnet_in_channels, encoder_out_channels, kernel_size=1),
                nn.ReLU()
            ) for i in range(len(sizes_of_conv_kernels)))
        self.speaker_linear = nn.Linear(
            in_features=speaker_encoder_out_channels, 
            out_features=num_speakers)

    def forward(self, audio_mix, audio_reference) :
        '''
        batch_size x 
        (num_decoder_ouput_channels) x 
        (audio_len // (min_decoder_kernel_size // 2) - 1)
        '''
        encoded_mix = self.encoded_mix_concater(torch.cat(self.encoder(audio_mix), 1))
        processed_audio_reference = torch.sum(self._process_reference(audio_reference), -1, True)
        processed_audio_mix_by_tcn = self.tcn(encoded_mix, processed_audio_reference)
        masked_mixes = []
        for mask_layer, mix_after_encoder in zip(self.after_encoder_masks, processed_audio_mix_by_tcn):
            masked_mixes.append(mix_after_encoder * mask_layer(encoded_mix))
        decoded_mix_parts = self.decoder(encoded_mix)
        decoded_mix_parts[0] = tfunc.pad(decoded_mix_parts[0], (0, audio_mix.shape[-1] - decoded_mix_parts[0].shape[-1]))
        for i in range(1, len(decoded_mix_parts)):
            decoded_mix_parts[i] = decoded_mix_parts[i][:, :, :audio_mix.shape[-1]]
        output = decoded_mix_parts + [self.speaker_linear(processed_audio_reference.squeeze())]
        return tuple(output) 
    
    def _process_reference(self, audio_reference):
        '''
        Tuple of num_decoders elements, each is:
        batch_size x 
        (num_parallel_encoders * num_decoder_ouput_channels) x 
        (reference_len // (min_decoder_kernel_size // 2) - 1)
        '''
        encoded_reference = torch.cat(self.encoder(audio_reference), 1)
        encoded_reference = self.speaker_encoder(encoded_reference)
        return encoded_reference