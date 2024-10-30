import torch
import torch.nn as nn
import torch.nn.functional as tfunc
from src.model.spex_plus.validate import validate_params
from typing import List, Tuple
from src.model.spex_plus.twin_speech_encoder import TwinSpeechEncoder
from src.model.spex_plus.mix_speech_decoder import MixSpeechDecoder

class SpexPlusModel(nn.Module):
    def __init__(
            self,
            sizes_of_conv_kernels: List[int],
            encoder_out_channels: int,
            tcn_in_channels: int):
        super(SpexPlusModel, self).__init__()
        validate_params(sizes_of_conv_kernels)
        self.encoder = TwinSpeechEncoder(
            sizes_of_conv_kernels=sizes_of_conv_kernels, 
            out_channels=encoder_out_channels)
        self.encoded_mix_concater = nn.Conv1d(len(sizes_of_conv_kernels) * encoder_out_channels, tcn_in_channels, kernel_size=1)
        self.decoder = MixSpeechDecoder(
            sizes_of_conv_kernels=sizes_of_conv_kernels,
            encoder_out_channels=encoder_out_channels)
        

    def forward(self, audio_mix, audio_reference) -> Tuple[torch.Tensor]:
        '''
        batch_size x 
        (num_decoder_ouput_channels) x 
        (audio_len // (min_decoder_kernel_size // 2) - 1)
        '''
        encoded_mix = self.encoded_mix_concater(torch.cat(self.encoder(audio_mix), 1))
        print(encoded_mix.shape)
        '''
        Tuple of num_decoders elements, each is:
        batch_size x 
        (num_parallel_encoders * num_decoder_ouput_channels) x 
        (reference_len // (min_decoder_kernel_size // 2) - 1)
        '''
        encoded_reference = self.encoder(audio_reference)

        decoded_mix_parts = self.decoder(encoded_mix)
        decoded_mix_parts[0] = tfunc.pad(decoded_mix_parts[0], (0, audio_mix.shape[-1] - decoded_mix_parts[0].shape[-1]))
        for i in range(1, len(decoded_mix_parts)):
            decoded_mix_parts[i] = decoded_mix_parts[i][:, :, :audio_mix.shape[-1]]
        return tuple(decoded_mix_parts)