# wrap around DVAE model
from typing import Dict 
import torch 

import torch.nn as nn
import torchaudio
import fsspec

from src.dvae import DiscreteVAE

class TorchMelSpectrogram(nn.Module): #https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/layers/tortoise/arch_utils.py
    def __init__(
        self,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0,
        mel_fmax=8000,
        sampling_rate=22050,
        normalize=False,
        mel_norm_file=None,
    ):
        super().__init__()
        # These are the default tacotron values for the MEL spectrogram.
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sampling_rate = sampling_rate
        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            power=2,
            normalized=normalize,
            sample_rate=self.sampling_rate,
            f_min=self.mel_fmin,
            f_max=self.mel_fmax,
            n_mels=self.n_mel_channels,
            norm="slaney",
        )
        self.mel_norm_file = mel_norm_file
        if self.mel_norm_file is not None:
            with fsspec.open(self.mel_norm_file) as f:
                self.mel_norms = torch.load(f)
        else:
            self.mel_norms = None

    def forward(self, inp):
        if (
            len(inp.shape) == 3
        ):  # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
            inp = inp.squeeze(1)
        assert len(inp.shape) == 2
        self.mel_stft = self.mel_stft.to(inp.device)
        mel = self.mel_stft(inp)
        # Perform dynamic range compression
        mel = torch.log(torch.clamp(mel, min=1e-5))
        if self.mel_norms is not None:
            self.mel_norms = self.mel_norms.to(mel.device)
            mel = mel / self.mel_norms.unsqueeze(0).unsqueeze(-1)
        return mel

class wrappedDVAE(torch.nn.Module):

    def __init__(
        self,
        codebook_dim: int=512,
        hidden_dim: int=512,
        num_resnet_blocks: int=1,
        kernel_size: int=3,
        num_layers: int=2,
        use_transposed_convs: bool=False,
        num_tokens: int=1024, 
        sampling_rate: int=22050,
        mel_norm_file: str=None
        
    ):
        super().__init__()
        self.torch_mel_spectrogram_style_encoder = True 
        self.mel = TorchMelSpectrogram(         
                filter_length=4096,
                hop_length=1024,
                win_length=4096,
                normalize=False,
                sampling_rate=sampling_rate,
                mel_fmin=0,
                mel_fmax=8000,
                n_mel_channels=80,
                mel_norm_file=None,
            )

        if isinstance(mel_norm_file, str):
            self.mel.mel_norms = torch.load(mel_norm_file)
        elif isinstance(mel_norm_file, torch.Tensor):
            self.mel.mel_norms = mel_norm_file
        else: self.mel.mel_norms = None
            
        self.dvae = DiscreteVAE(
            channels=80,
            normalization=None,
            positional_dims=1,
            num_tokens=num_tokens,
            codebook_dim=codebook_dim,
            hidden_dim=hidden_dim,
            num_resnet_blocks=num_resnet_blocks,
            kernel_size=kernel_size,
            num_layers=num_layers,
            use_transposed_convs=use_transposed_convs,
        ) 
    
    @torch.no_grad()
    def validate_seq_length(self, x):
        size =  x.size(-1)
        r = size % (2 ** self.dvae.num_layers)
        if r != 0:
            x = x[..., :-r]
        return x
    
    @torch.no_grad()
    def get_codebook_indices(self, wav) -> Dict[str, torch.Tensor]:
        mel = self.mel(wav)
        mel = self.validate_seq_length(mel)
        code = self.dvae.get_codebook_indices(mel)
        return {"code": code}
    
    def forward(self, wav) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            mel = self.mel(wav)
            mel = self.validate_seq_length(mel)
            
        rec_loss, commmit_loss, _ = self.dvae.forward(mel)
        rec_loss = rec_loss.mean()
        
        # will be removed
        return {"reconstruction_loss": rec_loss, "commitment_loss": commmit_loss}
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.mel = self.mel.to(*args, **kwargs) 
        self.dvae = self.dvae.to(*args, **kwargs)
        self.dvae.codebook.embed = self.dvae.codebook.embed.to(*args, **kwargs)
        return self

    def reset_parameters(self):
        
        for name, params in self.named_parameters():
            if "weight" in name:
                torch.nn.init.normal_(params, mean=0.0, std=0.02)
            elif "bias" in name:
                torch.nn.init.constant_(params, 0.0)