# wrap around DVAE model
from typing import Dict 
import torch 

from TTS.tts.layers.xtts.dvae import DiscreteVAE
from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram

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
            normalization=False,
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
        
        return {"reconstruction_loss": rec_loss, "commitment_loss": commmit_loss}
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.mel = self.mel.to(*args, **kwargs) 
        self.dvae.embed = self.dvae.embed.to(*args, **kwargs)
        
