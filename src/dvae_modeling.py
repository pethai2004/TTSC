# wrap around DVAE model
from TTS.tts.layers.xtts.dvae import DiscreteVAE
from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram

class wrappedDVAE(Module):

    def __init__(self, gpt_num_audio_tokens=1024, sample_rate=22050, mel_norm_file=None):
        super().__init__()
        self.torch_mel_spectrogram_style_encoder = True # always set to True
        self.gpt_num_audio_tokens = gpt_num_audio_tokens
        self.mel = TorchMelSpectrogram(         
                filter_length=4096,
                hop_length=1024,
                win_length=4096,
                normalize=False,
                sampling_rate=sample_rate,
                mel_fmin=0,
                mel_fmax=8000,
                n_mel_channels=80,
                mel_norm_file=None,
            )
        self.mel.mel_norms = mel_norm_file # use direct
        self.dvae = DiscreteVAE(
            channels=80,
            normalization=True,
            positional_dims=1,
            num_tokens=self.gpt_num_audio_tokens,
            codebook_dim=512,
            hidden_dim=512,
            num_resnet_blocks=1,
            kernel_size=3,
            num_layers=2,
            use_transposed_convs=False,
        ) 
        
    def validate_seq_length(self, x):
        size =  x.size(-1)
        r = size % (2 ** self.dvae.num_layers)
        if r != 0:
            x = x[..., :-r]
        return x
    
    @torch.no_grad()
    def get_codebook_indices(self, wav):
        mel = self.mel(wav)
        mel = self.validate_seq_length(mel)
        code = self.dvae.get_codebook_indices(mel)
        return {"code": code}
    
    def forward(self, wav):
        with torch.no_grad():
            mel = self.mel(wav)
            mel = self.validate_seq_length(mel)
            
        rec_loss, commmit_loss, _ = self.dvae.forward(mel)
        rec_loss = rec_loss.mean()
        
        return {"reconstruction_loss": rec_loss, "commitment_loss": commmit_loss}
    
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 