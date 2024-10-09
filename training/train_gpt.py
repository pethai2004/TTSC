
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchaudio.functional import resample

from huggingface_hub import login
login('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')
# from src.TTS.tts.configs.xtts_config import XttsConfig
# from src.TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig
# from src.TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
# from src.TTS.tts.layers.xtts.dvae import DiscreteVAE
# from src.TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# def train_gpt(
#     model_config: XttsConfig, 
#     audio_config: XttsAudioConfig,
#     tokenizer_file: str='./assets/tokenining/tokenizer.json',
#     mel_norm_file: str='./assets/mel_stats.pth',
#     dvae: DiscreteVAE=None,
#     model_checkpoint: str=None,
#     dvae_sampling_rate: int=16000,
    
#     text_weight_loss: float=1.0,
#     mel_weight_loss: float=1.0,
# ):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     model = Xtts(model_config)
#     model.tokenizer = VoiceBpeTokenizer(tokenizer_file)
#     model.init_models() # this initialize GPT2 and Hifi-GAN decoder modules
    
#     if model_checkpoint is not None:
#         state_dict = model.get_compatible_checkpoint_state_dict(model_checkpoint)
#         model.load_state_dict(state_dict, strict=False)
    
#     mel_stats = torch.load(mel_norm_file, map_location="cpu")
#     model.mel_stats = mel_stats
    
#     spectrogram_style_encoder = TorchMelSpectrogram(
#         filter_length=2048, hop_length=256, win_length=1024, normalize=False,
#         sampling_rate=audio_config.sample_rate, mel_fmin=0, mel_fmax=8000, n_mel_channels=80, mel_norm_file=mel_norm_file
#     )
#     spectrogram_dvae  = TorchMelSpectrogram(
#         mel_norm_file=mel_norm_file, sampling_rate=dvae_sampling_rate
#     )
#     @torch.no_grad()
#     def format_batch(batch): # get discrete spectrogram code
        
#         conditioning = batch['conditioning'] # [b, num_samples, c, t]
#         b, num_samples, c, t = conditioning.size()
#         conditioning = conditioning.view(b * num_samples, c, t) # [b * num_samples, c, t_mel]
#         mel = spectrogram_style_encoder(conditioning)
#         mel = mel.view(b, num_samples, mel.size(1), mel.size(2)) # [b, num_samples, c, t_mel]
        
#         wav = batch['wav']
#         if audio_config.sample_rate != dvae_sampling_rate:
#             wav = resample(wav, audio_config.sample_rate, dvae_sampling_rate, lowpass_filter_width=32, rolloff=95, beta=14.7)
#         dvae_mel = spectrogram_dvae(wav)
#         codebook = dvae.get_codebook_indices(dvae_mel)
        
#         return { 
#             'text_inputs': batch['text_inputs'],
#             'text_lengths': batch['text_lengths'],
#             'audio_codes': codebook,
#             'wav_lengths': batch['wav_lengths'],
#             'cond_mels': mel,
#             'cond_idxs': batch['cond_idxs'],
#             'cond_lens': batch['cond_lens'],
#         }
        
#     def compute_loss(batch):
        
#         losses = model.gpt( # forward pass
#             batch['text_inputs'], 
#             batch['text_lengths'], 
#             batch['audio_codes'], 
#             batch['wav_lengths'], 
#             cond_mels=batch['cond_mels'],
#             cond_idxs=batch['cond_idxs'],
#             cond_lens=batch['cond_lens'],
#         )
#         loss_text, loss_mel, _ = losses
#         losses['loss'] = loss_text * text_weight_loss + loss_mel * mel_weight_loss
        
#         return losses
        
#     # put only GPT params as trainable, the rest are frozen.
#     model.eval()
#     dvae.eval()
#     if hasattr(model, "module") and hasattr(model.module, "xtts"):
#         model.module.xtts.gpt.train()
#     else:
#         model.xtts.gpt.train()

#     logger.info(f"GPT params count: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
#     logger.info(f"Total params count: {sum(p.numel() for p in model.parameters())}")
#     logger.info(f"DVAE params count: {sum(p.numel() for p in dvae.parameters())}")

# if __name__ == "__main__":
    
#     dvae = DiscreteVAE(
#         positional_dims=1,
#         num_tokens=1024,
#         codebook_dim=512,
#         num_layers=2,
#         num_resnet_blocks=1,
#         hidden_dim=512,
#         channels=80,
#         use_transposed_convs=False
#     )
#     model_config = XttsConfig(
#         output_path='./xtts_output',
        
#     )
#     model_args = XttsArgs(
#         gpt_layers = 8,
#         gpt_n_model_channels = 256,
#         gpt_n_heads = 16,
#     )
#     audio_config = XttsAudioConfig(sample_rate=16000, output_sample_rate=16000)

#     train_gpt(
#         model_config=model_config,
#         audio_config=audio_config,
#         dvae=dvae,
#     )





##############################################################################################################################
from src.TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig

from trainer import Trainer, TrainerArgs

def main():
    
    audio_config = XttsAudioConfig(sample_rate=16000, dvae_sample_rate=16000, output_sample_rate=16000)
    
    gpt_model_args = GPTArgs(
        gpt_layers = 2,
        gpt_n_model_channels = 128,
        gpt_n_heads = 4,
        decoder_input_dim=128,
        d_vector_dim=128,
        gpt_num_audio_tokens= 1024 + 2,
        gpt_start_audio_token= 1022,
        gpt_stop_audio_token= 1023,
        tokenizer_file='/Users/owan/Documents/PROJECTs/TTScustom/TTSC/assets/tokenizing/tokenizer.json',
        mel_norm_file='/Users/owan/Documents/PROJECTs/TTScustom/TTSC/assets/mel_stats.pth',
        dvae_checkpoint='/Users/owan/Documents/PROJECTs/TTScustom/TTSC/assets/dvae.pth',
    )
    gpt_trainer_config = GPTTrainerConfig(
        output_path='./gpt_output',
        audio=audio_config,
        model_args=gpt_model_args,
        optimizer="AdamW",
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,  
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
    )
    gpt_trainer = GPTTrainer(gpt_trainer_config)
    
    logger.info(f"Model params count: {sum(p.numel() for p in gpt_trainer.parameters())}")

    from src.gigaspeech_dataset import create_gigaspeech_dataloader, format_batch_on_device
    from training.training_utils import create_default_optim_and_scheduler

    optimizer, scheduler = create_default_optim_and_scheduler(gpt_trainer, learning_rate=5e-5)

    train_loader = create_gigaspeech_dataloader(
        dataset_dir="./assets/gigaspeech",
        batch_size=16,
        subset='s',
        split='validation',
        token='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'
    )
    train_loader = train_loader['validation'] # do not forget
    
    for batch in train_loader:
        
        batch = format_batch_on_device(
            batch, 
            dvae=gpt_trainer.dvae,
            encoder_mel=gpt_trainer.torch_mel_spectrogram_style_encoder,
            dvae_mel=gpt_trainer.torch_mel_spectrogram_dvae,
            input_sample_rate=audio_config.sample_rate,
            dvae_sampling_rate=audio_config.dvae_sample_rate,
        )
        
        loss_text, loss_mel, _ = gpt_trainer(
            text_inputs=batch['text_inputs'], 
            text_lengths=batch['text_lengths'], 
            audio_codes=batch['audio_codes'], 
            wav_lengths=batch['wav_lengths'], 
            cond_mels=batch['cond_mels'],
            cond_idxs=batch['cond_idxs'],
            cond_lens=batch['cond_lens'],
        )
        loss = loss_text + loss_mel # will add weighted loss later
        
    
if __name__ == "__main__":
    main()