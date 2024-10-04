# Training script for DVAE model
import os 
import time
import logging 

import torch 
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter   
from .src.data_utility import _put
from .src.dvae_modeling import wrappedDVAE
from .src.data import AudioTrainingConfig, AudioDataConfig, PeopleSpeech, GigaSpeech

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
torch.set_grad_enabled(True)

def main(
    config: AudioTrainingConfig, 
    data_train_config: AudioDataConfig, 
    data_eval_config: AudioDataConfig,
    **model_args
):
    
    training_output_dir = os.path.join(config.output_dir, config.trial_name)
    if os.path.exists(training_output_dir):
        #add suffix to the directory trial name
        training_output_dir = training_output_dir + "_1"
    os.makedirs(training_output_dir, exist_ok=True)
    logger.info(f"------->>>> Training output directory: {training_output_dir}")
    
    dvae_path = os.path.join(training_output_dir, 'dvae.pth')
    summary_path = os.path.join(training_output_dir, 'logs')
    writer = SummaryWriter(summary_path)
    # prepare model, optimizer, scheduler
    model = wrappedDVAE(**model_args, sample_rate=config.sampling_rate, mel_norm_file=config.mel_norm_path)
    if config.model_checkpoint is not None:
        model.load_state_dict(torch.load(config.model_checkpoint), strict=False)
        logger.info(f"------->>>> Loaded model from {config.model_checkpoint}")
    model.to(device=config.device, dtype=config.wav_dtype, non_blocking=config.non_blocking)
    
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=0.000001)
    logger.info(f"------->>>> Model created with {model.num_parameters()} parameters")
    
    # prepare dataset 
    
    
    
    
if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output_dir", type=str, default="./TrainingOutputDir")
    parser.add_argument("--trial_name", type=str, default="default_trial")
    
    parser.add_argument("--model_checkpoint", type=str, default=None)
    parser.add_argument("--training_dataset_name", type=str, default="GigaSpeech")
    
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--max_batch_step", type=int, default=20_000)
    parser.add_argument("--max_eval_step", type=int, default=2000)
    
    args = parser.parse_args()
    
    assert args.training_dataset_name in ["GigaSpeech", "PeopleSpeech"] # no LJSpeech for now
    
    training_config = AudioTrainingConfig(
        batch_size=args.batch_size,
        model_checkpoint=args.model_checkpoint,
        num_epochs=args.num_epochs,
        max_batch_step=args.max_batch_step,
        max_eval_step=args.max_eval_step,

        output_dir=args.output_dir,
        trial_name=args.trial_name,
    )
    if args.training_dataset_name == "PeopleSpeech":
        data_train_config = AudioDataConfig(
            batch_size=args.batch_size,
            split="train"
        )
        data_eval_config = AudioDataConfig(
            batch_size=args.batch_size,
            split="validation"
        )
    elif args.training_dataset_name == "GigaSpeech":
        data_eval_config = AudioDataConfig(
            batch_size=args.batch_size,
            subset='s',
            split="train"
        )
        data_eval_config = AudioDataConfig(
            batch_size=args.batch_size,
            subset='s',
            split="validation"
        )
    
    model_args = {
        "codebook_dim": 512,
        "hidden_dim": 512,
        "num_resnet_blocks": 1,
        "kernel_size": 3,
        "num_layers": 2,
    }
    
    main(
        training_config,
        data_train_config=data_train_config,
        data_eval_config=data_eval_config,
        **model_args
    )