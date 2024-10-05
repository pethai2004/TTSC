# Training script for DVAE model
import os 
from dataclasses import dataclass
import time
import logging 

import torch 
import torch.distributed as D
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter   
from torch.profiler import profile, record_function, ProfilerActivity
from src.data_utility import _put
from src.dvae_modeling import wrappedDVAE
from src.gigaspeech_dataset import IterableGigaSpeech, collate_fn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
torch.set_grad_enabled(True)

@dataclass
class AudioTrainingConfig:
    
    output_dir : str = './TrainingOutputDir'
    trial_name : str = 'default_trial'    
    
    per_device_batch_size: int = 32
    gpt_num_audio_tokens : int = 1024
    num_epochs : int = 30
    max_batch_step : int = 10_000
    max_eval_step : int = 2000
    learning_rate : float = 5e-6
    gradient_clip_norm : float = 0.7
    
    mel_norm_path : str = './assets/mel_stats.pth'
    tokenizer_path : str = './assets/tokenizing/tokenizer.json'
    model_checkpoint: str = None
    
    non_blocking : bool = True
    wav_dtype : torch.dtype = torch.float32
    device : str = 'cuda'
    sampling_rate : int = 16_000
    
    log_interval : int = 10
    
def main(
    config: AudioTrainingConfig, 
    **model_args
):
    
    # if not D.is_initialized():
    #     D.init_process_group(backend='gloo', init_method='env://')
    # assert D.get_world_size() == 1, "This script is only for single GPU training"
    
    training_output_dir = os.path.join(config.output_dir, 'dvae.pth', config.trial_name)
    if os.path.exists(training_output_dir):
        #add suffix to the directory trial name
        training_output_dir = training_output_dir + f"_{time.time()}"
    os.makedirs(training_output_dir, exist_ok=True)
    logger.info(f"------->>>> Training output directory: {training_output_dir}")
    
    dvae_path = os.path.join(training_output_dir, 'dvae.pth')
    summary_path = os.path.join(training_output_dir, 'logs')
    writer = SummaryWriter(summary_path)
    # prepare model, optimizer, scheduler
    model = wrappedDVAE(**model_args, sampling_rate=config.sampling_rate, mel_norm_file=config.mel_norm_path)
    if config.model_checkpoint is not None:
        try:
            model.load_state_dict(torch.load(config.model_checkpoint), strict=False)
            logger.info(f"------->>>> Loaded model from {config.model_checkpoint}")
        except:
            pass
    else:
        model.reset_parameters()
    model = model.to(device=config.device, dtype=config.wav_dtype, non_blocking=config.non_blocking)
    
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=0.000001)
    logger.info(f"------->>>> Model created with {model.num_parameters()} parameters")
    
    # prepare dataset 
    train_data = IterableGigaSpeech(
        batch_size=config.per_device_batch_size, subset='s', split='train', tokenizer_path=config.tokenizer_path, 
        max_batch_sample=config.max_batch_step
    )
    eval_data = IterableGigaSpeech(
        batch_size=config.per_device_batch_size, subset='s', split='validation', tokenizer_path=config.tokenizer_path, 
        max_batch_sample=config.max_eval_step
    )
    train_data_iter = iter(train_data)
    eval_data_iter = iter(eval_data)
    
    best_evaluation = float('inf')
    global_training_steps = 0
    
    for epoch in range(config.num_epochs):
        
        model.eval()
        start_eval = time.time()
        with torch.autocast(device_type=config.device, 
                            dtype=config.wav_dtype,
                            enabled=True):
            
            logger.info(f"------->>>> Start evaluation at epoch {epoch}")
            with torch.no_grad(): 
                eval_loss = 0
                num_eval_steps = 0
                for i, batch in enumerate(eval_data_iter):
                    batch = collate_fn(batch, device=config.device, dtype=config.wav_dtype, non_blocking=config.non_blocking)
                    wav = batch['wav']
                    loss = model(wav)
                    eval_loss += (loss['reconstruction_loss'] + loss['commitment_loss']).item()
                    
                    if torch.isnan(torch.tensor(eval_loss)) or torch.isinf(torch.tensor(eval_loss)):
                        logger.info(f"------->>>> Evaluation Loss is NaN or Inf at global step {global_training_steps}")
                        continue    
                    
                    num_eval_steps += 1
                    if num_eval_steps >= config.max_eval_step and config.max_eval_step > 0:
                        # should be "{eval_data.global_batch_sample_so_far}== {num_eval_steps}")
                        break
                    
                eval_loss /= num_eval_steps
                if eval_loss < best_evaluation:
                    best_evaluation = eval_loss
                    torch.save(model.state_dict(), dvae_path)
                    logger.info(f"------->>>> Saved model at {dvae_path}")
                logger.info(f"------->>>> Epoch {epoch} Evaluation Loss: {eval_loss} with time {time.time() - start_eval} at global step {global_training_steps}")
        
            logger.info(f"------->>>> Start training at epoch {epoch}")
            try:
                model.train()
                start_train = time.time()
                for i, batch in enumerate(train_data_iter):
                    optimizer.zero_grad()
                    batch = collate_fn(batch, device=config.device, dtype=config.wav_dtype, non_blocking=config.non_blocking)
                    wav = batch['wav']
                    recon_loss, commit_loss = model(wav).values()
                    loss = recon_loss + commit_loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.info(f"------->>>> Training Loss is NaN or Inf at global step {global_training_steps}")
                        continue
                    loss.backward()
                    loss.detach()
                    
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                    optimizer.step()
                    scheduler.step()
                    
                    global_training_steps += 1
                    if global_training_steps % config.log_interval == 0:
                        logger.info(f"------->>>> Epoch {epoch} Step {global_training_steps} Loss: {loss.item()} with time {time.time() - start_train}")
                    
                    writer.add_scalar("Loss/recon_loss", recon_loss.item(), global_training_steps)
                    writer.add_scalar("Loss/commit_loss", commit_loss.item(), global_training_steps)
                    writer.add_scalar("Loss/total_loss", loss.item(), global_training_steps)
                    writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], global_training_steps)
                    writer.add_scalar("GradientNorm", norm, global_training_steps)

                    model_code = model.get_codebook_indices(wav)['code']
                    num_unique_ele = torch.unique(model_code).numel()
                    writer.add_scalar("UniqueCodebookElements", num_unique_ele, global_training_steps) # should increase over time
                    
                    if global_training_steps >= config.max_batch_step and config.max_batch_step > 0:
                        break
                    
                logger.info(f"------->>>> Epoch {epoch} completed in {time.time() - start_train} seconds")
            
            except KeyboardInterrupt:
                logger.info(f"------->>>> Training interrupted, saved model saved at {dvae_path}")
                torch.save(model.state_dict(), os.path.join(config.output_dir, 'dvae.pth'))
                break
            
            logger.info(f"------->>>> Training completed, saved model saved at {os.path.join(config.output_dir, 'dvae.pth')}")
            
    return dvae_path

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output_dir", type=str, default="./TrainingOutputDir")
    parser.add_argument("--trial_name", type=str, default="./default_trial")
    
    parser.add_argument("--model_checkpoint", type=str, default=None)
    
    parser.add_argument("--per_device_batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--max_batch_step", type=int, default=-1)
    parser.add_argument("--max_eval_step", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=30)
    
    args = parser.parse_args()
    
    training_config = AudioTrainingConfig(
        per_device_batch_size=args.per_device_batch_size,
        model_checkpoint=args.model_checkpoint,
        num_epochs=args.num_epochs,
        max_batch_step=args.max_batch_step,
        max_eval_step=args.max_eval_step,

        output_dir=args.output_dir,
        trial_name=args.trial_name,
        
        log_interval=args.log_interval
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
        **model_args
    )

        