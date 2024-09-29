
import os
import time
import logging

import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter   
from .src.data_utility import _put, AudioTrainingConfig
from .src.dvae_modeling import wrappedDVAE
from .src.data_utility import create_audio_dataloader
from .src.dataset_loader import create_split_metadata

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
torch.set_grad_enabled(True)

def main(
    config: AudioTrainingConfig, 
    output_path: str="./TrainingOutputDir", 
    mel_norm_path: str="./assets/mel_norm.json"
):
    
    start_time = time.time()
    dvae_path = os.path.join(output_path, 'dvae.pth')
    summary_path = os.path.join(output_path, 'logs')
    writer = SummaryWriter(summary_path)
    logger.info(f"------->>>> View tensorboard logs at {summary_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    
    model = wrappedDVAE(gpt_num_audio_tokens=config.gpt_num_audio_tokens, sample_rate=config.sampling_rate, mel_norm_file=mel_norm_path)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=0.000001)
    logger.info(f"------->>>> Model created with {model.num_params()} parameters")
    
    train_meta, eval_meta = create_split_metadata(
        dataset_dir=config.dataset_dir, seed=config.seed
    )
    
    train_dataloader = create_audio_dataloader(train_meta, config.per_device_batch_size, config.sampling_rate, 
                                               config.max_wav_length, config.max_training_example, pin_memory=config.pin_memory)
    eval_dataloader = create_audio_dataloader(eval_meta, config.per_device_batch_size, config.sampling_rate, 
                                                config.max_wav_length, is_train=False, pin_memory=config.pin_memory)
    
    best_evaluation = float('inf')
    global_training_steps = 0
    for epoch in range(config.num_epochs):
        
        model.eval()
        start_eval = time.time()
        with torch.no_grad():
            eval_loss = 0
            num_eval_steps = 0
            for i, batch in enumerate(eval_dataloader):
                batch = _put(batch, device, non_blocking=True)
                loss = model(**batch)
                eval_loss += (loss['reconstruction_loss'] + loss['commitment_loss']).item()
                num_eval_steps += 1
                if config.max_eval_step > 0 and num_eval_steps >= config.max_eval_step:
                    break
            eval_loss /= num_eval_steps
            if eval_loss < best_evaluation:
                best_evaluation = eval_loss
                torch.save(model.state_dict(), dvae_path)
                logger.info(f"------->>>> Saved model at {dvae_path}")
        
            logger.info(f"------->>>> Epoch {epoch} Evaluation Loss: {eval_loss} with time {time.time() - start_eval} at global step {global_training_steps}")
            
            
        model.train()
        start_train = time.time()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = _put(batch, device, non_blocking=True)
            recon_loss, commit_loss = model(**batch).values()
            loss = recon_loss + commit_loss
            loss.backward()
            loss.detach()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            
            global_training_steps += 1
            if global_training_steps % 200 == 0:
                logger.info(f"------->>>> Epoch {epoch} Step {global_training_steps} Loss: {loss.item()} with time {time.time() - start_train}")
            
            writer.add_scalar("Loss/recon_loss", recon_loss.item(), global_training_steps)
            writer.add_scalar("Loss/commit_loss", commit_loss.item(), global_training_steps)
            writer.add_scalar("Loss/total_loss", loss.item(), global_training_steps)
            
            torch.cuda.empty_cache()
        
        logger.info(f"------->>>> Epoch {epoch} completed in {time.time() - start_train} seconds")
    logger.info(f"------->>>> Training completed in {time.time() - start_time} seconds, best evaluation loss: {best_evaluation}, model saved at {dvae_path}")
    
    return dvae_path

if __name__ == "__main__":
    config = AudioTrainingConfig(
        dataset_dir="./assets/LJSpeech-1.1",
        gpt_num_audio_tokens=512,
        sampling_rate=22050,
        max_wav_length=255995,
        max_training_example=-1,
        max_eval_step=-1,
        num_epochs=10,
        per_device_batch_size=32,
        learning_rate=0.0001,
        gradient_clip_norm=1.0,
        pin_memory=True,
        seed=42
    )
    main(config)