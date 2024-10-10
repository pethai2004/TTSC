# checkpoint is not valid for different training set up (e.g. different worker, batch_size). For no assume that the training set up is the same.
import os 
import logging
import shutil
import pickle
import datetime
import contextlib
import torch

import functools
import torch
from torch.distributed import get_rank, is_initialized, barrier
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, schedule

from config import *

class CallBack:
    
    def __init__(
        self, 
        checkpoint_dir: str,
        num_profile_steps=3, 
        max_checkpoint=10,
        summary_logdir="logs",
    ):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.max_checkpoint = max_checkpoint
        self.summary_logdir = summary_logdir 
        self.num_profile_steps = num_profile_steps
        self.current_checkpoint = None

        self.summary_path = os.path.join(self.checkpoint_dir, self.summary_logdir)
        os.makedirs(self.summary_path, exist_ok=True)
    
        self.writer = SummaryWriter(self.summary_path)
        self.profiler = self.get_profiler_context()
    
    def get_profiler_context(self):
        '''Get profiler context, if `num_profile_steps` <= 0, return null context'''
        if self.num_profile_steps <= 0:
            return contextlib.nullcontext()
        
        on_trace = tensorboard_trace_handler(self.summary_path)
        activity = [ProfilerActivity.CPU]
        activity += [ProfilerActivity.CUDA] if torch.cuda.is_available() else []
        profiler = functools.partial(
            profile, 
            activities=activity,
            schedule=schedule(wait=1, warmup=1, active=5, repeat=self.num_profile_steps),
            on_trace_ready=on_trace,
            profile_memory=True,
            with_flops=True,
            with_modules=True
        )
        return profiler

    def write(self, step, rank=0, **kwargs):
        if is_initialized() and get_rank() != rank:
            return
        for k, v in kwargs.items():
            try:
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    self.writer.add_scalar(k, v, step)
                elif isinstance(v, str):
                    self.writer.add_text(k, v, step)
            except:
                continue
        
    def save_checkpoint(self, model, optimizer, scheduler, training_state, training_config):
        '''Save checkpoint and return path'''

        if is_initialized() and get_rank() != self.node:
            return None
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        self.current_checkpoint = save_checkpoint(
            self.checkpoint_dir, self.checkpoint_dir, model, optimizer,
            scheduler, training_state, training_config, max_checkpoint=self.max_checkpoint
        )
        barrier()
        
        return self.current_checkpoint
    
    def load_checkpoint(self, model, optimizer, scheduler, training_config=None, strict=True):
        '''Load latest checkpoint, if no valid checkpoint is found, return None'''
        latest_checkpoint =  self.current_checkpoint or get_latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint is None:
            return None
        else:
            try:
                chkp = load_checkpoint(latest_checkpoint)
                if chkp is not None:
                    model, optimizer_state, scheduler_state, training_state, new_training_config = chkp
                    model.load_state_dict(model.state_dict(), strict=strict)
                    optimizer.load_state_dict(optimizer_state)
                    scheduler.load_state_dict(scheduler_state)
                    logging.info(f"Loaded checkpoint from {latest_checkpoint} successfully")

                    if training_config is not None:
                        # sanity check, and log any inconsistency field
                        for f in fields(training_config):
                            if getattr(training_config, f.name) != getattr(new_training_config, f.name):
                                logging.warning(f"`TrainingConfig.{f.name}` is inconsistent: {getattr(training_config, f.name)} != {getattr(new_training_config, f.name)}")
                                
            except Exception as e:
                logging.error(f"Error loading checkpoint: {e}")
                return None 
            
        return model, optimizer, scheduler, training_state, training_config
    
    def has_valid_checkpoint(self):
        '''Check if the trial has valid checkpoint'''
        return has_valid_checkpoint(self.checkpoint_dir)
    
def validate_checkpoint(checkpoint_path: str) -> bool:

    return all([
        os.path.exists(os.path.join(checkpoint_path, MODEL_PATH)),
        os.path.exists(os.path.join(checkpoint_path, OPTIMIZER_PATH)),
        os.path.exists(os.path.join(checkpoint_path, SCHEDULE_PATH)),
        os.path.exists(os.path.join(checkpoint_path, TRAINING_STATE_PATH)),
        os.path.exists(os.path.join(checkpoint_path, TRAINING_CONFIG_PATH))
    ])
    
def has_valid_checkpoint(output_dir: str) -> bool:
    """output_dir (str) : path to the training output directory which potentially contain multiple checkpointr"""
    list_chk = [p for p in os.listdir(output_dir) if p.startswith(CHECKPOINT_PREFIX)]
    if not list_chk:
        return False
    for chk_path in list_chk:
        chk_path = os.path.join(output_dir, chk_path)
        if not os.path.isdir(chk_path):
            continue
        if validate_checkpoint(chk_path):
            return True # only one valid checkpoint is enough

def get_latest_checkpoint(output_dir: str) -> str:
    """ If no valid checkpoint is found, return None. Assume that the checkpoint directory is named as `checkpoint_{%Y-%m-%d_%H-%M-%S}`.
    Args: 
        output_dir (str) : path to the training output dir. 
    """
    if not has_valid_checkpoint(output_dir):
        return None
    
    sorted_paths =  sorted(
        [p for p in os.listdir(output_dir) if p.startswith(CHECKPOINT_PREFIX)],
        key=lambda x: os.path.getmtime(os.path.join(output_dir, x))
    )
    return os.path.join(output_dir, sorted_paths[-1])

def rotating_checkpoint(output_dir: str, max_keep: int = 8):
    """Rotate checkpoints by removing the oldest or worst checkpoint if the number exceeds `max_keep`."""
    all_checkpoints = [p for p in os.listdir(output_dir) if CHECKPOINT_PREFIX in p]
    if len(all_checkpoints) <= max_keep or max_keep < 0:
        return

    checkpoint_states = []
    for chk in all_checkpoints:
        try:
            with open(os.path.join(output_dir, chk, TRAINING_STATE_PATH), 'rb') as f:
                state = pickle.load(f)
                checkpoint_states.append((chk, state.best_score))
        #file folder `checkpoint...` may have been created but not yet written the training state, etc. 
        except Exception as e:
            continue
    while len(all_checkpoints) > max_keep:
        oldest_checkpoint = min(all_checkpoints, key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
        shutil.rmtree(os.path.join(output_dir, oldest_checkpoint))
        all_checkpoints.remove(oldest_checkpoint)
        checkpoint_states = [chk for chk in checkpoint_states if chk[0] != oldest_checkpoint]
    return checkpoint_states

def save_checkpoint(output_dir, model, optimizer, scheduler, training_state, training_config, max_checkpoint=10):

    chk_path = os.path.join(output_dir, f"{CHECKPOINT_PREFIX}{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(chk_path, exist_ok=True)
    torch.save(model, os.path.join(chk_path, MODEL_PATH))
    torch.save(optimizer.state_dict(), os.path.join(chk_path, OPTIMIZER_PATH))
    torch.save(scheduler.state_dict(), os.path.join(chk_path, SCHEDULE_PATH))
    pickle.dump(training_state, open(os.path.join(chk_path, TRAINING_STATE_PATH), 'wb'))
    pickle.dump(training_config, open(os.path.join(chk_path, TRAINING_CONFIG_PATH), 'wb'))

    rotating_checkpoint(output_dir, max_keep=max_checkpoint)

    return chk_path

def load_checkpoint(path, map_location=None, **kwargs):

    if map_location is None: 
        if torch.cuda.is_available() and torch.distributed.get_world_size() > 1:    
            map_location = "on_device"
        else:
            map_location = "cpu" 
            
    model = torch.load(os.path.join(path, MODEL_PATH), map_location=map_location)
    optimizer_state = torch.load(os.path.join(path, OPTIMIZER_PATH), map_location=map_location)
    scheduler_state = torch.load(os.path.join(path, SCHEDULE_PATH), map_location=map_location)
    
    train_state = pickle.load(open(os.path.join(path, TRAINING_STATE_PATH), 'rb'))
    train_config = pickle.load(open(os.path.join(path, TRAINING_CONFIG_PATH), 'rb'))

    return model, optimizer_state, scheduler_state, train_state, train_config
