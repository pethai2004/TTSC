# Modular Trainer class 

import os 
import time
import inspect
from functools import partial
from typing import List, Tuple, Dict, Union, Any
from contextlib import ExitStack, nullcontext, contextmanager

import torch
from torch import Tensor, amp, cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, is_initialized, get_rank

from config import TrainingConfig, State
from trainer_utility import log, set_seed, create_optimizer_and_scheduler, _put
from io_utility import *

class Trainer:

    def __init__(
        self, 
        config: TrainingConfig,
        model: nn.Module,
    ):
        self.config = config
        self.config._post_init()
        self.state = State() 
        self.state.set(config)
        self.train_dataloader = None
        self.eval_dataloader = None
        self.optimizer, self.scheduler = create_optimizer_and_scheduler(
            self.model, learning_rate=config.learning_rate) # use default optimizer and scheduler
        self.gradient_scaler = amp.GradScaler(enabled=config.device == "cuda") if config.mixed_precision != "off" else None
        self.model_input_name = None
        
        if not is_initialized():
            init_process_group(backend=self.config.backend, init_method='env://')
            log(f"Initialized process group", rank=-1)
            
        set_seed(self.config.seed)
        self.has_been_synced = False
        self._should_skip_update = False
        self._gradient_steps = 0
        self.put_func = partial(_put, device=self.config.device, non_blocking=self.config.non_blocking) 
        
        self.callback = CallBack(checkpoint_dir=self.output_dir, num_profile_steps=10)
        self.model = model
        self.move()
        
    @property
    def model_params(self):
        return list(self.model.module.parameters()) if isinstance(self.model, DDP) else list(self.model.parameters())
    
    @property
    def should_checkpoint(self):
        return (self.state.num_batch_so_far % self.config.checkpoint_interval == 0 
                and self.state.num_batch_so_far > 0)
    
    @property
    def _device(self):
        return f"{self.config.device}:{get_rank()}"
    
    def move(self):
        if self.config.num_processes > 1 and not isinstance(self.model, DDP):
            self.model = DDP(self.model, device_ids=[self.config.device], find_unused_parameters=True)
        self.model.to(self._device, dtype=self.config.precision, non_blocking=self.config.non_blocking)
        self.model_input_name = list(inspect.signature(self.model.forward).parameters.keys())
        
    @contextmanager
    def potential_sync(self, model):
        '''Context manager for gradient accumulation'''
        with ExitStack() as stack:
            if not self.state.should_sync: 
                stack.enter_context(getattr(model, "no_sync")())        
            else: stack.enter_context(nullcontext())
            yield 

    def update(self, loss):
        '''Potentially gradient updating step'''
        
        assert isinstance(loss, torch.Tensor), "Loss should be a tensor, got {}".format(type(loss))
        
        if self.gradient_scaler is not None:
            self.gradient_scaler.scale(loss).backward()
        else:
            loss.backward()
        loss = loss.detach() 
       
        norm = 0. 
        self.has_been_synced = self.state.should_sync and not self._should_skip_update
        if self.has_been_synced:
            
            if self.gradient_scaler is not None:
                self.gradient_scaler.unscale_(self.optimizer)
            if self.config.max_grad_norm > 0.:
                norm = torch.nn.utils.clip_grad_norm_(self.model_params, self.config.max_grad_norm)
                norm = norm.item()
                
            if self.config.max_grad_value > 0.: 
                torch.nn.utils.clip_grad_value_(self.model_params, self.config.max_grad_value)
            
            if self.gradient_scaler is not None:
                self.gradient_scaler.step(self.optimizer)
                self.gradient_scaler.update() 
            else: 
                self.optimizer.step()
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(loss, epoch=self.state.num_epoch_so_far)
            self.scheduler.step()
            self.optimizer.zero_grad()
            self._gradient_steps += 1
            
        self.has_been_synced = False
        
        return loss, norm

    def compute_loss(self, batch: Dict[str, Tensor], weight={}):
        '''Override this method to compute loss'''
        batch = {k: v for k, v in batch.items() if k in self.model_input_name}
        batch = self.put_func(batch)
        loss = self.model(**batch)
        # multiply loss dict by weight, if len(weight) < len(loss), pad value with 1.
        weight = {k: weight.get(k, 1.) for k in loss.keys()} # assume weight keys are subset of loss keys
        loss = {k: v * weight[k] for k, v in loss.items()}
        
        self._should_skip_update = False
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            self._should_skip_update = True
            log(f"Loss is NaN or Inf, skipping update", rank=0)
        
        loss_reduced = torch.sum(torch.stack(list(loss.values())))
        
        return loss, loss_reduced 
    
    def train_step(self, batch, *args, **kwargs):
        '''Override this method to train step'''
        with self.potential_sync(self.model):
            with torch.autocast(
                enabled=self.config.mixed_precision != "off", dtype=self.config.mixed_precision
            ):
                loss, loss_reduced = self.compute_loss(batch)
                loss = loss / self.config.gradient_accumulation_steps
                loss, norm = self.update(loss)
        
        return (loss, loss_reduced, norm)
    
    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        '''Override this method to evaluate'''
        if self.eval_dataloader is None or self.config.max_eval_batch_steps == 0:
            return 
        
    
    def evaluate_step(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
    
    def fit(self, *args, **kwargs) -> Dict[str, Any]:
        
        try:
            self._fit(*args, **kwargs)
        
        except KeyboardInterrupt:
            log("KeyboardInterrupt", rank=0)
            ## save checkpoint
        

    def _fit(self, *args, **kwargs) -> Dict[str, Any]:
        
        set_seed(self.config.seed)
        profiler = self.callback.get_profiler_context()
        
        if self.callback.has_valid_checkpoint():
            self.model, self.optimizer, self.scheduler, self.state, self.config = self.callback.load_checkpoint(
                self.model, self.optimizer, self.scheduler, self.state, self.config
            )
            self.move()
        
        start_epoch = self.state.num_epoch_so_far
        end_epoch = self.config.num_epochs
        
        batch_to_skip = self.state.num_batch_so_far

        with profiler() as prof:
            
            self.evaluate() # evaluate before training
            
            for epoch in range(start_epoch, end_epoch):
                
                self.model.train()
                self.optimizer.zero_grad()
                running_loss = 0.

                for batch in self.train_dataloader:
                    
                    if batch_to_skip > 0:
                        batch_to_skip -= 1
                        if batch_to_skip == 0:
                            log(f"Skipping {self.state.num_batch_so_far} batches at epoch {epoch}")
                        continue
                    
                    loss, loss_reduced, norm = self.train_step(batch)
                    running_loss += loss_reduced.item()
                    
                    self.state.step()
                    prof.step()
                    
                    if self.state.should_terminate_on_epoch_end:
                        break
                    
                    if self.should_checkpoint:
                        self.callback.save_checkpoint(self.model, self.optimizer, self.scheduler, self.state, self.config)

                    monitor_dict = {
                        "LOSS/loss": loss_reduced.item(),
                        "gradient_norm": norm,
                        "learning_rate": self.optimizer.param_groups[0]['lr'],
                    }
                    for k, v in loss.item():
                        monitor_dict[f"LOSS/{k}"] = v.item()
                    
                    if self.state.num_batch_so_far % self.config.log_interval == 0:
                        log(f"Epoch {epoch}, batch {self.state.num_batch_so_far}, loss {running_loss / self.config.log_interval}")
                        running_loss = 0. 
                    
                if self.state.should_terminate:
                    break
                    
                self.state.step_epoch()
                log(f"Epoch {epoch} finished")
                
    def create_dataloader(self, *args, **kwargs):
        raise NotImplementedError
    