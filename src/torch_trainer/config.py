# config
import os
import json
import time
from dataclasses import dataclass, asdict, fields

import torch
from trainer_utility import log

CHECKPOINT_PREFIX = "checkpoint_"
MODEL_PATH = "model.pth"
TRAINING_STATE_PATH = "training_state.pkl"
TRAINING_CONFIG_PATH = "training_config.pkl"
OPTIMIZER_PATH = "optimizer.pth"
SCHEDULE_PATH = "scheduler.pth"

@dataclass
class State:
    
    num_epoch_so_far: int = 0
    num_batch_so_far: int = 0 # per epoch and per process
    
    max_batch_steps: int = None # indicate that we have not yet set this via config
    max_epoch_steps: int = None
    per_device_batch_size: int = None
    num_processes: int = None
    gradient_accumulation_steps: int = None

    @property
    def should_sync(self):
        '''Whether the current step should be gradient synchronization'''
        return self.num_batch_so_far % self.gradient_accumulation_steps == 0

    @property
    def should_terminate_on_epoch(self):
        '''Whether the current epoch should terminate'''
        return self.max_batch_steps != -1 and self.num_batch_so_far >= self.max_batch_steps
    
    @property
    def should_terminate(self):
        '''Whether the training should terminate'''
        if self.num_epoch_so_far > self.max_epoch_steps and self.should_terminate_on_epoch:
            return True # reach max epoch and max batch steps at current epoch
    
    @property
    def global_batch_so_far(self):
        return self.num_batch_so_far * self.num_epoch_so_far * self.num_processes
    
    @property
    def global_update_step(self):
        return self.num_batch_so_far * self.num_epoch_so_far // self.gradient_accumulation_steps
    
    def step(self):
        '''Call every model forward pass'''
        self.num_batch_so_far += 1
    
    def step_epoch(self):
        '''Call every epoch'''
        self.num_epoch_so_far += 1
        self.num_batch_so_far = 0

    def set(self, config):
        '''Set the state from config'''
        self.max_batch_steps = config.max_train_batch_steps
        self.max_epoch_steps = config.num_epochs
        self.per_device_batch_size = config.per_device_batch_size
        self.num_processes = config.num_processes
        self.gradient_accumulation_steps = config.gradient_accumulation_steps

    def save(self, output_dir):
        path = os.path.join(output_dir, "state.json")
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)
    
    def load_checkpoint(self, path):
        '''Load JSON checkpoint'''
        with open(path, 'r') as f:
            value = json.load(f)
            ### TODO:
            if value['num_processes'] != self.num_processes:
                raise NotImplementedError(f"No implementation for loading checkpoint with different num_processes")
            if value['per_device_batch_size'] != self.per_device_batch_size:
                raise NotImplementedError(f"No implementation for loading checkpoint with different per_device_batch_size")

            self.__dict__.update(value)

@dataclass
class TrainingConfig:
    
    project_name: str = "DefaultProject"
    trial_name: str = ""
    
    learning_rate: float = 5e-5
    num_epochs: int = 3
    per_device_batch_size: int = 32
    max_eval_batch_steps = -1
    max_train_batch_steps = -1 # per epoch

    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    max_grad_value: float = 0.0
    
    precision = torch.float16
    seed = 42
    full_determinism = False
    mixed_precision = torch.float16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    backend: str = "nccl" if torch.cuda.is_available() else "gloo"
    num_processes = -1
    num_workers = 4
    pin_memory = True
    non_blocking = True
    
    max_checkpoint = 5
    log_interval = 20
    checkpoint_interval = 200
    num_profiled_steps = 5
    
    def _post_init(self):
        '''Call in Trainer class'''
        if self.trial_name == "":
            self.trial_name = f"trial_{time.strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = os.path.join(self.project_name, self.trial_name)
        os.makedirs(self.output_dir)
        
        if self.device == 'cuda':
            assert torch.cuda.is_available()
            self.num_processes = torch.cuda.device_count() if self.num_processes == -1 \
                else min(self.num_processes, torch.cuda.device_count())
        else:
            self.num_processes = max(1, self.num_processes)

        self.global_batch_size = self.per_device_batch_size * self.num_processes
        assert self.mixed_precision in ["off", torch.float16, torch.bfloat16]
        if self.mixed_precision == "off":
            self.precision = torch.float32
            
        self.is_post_init = True
        
    def print_all_field(self):
        if not hasattr(self, 'is_post_init') or not self.is_post_init:
            return
        for f in fields(TrainingConfig):
            log(f"{f.name:<30}: {getattr(self, f.name)}")
            
    def save(self, output_dir):
        path = os.path.join(output_dir, "training_config.json")
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)
    
    @classmethod
    def load(cls, output_dir):
        with open(output_dir, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
                
    def to_dict(self):
        return asdict(self)
    
class AudioTrainingConfig(TrainingConfig):
    
    sampling_rate: int = 16000
    dvae_sampling_rate: int = 16000

def parse_arg(parser, config, args=None):
    """Parse the arguments and set the fields of the TrainingConfig."""

    for f in fields(config):
        help_text = f.metadata.get("help", "No help available")
        parser.add_argument(f'--{f.name}', type=type(f.default), default=f.default, help=help_text)

    args = parser.parse_args(args)
    parsed_dict = vars(args)

    for field in parsed_dict:
        # if key is the field of the TrainingConfig, set it, else ignore
        if field in asdict(config) and parsed_dict[field] is not None:
            print(f"Setting {field} to {parsed_dict[field]}")
            setattr(config, field, parsed_dict[field])