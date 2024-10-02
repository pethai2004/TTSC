import logging
from dataclasses import dataclass
from collections.abc import Mapping

import torch
import torch.distributed as dist
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

@dataclass
class AudioTrainingConfig:
    
    num_epochs: int = 18

    dataset_dir: str = "./assets/LJSpeech-1.1"
    gradient_clip_norm: float = 0.8
    gpt_num_audio_tokens : int = 1024
    learning_rate: float = 5e-6
    seed = 42
    sampling_rate: int = 22050
    max_wav_length: int = 255955 # 11.6 secs
    max_training_example: int = -1
    max_eval_step: int = -1
    per_device_batch_size: int = 32
    pin_memory: bool = True
    persistent_workers: bool= True

def create_audio_dataloader(
    meta_data,
    per_device_batch_size=32,
    sampling_rate=22050,
    max_wav_length = 255995,
    max_training_example=-1,
    max_eval_step=-1,
    is_train=False,
    pin_memory=True
) -> DataLoader:
    
    limit = max_training_example if is_train else max_eval_step
    meta_data = meta_data[:limit] 
    data = AudioDataset(meta_data, max_wav_length, sampling_rate)
    if len(data) < len(meta_data):
        logging.getLogger().warning(f"Cound only load {len(data)} out of {len(meta_data)} samples")
        getattr( "max_training_examples" if is_train else "max_eval_steps") == len(data) 
    
    dataloader = DataLoader(
        data, batch_size=per_device_batch_size, 
        collate_fn=audio_collate_fn, num_workers=4, pin_memory=pin_memory, 
        persistent_workers=True
    )
    return dataloader

def _recurse_func(func, inputs, *arg, test_type=torch.Tensor, **kwargs):
    """Recursively apply the function to the inputs where output structure is preserved."""
    if isinstance(inputs, Mapping):   
        return type(inputs)(
            {k : _recurse_func(func, v, *arg, **kwargs) for k, v in inputs.items()}
        )
    elif isinstance(inputs, (list, tuple)):
        return type(inputs)(_recurse_func(func, t, *arg, **kwargs) for t in inputs)
    elif isinstance(inputs, torch.Tensor):
        return func(inputs, *arg, **kwargs)
    elif isinstance(inputs, test_type):
        return func(inputs, *arg, **kwargs)
    else:
        raise TypeError(f"Unsupported type {type(inputs)} passed to {func.__name__}.")
    
def _put(tensor, device, dtype=None, non_blocking=True):
    '''Put potential Tensor on device (recurse)'''
    try: # will try to convert as soon as possible
        return tensor.to(dtype=dtype, device=device, non_blocking=non_blocking)
    except: 
        pass 
    
    if isinstance(tensor, torch.Tensor) or hasattr(tensor, "to"):
        return tensor.to(dtype=dtype, device=device, non_blocking=non_blocking)
    elif isinstance(tensor, (list, tuple)):
        try: 
            tensor = torch.tensor(tensor, device=device)
            return tensor
        except: 
            return type(tensor)(
                _put(t, device, non_blocking=non_blocking) for t in tensor
            )
    elif isinstance(tensor, Mapping):
        return type(tensor)(
            {k: _put(v, device, non_blocking=non_blocking) for k, v in tensor.items()}
        )
    else:
        return tensor


def combine_dataset(dataset_list):
    
    return None 