import logging
from collections.abc import Mapping

import torch
import torch.distributed as dist
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset

def repeat(data, n: int=-1):
    '''Useful for training dataset wrapped with iter()'''
    c = 0
    sampled_so_far = 0
    while True:
        for data in data:
            sampled_so_far += 1
            yield {
                "data": data,
                "sampled_so_far": sampled_so_far,
                "num_epoch": c
            }
        c += 1
        if c == n and n >= 0:
            raise StopIteration(f"Done with num_epoch: {c} num_sampled_so_far: {sampled_so_far}")
        
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

def all_gather_object(input, flatten=True):
    target = [None] * dist.get_world_size()
    dist.all_gather_object(target, input)
    if flatten:
        return [t for sublist in target for t in sublist]
    else:
        return target

def all_gather_object_(input, flatten=True): # add shape and data config wrap check
    '''Recursive `all_gather_object`'''
    return _recurse_func(all_gather_object, input, flatten=flatten)

