import os
import logging
from collections.abc import Mapping

from typing import Tuple
import torch
from torch.distributed import get_rank, get_world_size, is_initialized
from torch.utils.data import get_worker_info
from torch import nn

def log(msg: str, rank=0, export_log_file: str = None):
    msg = f"-----------> {msg}" 
    if get_worker_info() is not None and get_worker_info().id != 0:
        return # prevent multiple logs in multi-worker DataLoaders etc... 
    if is_initialized() and get_world_size() > 1:
        if rank != -1 and get_rank() != rank: # -1 for log on all ranks
            return
        msg = f"[Rank {get_rank()}] {msg}"
    logging.info(msg)
    if export_log_file is not None:
        with open(export_log_file, "a") as f:
            f.write(f"{msg}\n")

def set_seed(seed: int=42, full_determinism: bool=False): # this function do not do tf-seed related stuff
    
    import random, numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if full_determinism:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True)

        # Enable CUDNN deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_class(name: str, module: str="torch.optim"):
    '''Get class from module, and its arguments'''
    import importlib, inspect
    try:
        mod = importlib.import_module(module)
        class_ = getattr(mod, name)
    except Exception as exc:
        raise ImportError(f"Error: {exc}")
    
    params = inspect.signature(class_).parameters
    return class_, params

def create_optimizer_and_scheduler(
    module: nn.Module, 
    optimizer_name="AdamW", 
    scheduler_name="CosineAnnealingWarmRestarts", 
    learning_rate=5e-5,
    weight_decay=0.0, 
    optim_kwargs={"betas": (0.9, 0.999), "eps": 1e-8},
    scheduler_kwargs={"eta_min": 0.000002, "T_mult": 2, "T_0": 10} ,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    
    module = module.module if isinstance(module, nn.parallel.DistributedDataParallel) else module
    norm_modules = (
        nn.BatchNorm2d, nn.InstanceNorm2d, nn.BatchNorm1d, nn.LayerNorm,
        nn.InstanceNorm1d, nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm
    )
    embedding_modules = (nn.Embedding, nn.EmbeddingBag)
    decay_params = set()
    non_decay_params = set()
    
    for mn, m in module.named_modules():
        for k, v in m.named_parameters():
            if "bias" in k or isinstance(m, norm_modules) or isinstance(m, embedding_modules):
                non_decay_params.add(v)
            elif "weight" in k:
                decay_params.add(v)
            else:
                raise ValueError(f"Unrecognized parameter: {k}")
            
    # now, some parameter will fall into both categories, so we need to remove them from non_decay_params
    non_decay_params = non_decay_params - decay_params
    
    group = [
        {"params": list(decay_params), "weight_decay": weight_decay},
        {"params": list(non_decay_params), "weight_decay": 0.0}
    ]

    opt, opt_params = get_class(optimizer_name)
    sch, sch_params = get_class(scheduler_name, "torch.optim.lr_scheduler")

    assert all(k in opt_params for k in optim_kwargs)
    assert all(k in sch_params for k in scheduler_kwargs)

    optimizer = opt(group, lr=learning_rate, **optim_kwargs)
    scheduler = sch(optimizer, **scheduler_kwargs)
    
    return optimizer, scheduler


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