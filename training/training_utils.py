
import inspect
import torch 
from torch import nn

def create_default_optim_and_scheduler(model: nn.Module, learning_rate:float=5e-5):
    '''Create Adam and CosineAnnealingWarmRestarts'''
    adam_kwargs = {
        "betas": [0.9, 0.999],
        "eps": 1e-6,
        "weight_decay": 0.0,
        "amsgrad": True
    }
    scheduler_kwargs = {
        "T_0": 10,
        "T_mult": 1,
        "eta_min": 1e-6
    }

    return create_optimizer_and_scheduler(
        model, "Adam", "CosineAnnealingWarmRestarts", learning_rate=learning_rate, 
        optim_kwargs=adam_kwargs, scheduler_kwargs=scheduler_kwargs
    )

def create_optimizer_and_scheduler(
    model: nn.Module, optim_cls, schedule_cls, learning_rate=5e-5, weight_decay=0.009, 
    optim_kwargs=None, scheduler_kwargs=None
):
    # norm_modules = (
    #     nn.BatchNorm2d, nn.InstanceNorm2d, nn.BatchNorm1d, nn.LayerNorm,
    #     nn.InstanceNorm1d, nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm
    # )
    # embedding_modules = (nn.Embedding, nn.EmbeddingBag)
    decay_params = set()
    non_decay_params = set()
    
    # for mn, m in model.named_modules():
    #     for k, v in m.named_parameters():
    #         if "bias" in k or isinstance(m, norm_modules) or isinstance(m, embedding_modules):
    #             non_decay_params.add(v)
    #         elif "weight" in k:
    #             decay_params.add(v)
    #         else:
    #             raise ValueError(f"Unrecognized parameter: {k}")
    for k, v in model.named_parameters():
        if "weight" in k:
            decay_params.add(v)
        else:
            non_decay_params.add(v)
    
    param_groups = [
        {"params": list(decay_params), "weight_decay": weight_decay},
        {"params": list(non_decay_params), "weight_decay": 0.0}
    ]
    
    optim_kwargs = optim_kwargs or {}
    scheduler_kwargs = scheduler_kwargs or {}

    optimizer = getattr(torch.optim, optim_cls)(param_groups, lr=learning_rate, **optim_kwargs)
    scheduler = getattr(torch.optim.lr_scheduler, schedule_cls)(optimizer, **scheduler_kwargs)

    return optimizer, scheduler
