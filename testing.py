# testing dataset 

import time
import torch
import torch.distributed as dist

from src.data_utility import _recurse_func, _put, all_gather_object
from src.gigaspeech_dataset import IterableGigaSpeech

def ensure_correct_batch():
    
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', init_method='env://')
    
    batch_size = 4
    gigaspeech = IterableGigaSpeech(batch_size)
    gigaspeech_iter = iter(gigaspeech) 
    sample = next(gigaspeech_iter)
    
    print(f"Rank: {dist.get_rank()} get sample: {sample['text'][0]}")
    gathered = all_gather_object(sample, flatten=False) # is list of dicts with keys 'text' and 'audio' etc...
    #when flatten it will has length of world_size * num_sample_key
    n_workers = dist.get_world_size()
    
    unique = set() 
    all_text = []
    for g in gathered:
        list_text = g['text']
        for text in list_text:
            unique.add(text)
            all_text.append(text)
    assert len(unique) == n_workers * batch_size, f"Unique text: {len(unique)} != {n_workers * batch_size}"


if __name__ == "__main__":
    
    ensure_correct_batch() # use torchrun --nproc_per_node=2 testing.py

