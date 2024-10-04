# testing dataset 

import time

import torch.distributed as dist

from src.data_utility import _recurse_func, _put, all_gather_object
from src.data import GigaSpeech, AudioDataConfig

def ensure_correct_batch():
    
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', init_method='env://')
    
    data_config = AudioDataConfig(batch_size=4, huggingface_token='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',)
    gigaspeech = GigaSpeech(config=data_config)
    gigaspeech.init_load_data()

    gigaspeech = iter(gigaspeech)
    
    sample = next(gigaspeech)
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
    assert len(unique) == n_workers * data_config.batch_size, f"Unique text: {len(unique)} != {n_workers * data_config.batch_size}"

def test_dataset_generation_time(num_iterations=10000):
    
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', init_method='env://')

    data_config = AudioDataConfig(batch_size=64, huggingface_token='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',)
    gigaspeech = GigaSpeech(config=data_config)
    gigaspeech.init_load_data()
    gigaspeech = iter(gigaspeech) # do not forget to convert to iterator
    
    start_time = time.time()
    
    # Iterate over the dataset for the specified number of iterations
    for i in range(num_iterations):
        try:
            sample = next(gigaspeech)
        except StopIteration:
            # If the dataset is exhausted, break early
            break
        if i % 100 == 0:
            print(f"Iteration {i}: with time {time.time() - start_time:.2f} seconds")
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"Time taken for {num_iterations} iterations: {total_time:.2f} seconds")
    return total_time

if __name__ == "__main__":
    
    ensure_correct_batch() # use torchrun --nproc_per_node=2 testing.py
    test_dataset_generation_time()
