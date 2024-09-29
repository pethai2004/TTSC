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
    
def load_validate_audio(audiopath, sampling_rate):
    audio, lsr = torchaudio.load(audiopath)

    if audio.size(0) != 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)
    if torch.any(audio > 10) or not torch.any(audio < 0):
        raise ValueError(f"Audio not normalized: {audio.min()}, {audio.max()}")

    audio.clip_(-1, 1)
    if audio.shape[-1] < (0.5 * sampling_rate):
        raise ValueError(f"Audio too short: {audio.shape[-1]}")
    
    return audio

class AudioDataset(Dataset):
    
    def __init__(self, samples, max_wav_lengt=255955, sampling_rate=22050):
        torch.set_num_threads(1)
        self.samples = []
        for s in samples:
            try:
                load_validate_audio(s['audio_file'], sampling_rate)
            except Exception as e:
                logging.warning(f"Error loading {s['audio_file']}: {e}")
                continue
            self.samples.append(s)
        logging.info(f"Loaded and validated {len(self.samples)} samples out of {len(samples)}")
            
        self.max_wav_length = max_wav_lengt # 255955 (approx 11.6 secs) which is supported by model (see in model.arg)
        self.sampling_rate = sampling_rate
        self._failed_idx = set()
        
    def __len__(self):  
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Always assume that the audio data is valid, otherwise ignore
        
        if idx in self._failed_idx:
            return self.__getitem__(idx + 1)
        try:
            wav = load_validate_audio(self.samples[idx]['audio_file'], self.sampling_rate)
        except Exception as e:
            logging.error(f"Error loading {self.samples[idx]['audio_file']}: {e}")
            if not idx in self._failed_idx:
                self._failed_idx.add(idx)
            #try calling the next index
            return self.__getitem__(idx + 1)
        
        wav_length = torch.tensor(wav.shape[-1], dtype=torch.long)
        audio_dict = {"wav": wav, 'wav_length': wav_length, 'audio_path': self.samples[idx]['audio_file']}
        return audio_dict

def audio_collate_fn(batch):
    '''Collate function for DVAE dataset'''
    
    batch = {k: [dic[k] for dic in batch] for k in batch[0]}
    batch["wav_length"] = torch.stack(batch["wav_length"])
    # note batch['wav'] has shape [1, seq_length]
    wav_padded = pad_sequence([wav[0] for wav in batch["wav"]], batch_first=True)
    wav_padded = wav_padded.unsqueeze(1)    # Add a channel dimension
    batch["wav"] = wav_padded
    
    return batch

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
