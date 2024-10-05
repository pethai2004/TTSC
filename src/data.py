
import os
import json
from dataclasses import dataclass
from typing import List
from functools import partial
from tokenizers import Tokenizer
import numpy as np
import torchaudio
import torch 
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, DownloadConfig
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from src.preprocessing import preprocess_text, preprocess_audio
from src.data_utility import _put

@dataclass
class AudioTrainingConfig:
    
    output_dir : str = './TrainingOutputDir'
    trial_name : str = 'default_trial'    
    
    batch_size: int = 32
    gpt_num_audio_tokens : int = 1024
    num_epochs : int = 30
    max_batch_step : int = 10_000
    max_eval_step : int = 2000
    learning_rate : float = 5e-6

    mel_norm_path : str = './assets/mel_norm.json'
    model_checkpoint: str = None
    
@dataclass
class AudioDataConfig:

    batch_size: int = 32 # per device batch size
    sampling_rate: int = 16000
    wav_dtype = torch.float16
    max_wav_sec: int = 15 
    min_wav_sec: int = 5
    loader_batch_size: int = 512
    shuffle: bool = False 
    seed: int = 42
    tokenizer_path: str = './assets/tokenizing/tokenizer.json'
    token_pad_id: int = 0
    pad_token: str = '[PAD]'
    wav_pad_value: int = 0
    
    subset: str = 'dev'
    split: str = 'validation'
    streaming: bool = True
    huggingface_token: str = None
    num_proc: int = 16
    
    should_put: bool= False
    device : torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    non_blocking: bool = True 
    

########################################################################################
class AudioData(IterableDataset):
    
    def __init__(self, config: AudioDataConfig, cache_dir: str = '.assets/.cache'):
        super().__init__()
        self.repo_name = None
        self.config = config
        self.cache_dir = cache_dir
        self.tokenizer = None 
        if config.tokenizer_path is not None:
            self.tokenizer: Tokenizer = Tokenizer.from_file(config.tokenizer_path)
            self.tokenizer.enable_padding(pad_id=config.token_pad_id, pad_token=config.pad_token, pad_to_multiple_of=8)
        self._data = None # will always be `datasets.IterableDataset` object for now that yield batched data
        self.batch_so_far = 0 
        self._put_func = partial(_put, device=config.device, dtype=config.wav_dtype, non_blocking=self.config.non_blocking)

    def __iter__(self):
        worker_info = get_worker_info()
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        shard_id = rank * num_workers + worker_id
        total_shards = world_size * num_workers

        for i, batch in enumerate(self._data):
            if i % total_shards == shard_id:
                yield batch

    def __next__(self):
        assert self.config.streaming 
        if self.config.streaming:
            self.batch_so_far += 1
        return next(self._data)

    def _split_data_for_worker(self, worker_id, num_workers):
        return self._data.shard(num_shards=num_workers, index=worker_id)

    def filter(self, x, *args, **kwargs):
        if x['duration'] > self.config.max_wav_sec or x['duration'] < self.config.min_wav_sec:
            return False
        if x['text'] == "<MUSIC>":
            return False
        if x['text'] == '' or len(x['text']) < 5: # some hard threshold
            return False
        return True 
    
    @staticmethod
    def collate_fn(x, padding_value=0, dtype=torch.float16, device='cuda', non_blocking=True):
        x['wav'] = pad_sequence([torch.tensor(wav, dtype=dtype, device=device) for wav in x['wav']], 
                                batch_first=True, padding_value=padding_value)
        x['wav'] = x['wav'].unsqueeze(1)
        x = _put(x, device=device, dtype=dtype, non_blocking=non_blocking)
        return x
    
    def init_load_data(self):
        '''Reset the dataset to the initial state / or load the dataset'''
        raise NotImplementedError
    
    def reset(self, clear_cache: bool = False):
        '''Reset the dataset to the initial state (call at the end of epoch)'''
        assert self.config.streaming
        del self._data
        if clear_cache:
            # recursively delete the cache directory and file
            import shutil
            shutil.rmtree(self.cache_dir)
        self.init_load_data()
        
class GigaSpeech(AudioData):

    def __init__(self, config: AudioDataConfig, cache_dir: str = './assets/.cache/gigaspeech'):
        super().__init__(config, cache_dir)
        # valid the provided data subset and split
        self.repo_name = 'speechcolab/gigaspeech'
        assert config.huggingface_token is not None
        available_subset = eval(open('./assets/gigaspeech_subset.txt', 'r').read())
        assert [config.subset, config.split] in available_subset, f"Invalid subset/split: {config.subset}/{config.split}, available: {available_subset}"
        assert config.sampling_rate == 16000, 'Support only 16kHz sampling rate'
        
    def init_load_data(self):
        
        self._load_config = DownloadConfig(cache_dir=self.cache_dir, resume_download=True, num_proc=self.config.num_proc)
        self._data = load_dataset(self.repo_name, self.config.subset, split=self.config.split, streaming=self.config.streaming, 
                                  download_config=self._load_config, trust_remote_code=True)
        self._data = self._data.map(
            lambda _x : self.preprocess(_x),
            remove_columns=['audio', 'segment_id', 'speaker', 'original_full_path', 'url', 'title', 'audio_id', 'begin_time', 'end_time'],
            batched=True, batch_size=self.config.loader_batch_size
        ) # map before filter since filter need 'duration' key which is created in preprocess
        self._data = self._data.filter(self.filter, batched=False, batch_size=self.config.loader_batch_size)  
        if self.config.shuffle and not self.config.streaming:
            self._data = self._data.shuffle(seed=self.config.seed)
        self._data = self._data.with_format('torch')
        self._data = self._data.iter(batch_size=self.config.batch_size, drop_last_batch=True)
    
    def preprocess(self, x, *args, **kwargs):
        
        audio_value = x['audio'] 
        # array = [value['array'].astype(self.config.wav_dtype) for value in audio_value]
        array = [value['array'] for value in audio_value]
        # array = pad_sequence([torch.tensor(wav, device=self.config.device, dtype=self.config.wav_dtype) for wav in array], 
        #                      batch_first=True, padding_value=self.config.wav_pad_value)
        # array = array.unsqueeze(1) # add channel dimension
        
        sampling_rate = [value['sampling_rate'] for value in audio_value]
        duration = [len(A) / sr for A, sr in zip(array, sampling_rate)]
        text = list(map(preprocess_text, x['text']))
        output = {'wav': array, 'text': text, 'sampling_rate': sampling_rate, 'duration': duration}
        
        if self.tokenizer is not None:
            text = self.tokenizer.encode_batch_fast(text)
            output['token_ids'] = [t.ids for t in text]
            output['attention_mask'] = [t.attention_mask for t in text]
            
        return output   
    
class PeopleSpeech(AudioData):
        
        def __init__(self, config: AudioDataConfig, cache_dir: str = './assets/.cache/peoplespeech'):
            super().__init__(config, cache_dir)
            self.repo_name = 'MLCommons/peoples_speech'
            assert config.split in ['train', 'validation', 'test']
            assert config.sampling_rate == 16000, 'Support only 16kHz sampling rate'
            
        def init_load_data(self):
            self._load_config = DownloadConfig(cache_dir=self.cache_dir, resume_download=True, num_proc=self.config.num_proc) 
            self._data = load_dataset(
                self.repo_name, split=self.config.split, streaming=self.config.streaming, 
                download_config=self._load_config, trust_remote_code=True
            )
            self._data = self._data.map(
                lambda _x : self.preprocess(_x),
                remove_columns=['audio', 'id', 'duration_ms'],
                batched=True, batch_size=self.config.loader_batch_size
            ) # map before filter since filter need 'duration' key which is created in preprocess
            self._data = self._data.filter(self.filter, batched=False, batch_size=self.config.loader_batch_size)  
            if self.config.shuffle and not self.config.streaming:
                self._data = self._data.shuffle(seed=self.config.seed)
            self._data = self._data.with_format('torch')
            self._data = self._data.iter(batch_size=self.config.batch_size, drop_last_batch=True)
    
            
        def preprocess(self, x, *args, **kwargs):
            
            
            audio_value = x['audio']
            array = [value['array'] for value in audio_value]
            duration = [v / 1000 for v in x['duration_ms']]
            
            sampling_rate = [value['sampling_rate'] for value in audio_value]
            text = list(map(preprocess_text, x['text']))
            output = {'wav': array, 'text': text, 'sampling_rate': sampling_rate, 'duration': duration}
            
            if self.tokenizer is not None:
                text = self.tokenizer.encode_batch_fast(text)
                output['token_ids'] = [t.ids for t in text]
                output['attention_mask'] = [t.attention_mask for t in text]
            
            return output
    
class _LJSpeechDataset: 
    
    @staticmethod
    def _load_dataset(self, target_dir='./assets/'):
        from src.dataset_loader import load_ljspeech
        load_ljspeech(target_dir) 
        
    def __init__(self, dataset_path: str='./assets/LJSpeech-1.1', sampling_rate=16000):
        self.dataset_path = dataset_path
        self.sampling_rate = sampling_rate
        torch.set_num_threads(1)
        
    def _init_load_data(self):
        
        if not os.path.exists(self.dataset_path):
            self._load_dataset(self.dataset_path)
            
        samples = json.loads(open(os.path.join(self.dataset_path, 'metadata.json')).read())
        self.samples = []
        
        for s in samples:
            try:
                __class__.load_validate_audio(s['audio_file'], self.sampling_rate)
            except Exception as e:
                continue
            self.samples.append(s)

        self.sampling_rate = self.sampling_rate
        self._failed_idx = set()
        
    def __len__(self):  
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Always assume that the audio data is valid, otherwise ignore
        
        if idx in self._failed_idx:
            return self.__getitem__(idx + 1)
        try:
            wav = __class__.load_validate_audio(self.samples[idx]['audio_file'], self.sampling_rate)
        except Exception as e:
            if not idx in self._failed_idx:
                self._failed_idx.add(idx)
            #try calling the next index
            return self.__getitem__(idx + 1)
        
        wav_length = torch.tensor(wav.shape[-1], dtype=torch.long)
        audio_dict = {"wav": wav, 'wav_length': wav_length, 'audio_path': self.samples[idx]['audio_file']}
        return audio_dict

    @staticmethod
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
    
    @staticmethod
    def audio_collate_fn(batch):
        '''Collate function for DVAE dataset'''
    
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}
        batch["wav_length"] = torch.stack(batch["wav_length"])
        # note batch['wav'] has shape [1, seq_length]
        wav_padded = pad_sequence([wav[0] for wav in batch["wav"]], batch_first=True)
        wav_padded = wav_padded.unsqueeze(1)    # Add a channel dimension
        batch["wav"] = wav_padded
        # now has shape
        return batch

class LJSpeech:
    
    def __init__(self):
        raise NotImplementedError