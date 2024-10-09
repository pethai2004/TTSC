# GigaSpeech Dataset 
import regex as re
import random
import os
import requests

from functools import partial
import unicodedata
import logging
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchaudio.functional import resample
from tokenizers import Tokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, Sampler, DataLoader

from src.data_utility import _recurse_func, _put

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

available_subset = eval(open('./assets/gigaspeech_subset.txt', 'r').read())

replace_abbrev = {"Mr.": "Mister", "Mrs.": "Missus", "Ms.": "Miss", "Dr.": "Doctor", "Prof.": "Professor",
                  "No.": "Number", "U.S.": "United States", "U.K.": "United Kingdom"}  # etc
punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
gigaspeech_replacement = {'<COMMA>': ',', '<PERIOD>': '.', '<QUESTIONMARK>': '?', '<EXCLAMATIONPOINT>': '!',
                          '<SIL>': '', '<MUSIC>': '', '<NOISE>': '', '<OTHER>': ''}

pat = re.compile('|'.join(re.escape(key) for key in gigaspeech_replacement.keys()))

def preprocess_text(input, lower=True, remove_non_english=True, should_abbreviate=True):
    def _preprocess(text: str):
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        replace_eng = '[UNK]' if remove_non_english else ''
        text = re.sub(r'[^A-Za-z0-9' + punctuation + ']+', replace_eng, text)
        text = re.sub(r'\s+', ' ', text)

        if should_abbreviate:
            pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in replace_abbrev.keys()) + r')\b')
            text = pattern.sub(lambda m: replace_abbrev[m.group(0)], text)
        
        text = pat.sub(lambda x: gigaspeech_replacement[x.group()], text)
        if lower:
            text = text.lower() # apply the lastest so that we can replace the uppercase abbreviations
        return text

    return _recurse_func(_preprocess, input, test_type=str)

def preprocess(x, tokenizer: Tokenizer=None):
    array = [value['array'] for value in x['audio']]
    sampling_rate = [value['sampling_rate'] for value in x['audio']]
    duration = [len(A) / sr for A, sr in zip(array, sampling_rate)]
    text = list(map(preprocess_text, x['text']))
    
    output = {'wav': array, 'text': text, 'sampling_rate': sampling_rate, 'duration': duration}
    if tokenizer is not None:
        text_tokens = tokenizer.encode_batch_fast(text)
        output['token_ids'] = [t.ids for t in text_tokens]
        output['attention_mask'] = [t.attention_mask for t in text_tokens]
    
    return output

def load_gigaspeech_files(token, subset="s", split="train", target_dir="gigaspeech"):
    url = f"https://huggingface.co/api/datasets/speechcolab/gigaspeech/parquet/{subset}/{split}"
    headers = {'Authorization': f'Bearer {token}'}
    mapped_path = {}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to retrieve file list. Status code: {response.status_code}")
        return mapped_path

    file_list = response.json()
    print(f"Found {len(file_list)} files")
    
    target_dir = os.path.join(target_dir, subset, split)
    os.makedirs(target_dir, exist_ok=True)
    
    for file_url in file_list:
        path = file_url.split('/')[-1]
        file_path = os.path.abspath(os.path.join(target_dir, path))
        mapped_path[file_url] = file_path
        
        if os.path.exists(file_path):
            print(f"File {file_path} already exists")
            continue
        
        file_data = requests.get(file_url, headers=headers)
        with open(file_path, 'wb') as f:
            f.write(file_data.content)
            print(f"Downloaded {file_path}")

    return target_dir, mapped_path
class GigaSpeech(Dataset):
    '''GigaSpeech Dataset'''
    
    def __init__(
        self, 
        dataset_dir: str, 
        tokenizer_path: str='TTSC/assets/tokenizing/tokenizer.json',
        max_samples=-1, 
        target_sampling_rate=16000,
        min_wav_length=80_000, # ~ 5 seconds
        max_wav_length=220_000, # ~ 15 seconds
        min_condition_length=80_000, # ~ 5 seconds
        max_condition_length=220_000, # ~ 15 seconds
        min_text_length=5, # tokens
        max_text_length=100,
        wav_dtype=torch.float32,
        is_eval=False, 
        **kwargs
    ): # TODO: maybe create seperate dict key/class argument for conditioning wav (not same as wav)
        super().__init__(**kwargs)
        
        self.dataset_dir = dataset_dir
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_samples = max_samples
        self.target_sampling_rate = target_sampling_rate
        
        self.min_wav_length = min_wav_length
        self.max_wav_length = max_wav_length
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.max_condition_length = max_condition_length
        self.min_condition_length = min_condition_length
        
        self.wav_dtype = wav_dtype
        self.is_eval = is_eval

        self.num_proc = 4
        self.data = None 
        self.process_batch_size = 512 
        
        os.makedirs(self.dataset_dir, exist_ok=True)
        parquets = [f for f in os.listdir(self.dataset_dir) if f.endswith('.parquet')]
        assert len(parquets) > 0, f"No parquet files found in {self.dataset_dir}"
        
        self.cache_dir = os.path.join(self.dataset_dir, '.cache') 
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Cache directory: {self.cache_dir}")

        self.create_dataset()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx] # without conditioning
        cond, cond_length, cond_idxs = self.create_conditioning(
            sample['wav'], self.max_condition_length, self.min_condition_length, self.is_eval
        )
        sample['conditioning'] = cond.unsqueeze(0)
        sample['conditioning_length'] = cond_length
        sample['conditioning_idxs'] = cond_idxs
        
        return sample
    
    @staticmethod
    def _load_data(token, subset="s", split="train", target_dir="gigaspeech"):
        target_dir, _ = load_gigaspeech_files(token, subset=subset, split=split, target_dir=target_dir)
        return target_dir
    
    def create_dataset(self):
        
        self.data = load_dataset(path=self.dataset_dir, cache_dir=self.cache_dir, num_proc=self.num_proc)['train']
        if self.max_samples > 0:
            self.data = self.data.select(range(self.max_samples))
        self.data = self.data.remove_columns(
            ['segment_id', 'speaker', 'begin_time', 'end_time', 'audio_id', 'title', 'url', 'source', 'category', 'original_full_path']
        )
        self.data = self.data.map(
            self.preprocess, batched=True, num_proc=self.num_proc, remove_columns=['audio'], batch_size=self.process_batch_size
        )
        self.data = self.data.with_format('torch')
        self.data = self.data.filter(self.filter, num_proc=self.num_proc, batch_size=self.process_batch_size)
    
    def preprocess(self, x): # batch process
        
        audio = [v['array'] for v in x['audio']]
        sampling_rate = [v['sampling_rate'] for v in x['audio']] 
        
        for i, (A, sr) in enumerate(zip(audio, sampling_rate)):
            if sr != self.target_sampling_rate:
                audio[i] = resample(A, sr, self.target_sampling_rate)
                sampling_rate[i] = self.target_sampling_rate
        
        duration = [len(A) / sr * 1000 for A, sr in zip(audio, sampling_rate)]
        
        text = list(map(preprocess_text, x['text']))
        text_tokens = self.tokenizer.encode_batch_fast(text)
        token_ids = [t.ids for t in text_tokens]
        
        return {
            'wav': audio, 
            'text': text, 
            'sampling_rate': sampling_rate, 
            'duration': duration,
            'token_ids': token_ids,
        }
    
    def filter(self, x):
        if not x['wav'].size(-1) > self.min_wav_length and not x['wav'].size(-1) < self.max_wav_length :
            return False    
        if not len(x['token_ids']) > self.min_text_length and not len(x['token_ids']) < self.max_text_length:
            return False
        if x['text'] == "<music>" or x['text'] == '' or len(x['text']) < 5:
            return False
        return True
    
    @staticmethod 
    def create_conditioning(x, max_length, min_length, is_eval=False):
        # create random subset of original wav as conditioning, if is_eval, then use middle of wav
        length = random.randint(min_length, max_length) if not is_eval else int((min_length + max_length) / 2)
        
        gap = x.shape[-1] - length
        if gap < 0: 
            length = x.shape[-1] // 2 # if wav is too short, use half of it
        gap = x.shape[-1] - length

        rand_start = random.randint(0, gap) if not is_eval else 0
        rand_end = rand_start + length
        x = x[rand_start:rand_end]
        x = F.pad(x, pad=(0, max_length - x.shape[-1]))
        cond_idxs = [rand_start, rand_end]
        
        return x, x.shape[-1], cond_idxs
        
    def collate_func(self, x, padding_value=0, device='cpu', return_attention_mask=False):
        
        batched_token_ids = [d['token_ids'] for d in x]
        batched_text = [d['text'] for d in x]
        batched_wav = [d['wav'] for d in x]
        sampling_rate = [d['sampling_rate'] for d in x]
        duration = [d['duration'] for d in x]
                # Create conditioning for each sample in the batch
        conditionings = []
        conditioning_lengths = []
        conditioning_idxs = []
        for wav in batched_wav:
            cond, cond_length, cond_idxs = self.create_conditioning(
                wav, self.max_condition_length, self.min_condition_length, self.is_eval
            )
            conditionings.append(cond.unsqueeze(0))
            conditioning_lengths.append(cond_length)
            conditioning_idxs.append(cond_idxs)

        # Pad sequences
        batched_token_ids = pad_sequence(batched_token_ids, batch_first=True, padding_value=padding_value)
        batched_wav = pad_sequence(batched_wav, batch_first=True, padding_value=padding_value)
        conditionings = pad_sequence(conditionings, batch_first=True, padding_value=padding_value)

        batched_wav = batched_wav.unsqueeze(1).to(dtype=self.wav_dtype)
        conditionings = conditionings.to(dtype=self.wav_dtype)
        
        text_lengths = torch.tensor([len(t) for t in batched_token_ids])
        wav_lengths = torch.tensor([t.shape[-1] for t in batched_wav])
        
        out = {
            'text': batched_text,
            'token_ids': batched_token_ids,
            'text_lengths': text_lengths,
            'wav_lengths': wav_lengths,
            'wav': batched_wav,
            'sampling_rate': sampling_rate,
            'duration': duration,
            'conditioning': conditionings,
            'conditioning_length': torch.tensor(conditioning_lengths),
            'conditioning_idxs': torch.tensor(conditioning_idxs)
        }
        
        if return_attention_mask:
            mask = batched_token_ids != padding_value
            out['attention_mask'] = mask.to(dtype=torch.int32)
        
        out = _put(out, device=device)
        return out
    
def format_batch_on_device(batch, dvae, encoder_mel, dvae_mel, input_sample_rate=16000, dvae_sampling_rate=16000):
    
    batch['cond_mels'] = encoder_mel(batch['conditioning']) # from [B, 1, T] 
    batch['cond_mels']  = batch['cond_mels'].unsqueeze(1) # to [B, 1, C, T]
    del batch['conditioning']
    if input_sample_rate != dvae_sampling_rate:
        dvae_wav = resample(
            batch['wav'], input_sample_rate, dvae_sampling_rate, 
            lowpass_filter_width=32, rolloff=95, beta=14.7
        )
    else:
        dvae_wav = batch['wav']
    del batch['wav']
    dvae_wav = dvae_mel(dvae_wav)
    codes = dvae.get_codebook_indices(dvae_wav)
    batch['audio_codes'] = codes
    
    return {
        'text_inputs': batch['token_ids'],
        'text_lengths': batch['text_lengths'],
        'audio_codes': batch['audio_codes'],
        'wav_lengths': batch['wav_lengths'],
        'cond_mels': batch['cond_mels'],
        'cond_idxs': batch['conditioning_idxs'],
        'cond_lens': batch['conditioning_length'],
    }

class GroupedSampler(Sampler):
    '''Grouped Sampler for minimizing padding in DataLoader, supporting DDP training'''
    def __init__(
        self, 
        dataset, 
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 42,
        model_input_name: str = None,
        model_length_name: str = 'duration', # or 'text_lengths' or 'wav_lengths'
        num_multi_batch: int = None,
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        self.model_input_name = model_input_name
        self.model_length_name = model_length_name
        
        self.num_samples = len(self.dataset) # actual size of dataset
        # modify total size to be divisible by num_replicas, and not more than num_samples
        self.total_size = self.num_samples + (self.num_replicas - self.num_samples % self.num_replicas)
        if self.total_size > self.num_samples:
            self.total_size -= self.num_replicas
            
        assert self.total_size % self.num_replicas == 0
        
        if num_multi_batch is None:
            num_multi_batch = max(self.total_size // 100, 1)
        self.num_multi_batch = min(num_multi_batch, self.total_size)
        
    def __iter__(self):
        
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        indices = torch.randperm(self.total_size, generator=g).tolist()
        grouped_indices = [
            indices[i: i + self.num_multi_batch]
            for i in range(0, self.total_size, self.num_multi_batch)
        ]
        for batch_indices in grouped_indices:
            
            if self.shuffle:
                random.shuffle(batch_indices)
                
            if self.model_length_name is not None:
                length = lambda i: self.dataset[i][self.model_length_name]
            else:
                length = lambda i: self.dataset[i][self.model_input_name].shape[-1]
            batch_indices = sorted(
                batch_indices, 
                key=length, 
                reverse=True) 
        
        flat_indices = [i for m in grouped_indices for i in m]
        assert len(flat_indices) == self.total_size
        
        subsamples = flat_indices[self.rank: self.total_size : self.num_replicas] 
        
        return iter(subsamples)

    def __len__(self) -> int:
        return self.total_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

def create_gigaspeech_dataloader(
    dataset_dir: str, # dir containing parquet files
    tokenizer_path: str='./assets/tokenizing/tokenizer.json',
    subset: str='m',
    split: str='train',
    token: str=None,
    max_samples: int=-1,
    batch_size: int=64,
    num_workers: int=4,  
    pin_memory: bool=True,
    shuffle: bool=True,
    device: str='cpu',
    is_eval: bool=False,
    target_sampling_rate: int=16000,
    min_wav_length: int=80_000,
    max_wav_length: int=220_000,
    min_condition_length: int=80_000,
    max_condition_length: int=220_000,
    min_text_length: int=5,
    max_text_length: int=100,
    num_multi_batch: int=200,
    wav_dtype=torch.float32,
    rank=0,
    num_replicas=1,
    seed=42
):
    '''Create GigaSpeech DataLoader'''
    has_data = (
        os.path.exists(dataset_dir) and 
        len([f for f in os.listdir(dataset_dir) if f.endswith('.parquet')]) > 0
    )
    if not has_data:
        assert token is not None
        dataset_dir = GigaSpeech._load_data(
            token, subset=subset, split=split, target_dir=dataset_dir
        )
    logger.info(f"-------> Dataset directory: {dataset_dir}")
    
    dataset = GigaSpeech(
        dataset_dir, tokenizer_path, max_samples, target_sampling_rate,
        min_wav_length, max_wav_length, min_condition_length, max_condition_length,
        min_text_length, max_text_length, wav_dtype, is_eval
    )
    sampler = GroupedSampler(
        dataset, num_replicas, rank, shuffle, seed, num_multi_batch=num_multi_batch
    )
    collate_func = partial(dataset.collate_func, padding_value=0, device=device)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
        sampler=sampler, collate_fn=collate_func
    )

    return {split: dataloader}