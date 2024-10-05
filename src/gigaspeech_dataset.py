# GigaSpeech Dataset 

import regex as re 
import unicodedata  
import logging
import torchaudio
import torch
from torch.nn.utils.rnn import pad_sequence

from tokenizers import Tokenizer
from datasets import load_dataset, DownloadConfig
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torch.distributed import get_rank, get_world_size, is_initialized

from src.data_utility import _recurse_func, _put

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

repo_name = 'speechcolab/gigaspeech'
available_subset = eval(open('./assets/gigaspeech_subset.txt', 'r').read())

### text ###
replace_abbrev = {
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Ms.": "Miss",
    "Dr.": "Doctor",
    "Prof.": "Professor",
    "No.": "Number",
    "U.S.": "United States",
    "U.K.": "United Kingdom", # etc
} 
punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
gigaspeech_replacement = {'<COMMA>': ',', '<PERIOD>': '.', '<QUESTIONMARK>': '?', '<EXCLAMATIONPOINT>': '!', '<SIL>': '', '<MUSIC>': '', '<NOISE>': '', '<OTHER>': ''}
pat = re.compile('|'.join(re.escape(key) for key in gigaspeech_replacement.keys()))

def preprocess_text(input, lower=True, remove_non_english=True, should_abbreviate=True):
    
    def _preprocess(text: str):
        '''Preprocess text for Tokenizer'''
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # remove extra spaces
        
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')    
        # remove emojis separately
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        # detect any non English characters (only accept A-Z, a-z, 0-9, and punctuation)    
        replace_eng = '[UNK]' if remove_non_english else ''
        text = re.sub(r'[^A-Za-z0-9' + punctuation + ']+', replace_eng, text)
            
        text = re.sub(r'\s+', ' ', text) # remove extra spaces

        if should_abbreviate:
            pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in replace_abbrev.keys()) + r')\b')
            def replace_match(match):
                return replace_abbrev[match.group(0)]
            pattern.sub(replace_match, text)
        
        # for GigaSpeech, we also need to replace some special tokens
        text = pat.sub(lambda x: gigaspeech_replacement[x.group()], text)
        if lower:
            text = text.lower() # lower last to avoid lowercasing special tokens
        return text 
    
    return _recurse_func(_preprocess, input, test_type=str)

### audio ###
def preprocess_audio(input, mel_transform=None, sampling_rate=16000, remove_wav_column=True):
    
    if 'sampling_rate' in input:
        if sampling_rate != input['sampling_rate']:
            input['wav'] = torchaudio.transforms.Resample(input['sampling_rate'], sampling_rate)(input)
    else:
        input['sampling_rate'] = sampling_rate
        
    if mel_transform is not None:
        input['mel'] = mel_transform(input['wav'])
        if remove_wav_column:
            input.pop('wav')
    
    return input

def preprocess(x, tokenizer: Tokenizer=None):
    
    audio_value = x['audio'] 
    # array = [value['array'].astype(wav_dtype) for value in audio_value]
    array = [value['array'] for value in audio_value]
    sampling_rate = [value['sampling_rate'] for value in audio_value]
    duration = [len(A) / sr for A, sr in zip(array, sampling_rate)]
    text = list(map(preprocess_text, x['text']))
    output = {'wav': array, 'text': text, 'sampling_rate': sampling_rate, 'duration': duration}
    
    if tokenizer is not None:
        text = tokenizer.encode_batch_fast(text)
        output['token_ids'] = [t.ids for t in text]
        output['attention_mask'] = [t.attention_mask for t in text]
        
    return output   

def filter_func(x, max_wav_sec=15, min_wav_sec=5):
    if x['duration'] > max_wav_sec or x['duration'] < min_wav_sec:
        return False
    if x['text'] == "<MUSIC>":
        return False
    if x['text'] == '' or len(x['text']) < 5: # some hard threshold
        return False
    return True 

def collate_fn(x, padding_value=0, dtype=torch.float16, device='cuda', non_blocking=True):
    x['wav'] = pad_sequence([wav.clone().detach() for wav in x['wav']], 
                            batch_first=True, padding_value=padding_value) 
    x['wav'] = x['wav'].unsqueeze(1)
    # some key tensor will not be used, consider remove them / manual move only necessary tensors
    x = _put(x, device=device, dtype=dtype, non_blocking=non_blocking)
    return x

class IterableGigaSpeech(IterableDataset):
    '''GigaSpeech Dataset in streaming mode without load all data in disk. This support Distributed training scheme\n
        gigaspeech = IterableGigaSpeech(32) \n
        giga_speech_iter = iter(gigaspeech) \n
        sample = collate_fn(next(giga_speech_iter)) 
    '''
    def __init__(
        self, batch_size=1, tokenizer_path: str=None, subset='l', split='train', 
        cache_dir='./assets/.cache/gigaspeech', token_pad_id=0, pad_token='[PAD]', num_sample_to_skip=0, max_batch_sample=-1,
        num_proc=32, loader_batch_size=1024, min_wav_sec=5, max_wav_sec=15, infinite_loop=True
    ):
        
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.infinite_loop = infinite_loop
        self.max_batch_sample = max_batch_sample
        self._auto_handle_distributed = True # flag for handling yield data in distributed setting 
        
        if tokenizer_path is not None:
            self.tokenizer: Tokenizer = Tokenizer.from_file(tokenizer_path)
            self.tokenizer.enable_padding(pad_id=token_pad_id, pad_token=pad_token, pad_to_multiple_of=8)
            
        load_config = DownloadConfig(cache_dir=self.cache_dir, resume_download=True, num_proc=num_proc)
        self.data_ = load_dataset(repo_name, subset, split=split, streaming=True, 
                                  download_config=load_config, trust_remote_code=True)
        if num_sample_to_skip > 0:
            self.data_ = self.data_.skip(num_sample_to_skip)
        self.data_ = self.data_.map(
            lambda _x : preprocess(_x), remove_columns=['audio', 'segment_id', 'speaker', 'original_full_path', 'url', 'title', 'audio_id', 'begin_time', 'end_time'],
            batched=True, batch_size=loader_batch_size
        ) # map before filter since filter need 'duration' key which is created in preprocess
        self.data_ = self.data_.filter(
            lambda x: filter_func(x, max_wav_sec=max_wav_sec, min_wav_sec=min_wav_sec)
        )
        self.data_ = self.data_.with_format('torch')
        self.num_epochs = 0
        self.batch_sample_so_far = 0
        self.global_batch_sample_so_far = 0
        
    def _get_iterator(self, batch_size=None):
        '''get dataset.iter (reseting the iterator)'''
        batch_size = self.batch_size if batch_size is None else batch_size
        self.num_epochs += 1
        return self.data_.iter(batch_size=batch_size, drop_last_batch=True)
    
    def __iter__(self):
        
        while True:
            data = self._get_iterator()
            worker_info = get_worker_info()
            # if use DDP but not set _auto_handle_distributed, then assume that the data will be wrapped in downstream code (e.g. accelerator.prepare)
            rank = get_rank() if is_initialized() and self._auto_handle_distributed else 0
            world_size = get_world_size() if is_initialized() and self._auto_handle_distributed else 1
            worker_id = worker_info.id if worker_info is not None else 0
            num_workers = worker_info.num_workers if worker_info is not None else 1

            shard_id = rank * num_workers + worker_id
            total_shards = world_size * num_workers

            
            for i, batch in enumerate(data):
                if i % total_shards == shard_id:
                    self.batch_sample_so_far += 1
                    self.global_batch_sample_so_far += 1
                    yield batch
                
                if self.max_batch_sample > 0 and self.global_batch_sample_so_far >= self.max_batch_sample:
                    logger.info(f"------->>>> Early Data exhausted after {self.global_batch_sample_so_far} samples")
                    break # Stop for current data stream if max_batch_sample is reached
                     
            if not self.infinite_loop:
                logger.info(f"------->>>> Data exhausted after {self.num_epochs} epochs with {self.batch_sample_so_far} samples")
                break  # Stop after one pass if not looping infinitely
            logger.info(f"------->>>> Reset at {self.num_epochs} completed with {self.batch_sample_so_far} samples")
            
class GigaSpeech(Dataset):
    
    def __inti__(self):
        raise NotImplementedError("Use IterableGigaSpeech instead")