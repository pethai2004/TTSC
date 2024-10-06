# GigaSpeech Dataset 
import regex as re
import math
import unicodedata
import logging
import torchaudio
import torch
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
from datasets import load_dataset, DownloadConfig
from torch.utils.data import Dataset, IterableDataset, get_worker_info, Sampler, DataLoader
from torch.distributed import get_rank, get_world_size, is_initialized
from src.data_utility import _recurse_func, _put

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

repo_name = 'speechcolab/gigaspeech'
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
            text = text.lower()
        return text

    return _recurse_func(_preprocess, input, test_type=str)

def preprocess_audio(input, mel_transform=None, sampling_rate=16000, remove_wav_column=True):
    if 'sampling_rate' in input:
        if sampling_rate != input['sampling_rate']:
            input['wav'] = torchaudio.transforms.Resample(input['sampling_rate'], sampling_rate)(input['wav'])
        else:
            input['sampling_rate'] = sampling_rate
    
    if mel_transform is not None:
        input['mel'] = mel_transform(input['wav'])
    
    if remove_wav_column:
        input.pop('wav')
    
    return input

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

def filter_func(x, max_wav_sec=15, min_wav_sec=5):
    return not (x['duration'] > max_wav_sec or x['duration'] < min_wav_sec 
                or x['text'] == "<MUSIC>" or x['text'] == '' or len(x['text']) < 5)

def collate_fn(x, padding_value=0, dtype=torch.float16, device='cuda', non_blocking=True):
    x['wav'] = pad_sequence([wav.clone().detach() for wav in x['wav']],
                            batch_first=True, padding_value=padding_value)
    x['wav'] = x['wav'].unsqueeze(1)
    x = _put(x, device=device, dtype=dtype, non_blocking=non_blocking)
    return x

class BaseGigaSpeech:
    '''Base class for handling GigaSpeech dataset loading with common preprocessing steps'''
    
    def __init__(self, tokenizer_path: str=None, subset='l', split='train', cache_dir='./assets/.cache/gigaspeech', 
                 token_pad_id=0, pad_token='[PAD]', num_proc=32, loader_batch_size=1024, 
                 min_wav_sec=5, max_wav_sec=15, streaming=False):
        
        self.cache_dir = cache_dir
        self.tokenizer = None
        if tokenizer_path is not None:
            self.tokenizer: Tokenizer = Tokenizer.from_file(tokenizer_path)
            self.tokenizer.enable_padding(pad_id=token_pad_id, pad_token=pad_token, pad_to_multiple_of=8)
        
        load_config = DownloadConfig(cache_dir=self.cache_dir, resume_download=True, num_proc=num_proc)
        
        self.data_ = load_dataset(
            repo_name, subset, split=split, streaming=streaming, 
            download_config=load_config, trust_remote_code=True
        )
        
        self.data_ = self.data_.map(
            lambda _x: preprocess(_x, tokenizer=self.tokenizer), 
            remove_columns=['audio', 'segment_id', 'speaker', 
                            'original_full_path', 'url', 'title', 'audio_id', 
                            'begin_time', 'end_time'],
            batched=True, batch_size=loader_batch_size
        )
        
        self.data_ = self.data_.filter(
            lambda x: filter_func(x, max_wav_sec=max_wav_sec, min_wav_sec=min_wav_sec)
        )
        
        self.data_ = self.data_.with_format('torch')

class GigaSpeech(BaseGigaSpeech, Dataset):
    '''
    GigaSpeech Dataset class for non-streaming mode
    '''
    
    def __init__(self, data_save_path: str="./assets/gigaspeech", **kwargs):
        super().__init__(streaming=False, **kwargs)
        self.data_ = list(self.data_)  # Load the streamed data into memory
        if data_save_path is not None:
            self.data_.save_to_disk(data_save_path)
            
    def __len__(self):
        return len(self.data_)

    def __getitem__(self, index):
        return self.data_[index]

class IterableGigaSpeech(BaseGigaSpeech, IterableDataset):
    '''
    GigaSpeech Dataset in streaming mode without load all data in disk. This support Distributed training scheme
    '''
    
    def __init__(self, batch_size=1, max_batch_sample=-1, infinite_loop=True, num_sample_to_skip=0, **kwargs):
        super().__init__(streaming=True, **kwargs)
        self._auto_handle_distributed = True # whether to handle DDP training scheme automatically, otherwise should be handle in downstream code to yield correct batch
        self.batch_size = batch_size
        self.max_batch_sample = max_batch_sample
        self.infinite_loop = infinite_loop
        self.num_sample_to_skip = num_sample_to_skip

        self.num_epochs = 0
        self.batch_sample_so_far = 0
        self.global_batch_sample_so_far = 0

        if num_sample_to_skip > 0:
            self.data_ = self.data_.skip(num_sample_to_skip)

    def _get_iterator(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        self.num_epochs += 1
        return self.data_.iter(batch_size=batch_size, drop_last_batch=True)

    def __iter__(self):
        
        while True:
            data = self._get_iterator()
            worker_info = get_worker_info()
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
                    logger.info(f"------->>>> Early Data exhausted after {self.global_batch_sample_so_far} batch")
                    break
            if not self.infinite_loop:
                logger.info(f"------->>>> Data exhausted after {self.num_epochs} epochs with {self.batch_sample_so_far} batch")
                break
            logger.info(f"------->>>> Reset at {self.num_epochs} completed with {self.batch_sample_so_far} batch")

class GroupSampler(Sampler):
    
    '''Grouped sampler where specify key of tensor with minimal length difference will be grouped together, minimizing padding. (Support DDP)'''
    def __init__(
        self, 
        dataset, 
        num_replicas=1,
        rank=0,
        num_multi_batch=None,
        shuffle: bool = True,
        seed: int = 42,
        feature: str = 'wav',
        length_feature: str = 'duration'
    ):

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        self.feature = feature
        self.length_feature = length_feature
        self.dataset_length = len(dataset)
        
        if num_multi_batch is None:
            num_multi_batch = max(self.dataset_length // 1000, 1)
        self.num_multi_batch = num_multi_batch

        self.num_samples = math.ceil(
            (len(self.dataset) - self.num_replicas) / self.num_replicas
        )

        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        indices = torch.randperm(self.dataset_length, generator=g).tolist()
        grouped_indices = [
            indices[i: i + self.num_multi_batch]
            for i in range(0, self.dataset_length, self.num_multi_batch)
        ]

        grouped_indices = []
        for batch in grouped_indices:
            if self.length_feature is None:
                key = lambda i: len(self.dataset[i][self.feature])
            else:
                key = lambda i: self.dataset[i][self.length_feature]
                
            sorted_batch = sorted( batch,key=lambda i: key(i), reverse=True)
            grouped_indices.append(sorted_batch)
            
        flat_indices = [i for m in grouped_indices for i in m][:self.total_size]

        assert len(flat_indices) == self.total_size

        # Subsample for the current process
        subsampled_indices = flat_indices[self.rank:self.total_size:self.num_replicas]
        assert len(subsampled_indices) == self.num_samples

        return iter(subsampled_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        
def create_gigaspeech_dataloader(
    *args, data_save_path="./assets/gigaspeech", rank=0, world_size=1, num_replicas=1
):
    
    data = GigaSpeech(data_save_path=data_save_path, *args)
    sampler = GroupSampler(data, num_replicas=num_replicas, rank=rank)
    dataloader = DataLoader(data, batch_size=None, sampler=sampler, collate_fn=collate_fn)

    return dataloader