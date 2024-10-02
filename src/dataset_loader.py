
import re
from queue import Queue
import logging
import json
import unicodedata
import os
import math
import urllib.request
import tarfile
import shutil
from queue import Queue

import torch 
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import datasets
from datasets import load_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# hf_XEjvuYDlxCKIvmRnruVhizRQziiCpAhacgp

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

def get_path_recursive(path):

    if isinstance(path, dict):
        return get_path_recursive(next(iter(path.values())))
    elif isinstance(path, list):
        return [get_path_recursive(p) for p in path]
    return path

def remove_recursive(path, except_paths=None):

    except_paths = except_paths or []
    if os.path.isdir(path):
        for p in os.listdir(path):
            remove_recursive(os.path.join(path, p), except_paths)
        if path not in except_paths:
            os.rmdir(path)
    elif path not in except_paths:
        os.remove(path)

def flatten_audio(example):

    example['wav'] = example['audio']['array']
    example['sampling_rate'] = example['audio']['sampling_rate']
    example['path'] = example['audio']['path']
    del example['audio']
    if 'duration_ms' not in example:
        example['duration_ms'] = len(example['wav']) / example['sampling_rate'] * 1000
        
    return example

class BaseDataset:
    def __init__(self, repo, split, cache_dir, num_proc=16):
        assert split in ['train', 'validation', 'test']
        self.repo = repo
        self.split = split
        self.cache_dir = cache_dir
        self.num_proc = num_proc
        self.load_queue = Queue()
        self.dataset = None
        self.current_parquet_path = None

        if os.path.exists(self.cache_dir):
            remove_recursive(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_parquet(self, url):

        self.current_parquet_path = os.path.join(self.cache_dir, os.path.basename(url))
        urllib.request.urlretrieve(url, self.current_parquet_path)

    def _create_dataset(self, remove_columns=[]):

        if self.dataset is not None:
            # Remove existing dataset files
            cache_files = get_path_recursive(self.dataset.cache_files)
            remove_recursive(self.current_parquet_path)
            for f in cache_files:
                os.remove(f)

        self._load_parquet(self.load_queue.get())
        self.dataset = load_dataset('parquet', data_files=self.current_parquet_path, cache_dir=self.cache_dir, keep_in_memory=True)['train']
        self.dataset = self.dataset.remove_columns(remove_columns)
        self.dataset = self.dataset.map(flatten_audio, num_proc=self.num_proc, batch_size=1024)
        use_col = ['text', 'wav', 'sampling_rate', 'path', 'duration_ms']
        self.dataset = self.dataset.remove_columns([col for col in self.dataset.column_names if col not in use_col])
        
    @staticmethod
    def audio_collate_fn(batch):
        '''Collate function for DVAE dataset'''
        #batch["wav_length"] = torch.stack(batch["wav_length"])
        # note batch['wav'] has shape [1, seq_length]
        wav_padded = pad_sequence([wav[0] for wav in batch["wav"]], batch_first=True)
        wav_padded = wav_padded.unsqueeze(1)    # Add a channel dimension
        batch["wav"] = wav_padded
        # now has shape
        return batch
    
class PeopleSpeechDataset(BaseDataset):
    endpoint = "https://huggingface.co/api/datasets/MLCommons/peoples_speech/parquet/default/train"
    
    def __init__(self, split='train', cache_dir='./assets/.peoples_speech_cache'):
        super().__init__('MLCommons/peoples_speech', split, cache_dir)
        self.endpoint_url = sorted(eval(urllib.request.urlopen(PeopleSpeechDataset.endpoint).read().decode('utf-8')))
        for url in self.endpoint_url:
            self.load_queue.put(url)

    @property
    def current_data_length(self):
        return len(self.dataset)

class GigaSpeechDataset(BaseDataset):
    
    def __init__(self, hf_token, subset='dev', split='validation', cache_dir='./assets/.gigaspeech_cache', num_proc=16):
        super().__init__('speechcolab/gigaspeech', split, cache_dir, num_proc)
        self.hf_token = hf_token
        self.endpoint = f"https://huggingface.co/api/datasets/speechcolab/gigaspeech/parquet/{subset}/{split}"
        req = urllib.request.Request(self.endpoint)
        req.add_header("Authorization", f"Bearer {hf_token}")
        self.endpoint_url = sorted(eval(urllib.request.urlopen(req).read().decode('utf-8')))
        for url in self.endpoint_url:
            self.load_queue.put(url)

    def _load_parquet(self, url):

        request = urllib.request.Request(url)
        request.add_header('Authorization', f'Bearer {self.hf_token}')
        self.current_parquet_path = os.path.join(self.cache_dir, os.path.basename(url))
        with urllib.request.urlopen(request) as response, open(self.current_parquet_path, 'wb') as out_file:
            out_file.write(response.read())

class LJSpeechDataset(BaseDataset): # we will load entire dataset at once (small dataset) 
    
    @staticmethod
    def _load_dataset(self, target_dir='./assets/'):
        load_ljspeech(target_dir)  
        
    def __init__(self, dataset_path: str, sampling_rate=16000, split='train', cache_dir='./assets/.ljspeech_cache'):
        self.dataset_path = dataset_path
        self.meta_data_path = os.path.join(self.dataset_path, 'metadata.json')
        assert os.path.exists(self.dataset_path) and os.path.exists(self.meta_data_path)
        
        torch.set_num_threads(1)
        samples = json.loads(open(os.path.join(self.dataset_path, 'metadata.json')).read())
        self.samples = []
        
        for s in samples:
            try:
                load_validate_audio(s['audio_file'], sampling_rate)
            except Exception as e:
                logging.warning(f"Error loading {s['audio_file']}: {e}")
                continue
            self.samples.append(s)
        logging.info(f"Loaded and validated {len(self.samples)} samples out of {len(samples)}")
            
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

############################################################################################################

def load_ljspeech(target_dir='./assets/', create_meta_json="formatted", add_root_path_key=True, preprocess_metadata=True):
    
    '''load LJSpeech dataset, then create metadata in JSON format if create_meta_json is "original", the JSON key will be same 
        else, the key will be "audio_file" (path to audio ) and "text" (corresponding text), etc'''
        
    assert create_meta_json in ["original", "formatted"] 
    target_dir = os.path.abspath(target_dir)
    assert os.path.exists(target_dir), f"------->>>> Target directory {target_dir} does not exist"
    path_to_data = os.path.join(target_dir, 'LJSpeech-1.1')
    exist = os.path.exists(path_to_data) and os.path.exists(os.path.join(path_to_data, 'metadata.json'))
    if exist: # assume the dataset is not loaded
        url =  'http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'
        file_path ='LJSpeech-1.1.tar.bz2'
        logger.info(f"Loading LJSpeech dataset ...")
        
        urllib.request.urlretrieve(url, file_path)
        with tarfile.open(file_path, 'r:bz2') as t:
            t.extractall()
            try:
                shutil.move('LJSpeech-1.1', target_dir)
            except shutil.Error:
                logger.info(f"------->>>> Dataset already exists at {path_to_data}")
                
            os.remove(file_path)
            logger.info(f"------->>>> Finished loading dataset path: {path_to_data}")
    else:
        logger.info(f"------->>>> Dataset already exists at {path_to_data}, skipping download")    
    
    metadata = load_metadata_LJSpeech(path_to_data)
    if create_meta_json == "formatted": # modify metadata to be in the format of audio_file and text (is list of dict)
        metadata_formatted = []
        for k, v in metadata.items():
            wav_path = os.path.join(path_to_data, 'wavs', k + '.wav')
            if not os.path.exists(wav_path):
                logger.info(f"Missing file: {wav_path}")
                continue
            metadata_formatted.append({
                'audio_file': wav_path, 
                'text': v, 
                'speaker_name': 'ljSpeech',
                'language': 'en',
                })           
            if add_root_path_key:
                metadata_formatted[-1]['root_path'] = path_to_data
            
        metadata = metadata_formatted
        print("------->>>> DONE formatting")
        
    with open(os.path.join(path_to_data, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
        
    if preprocess_metadata:
        preprocess_json_txt(os.path.join(path_to_data, 'metadata.json'))
        
    with open(os.path.join(path_to_data, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
        
    logger.info(f"Created metadata at {os.path.join(path_to_data, 'metadata.json')} with {len(metadata)} samples")
    
def load_metadata_LJSpeech(dataset_dir: str):
    '''Load metadata.csv from LJSpeech dataset'''
    meta_path = os.path.join(dataset_dir, 'metadata.csv') # check if metadata exists in ./LJSpeech-1.1
    assert os.path.exists(meta_path), f"------->>>> Expected metadata.csv in {meta_path}"
    
    lines = open(meta_path, 'rb').read().decode('utf-8')
    meta = re.compile(r'\n').split(lines)
    metadict = {}
    for line in meta:
        split = line.split('|')
        if split.__len__() < 3:
            continue
        name, text, _ = split
        metadict[name] = text
    return metadict

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

def txt_preprocessor(text: str, remove_non_english=True, should_abbreviate=True):
    '''Preprocess text for Tokenizer'''
    text = text.strip().lower()
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
    
    return text 

def preprocess_json_txt(path_to_json: str):
    '''Preprocess text in JSON file (Recurse)'''
    data = json.loads(open(path_to_json).read())
    def _preprocess_text(text):
        if isinstance(text, dict):
            return {k: _preprocess_text(v) for k, v in text.items()}
        elif isinstance(text, list):
            return [_preprocess_text(v) for v in text]
        elif isinstance(text, str):
            return txt_preprocessor(text)
        else:
            raise ValueError(f"Unsupported type: {type(text)}")
    data = _preprocess_text(data)
    with open(path_to_json, 'w') as f:
        json.dump(data, f)
    
    logger.info(f"Preprocessed text in {path_to_json}")
    
def create_split_metadata(dataset_dir: str="./assets/LJSpeech-1.1", split_ratio=0.9, limit_examples=-1):
    '''Split metadata into training and validation set'''
    metadata = json.loads(open(os.path.join(dataset_dir, 'metadata.json')).read())
    metadata = metadata[:limit_examples]     
    assert isinstance(metadata, list), "------->>>> metadata should be a list of dict, use  `formatted` option in load_ljspeech"
    num_training_split = math.ceil(len(metadata) * split_ratio)
    logger.info(f"------->>>> Splitting dataset into {num_training_split} training samples and {len(metadata) - num_training_split} validation samples. All samples: {len(metadata)}")

    train_meta, eval_meta = metadata[:num_training_split], metadata[num_training_split:]
    
    return train_meta, eval_meta

if __name__ == "__main__":
    
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, default='./assets/')

    args = parser.parse_args()
    load_ljspeech(args.target_dir)
    