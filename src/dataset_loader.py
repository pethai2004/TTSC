
import re
import logging
import json
import unicodedata
import os
import math
import urllib.request
import tarfile
import shutil

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

############################################################################################################

def load_ljspeech(target_dir='./assets/', create_meta_json="formatted", add_root_path_key=True, preprocess_metadata=True):
    
    '''load LJSpeech dataset, then create metadata in JSON format if create_meta_json is "original", the JSON key will be same 
        else, the key will be "audio_file" (path to audio ) and "text" (corresponding text), etc'''
        
    assert create_meta_json in ["original", "formatted"] 
    target_dir = os.path.abspath(target_dir)
    assert os.path.exists(target_dir), f"------->>>> Target directory {target_dir} does not exist"
    path_to_data = os.path.join(target_dir, 'LJSpeech-1.1')
    exist = os.path.exists(path_to_data) and os.path.exists(os.path.join(path_to_data, 'metadata.json'))
    if not exist: # assume the dataset is not loaded
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
    