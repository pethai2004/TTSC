# preprocessing step (text, audio)
import regex as re 
import unicodedata  
import torchaudio

from src.data_utility import _recurse_func

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