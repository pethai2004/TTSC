import os
import time
import urllib.request
import threading
import logging
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import ujson  

from datasets import load_dataset 


class Timer:
    '''context manager to measure time'''
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        print(f"Time taken: {self.end - self.start}")

class ppsDataset:
    ''' People Speech dataset manager'''
    end_point = "https://huggingface.co/api/datasets/MLCommons/peoples_speech/parquet/default/train"

    def __init__(self, batch_size: int = 128, infinite_load: bool = False, as_json=False, max_rows=3000, only_text_and_audio=True):
        self.max_rows = max_rows
        try:
            self.parquet_endpoints = eval(urllib.request.urlopen(self.end_point).read().decode('utf-8'))
        except Exception as e:
            raise Exception(f"Failed to retrieve dataset endpoints: {e}")

        self.available_parquet_files = []
        self._temp_dir = os.path.join(os.getcwd(), 'temp')
        if os.path.exists(self._temp_dir):
            parquets = [os.path.join(self._temp_dir, i) for i in os.listdir(self._temp_dir) if i.endswith('.parquet')]
            self.available_parquet_files.extend(parquets)
            endpoint_ = [url for url in self.parquet_endpoints if os.path.basename(url) not in [os.path.basename(p) for p in parquets]]
        else:
            endpoint_ = self.parquet_endpoints

        self.parquet_left = Queue(maxsize=len(endpoint_))
        for url in endpoint_:
            self.parquet_left.put(url)

        self.current_parquet = None
        self.current_parquet_path = None
        self.current_parquet_length = None
        self.num_parquet_row_read = 0

        os.makedirs(self._temp_dir, exist_ok=True)
        self._max_loaded = 10
        self.batch_size = batch_size
        self.infinite_load = infinite_load
        self.as_json = as_json
        self.only_text_and_audio = only_text_and_audio
        # Store preprocessed data
        self.available_result = {
            "id": [],
            "audio": [],
            "path": [],
            "text": []
        }
        self.lock = threading.Lock()
        
        self._start_threads()
    
    @property
    def num_dataset_left(self):
        '''Number of batches left in the dataset before exhausted'''
        return self.parquet_left.qsize()

    def _preload_parquet(self):
        ''' Load the next parquet file '''
        while True:
            with self.lock:
                if len(self.available_parquet_files) < self._max_loaded and not self.parquet_left.empty():
                    parquet_url = self.parquet_left.get()
                    parquet_path = os.path.join(self._temp_dir, os.path.basename(parquet_url))
                    try:
                        with Timer():
                            urllib.request.urlretrieve(parquet_url, parquet_path)
                        self.available_parquet_files.append(parquet_path)
                    except Exception as e:
                        logging.error(f"Error downloading file {parquet_url}: {e}")
            time.sleep(1)

    def _preprocess_data(self):
        '''Preprocess loaded parquet files and store the result'''
        while True:
            if len(self.available_result["id"]) >= self.max_rows:
                time.sleep(5)
                continue

            if self.available_parquet_files:
                with self.lock:
                    if self.available_parquet_files:
                        current_parq = self.available_parquet_files.pop(0)
                    else:
                        current_parq = None

                if current_parq:
                    try:
                        parquet = pq.read_table(current_parq).to_pandas()
                        processed_data = self.preprocess_parquet(parquet, only_text_and_audio=self.only_text_and_audio)
                        with self.lock:
                            for key in self.available_result.keys():
                                self.available_result[key].extend(processed_data[key])
                    except Exception as e:
                        logging.error(f"Error processing file {current_parq}: {e}")
                        if os.path.exists(current_parq):
                            os.remove(current_parq)
                else:
                    time.sleep(1)

    def _start_threads(self):
        preload_thread = threading.Thread(target=self._preload_parquet)
        preload_thread.daemon = True
        preload_thread.start()
        
        preprocess_thread = threading.Thread(target=self._preprocess_data)
        preprocess_thread.daemon = True
        preprocess_thread.start()

    def _get_next(self):
        '''Return next batch as soon as it is available'''
        batch = {key: [] for key in self.available_result.keys()}
        with self.lock:
            for i in range(self.batch_size):
                if not self.available_result["id"]:
                    break
                for key in batch.keys():
                    batch[key].append(self.available_result[key].pop(0))
        if batch['id'].__len__() == 0:
            # try again
            time.sleep(1)
            return self._get_next()
        return batch

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._get_next())

    @staticmethod
    def preprocess_parquet(parq: pd.DataFrame, pad_value=0, max_audio_length=None, only_text_and_audio=True):
        '''Convert to pandas to dictionary'''
        if not only_text_and_audio:
            id_row = parq["id"].values.tolist()
            text = parq["text"].values.tolist()

        bytes_list = parq["audio"].to_list()
        
        with ThreadPoolExecutor() as executor:
            audio = list(executor.map(lambda p: list(bytearray(p["bytes"])), bytes_list))

        max_audio_length = max(len(a) for a in audio) if max_audio_length is None else max_audio_length
        audio_padded = [a + [pad_value] * (max_audio_length - len(a)) for a in audio]
        
        if not only_text_and_audio:
            path = [p["path"] for p in bytes_list]

        if not only_text_and_audio:
            return {
                "id": id_row,
                "audio": audio_padded,
                "path": path,
                "text": text
            }
        else:
            return {
                "audio": audio_padded,
                "text": text
            }
    
def load_people_speech(target_dir='./assets/PeopleSpeech', cache_dir="./assets/cache"):
    '''Load hugginface People Speech dataset'''
    
    repo = 'MLCommons/peoples_speech'
    stream = load_dataset(repo, split='train', cache_dir=cache_dir, streaming=True)
    return stream

def test_timing():
    
    '''test the timing of loading People Speech dataset'''
    batch_size = 128
    num_iterate = 30 
    
    dataset = ppsDataset(batch_size=128, max_rows=3000)        
    streamer = load_people_speech()
    streamer_iter = streamer.iter(batch_size=batch_size)
    
    _ = dataset._get_next() # warm up
    _ = next(streamer_iter) # warm up
    
    with Timer() as t:
        
        for _ in range(num_iterate):
            _ = next(streamer_iter)

    print(f"Time taken to load {num_iterate * batch_size} samples from streamer: {t.end - t.start} seconds")
    
    with Timer() as t:
        
        for _ in range(num_iterate):
            _ = dataset._get_next()
            
    print(f"Time taken to load {num_iterate * batch_size} samples custom data class: {t.end - t.start} seconds")
    
    