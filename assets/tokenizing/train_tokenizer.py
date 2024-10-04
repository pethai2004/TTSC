
import os 

from tokenizers import Tokenizer, models, trainers, decoders, pre_tokenizers
from tokenizers.normalizers import NFKC, Lowercase
from tokenizers.normalizers import Sequence as norm_sequence

txt_path = ["text_concatenated.txt"] # train except list of path
if not os.path.isfile(txt_path[0]):
    raise FileNotFoundError(f"File not found: {txt_path[0]}")

save_path = "tokenizer.json"
vocab_size = 3000
min_freq = 10
save_tokenizer_path = "tokenizer.json"
special_token = ["[STOP]", "[UNK]", "[SPACE]", "[en]", "[START]"]
unk_token = "[UNK]"
pretokenizer = pre_tokenizers.Whitespace()
decoder = None 

def train_tokenizer():

    tok = Tokenizer(models.BPE(unk_token=unk_token))
    tok.normalizer = norm_sequence([NFKC(), Lowercase()])
    tok.pre_tokenizer = pretokenizer
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_token,
        show_progress=True,
        max_token_length=20
    )
    
    tok.train(txt_path, trainer)
    tok.save(save_path,pretty=False)

if __name__ == "__main__":
    train_tokenizer()
    
