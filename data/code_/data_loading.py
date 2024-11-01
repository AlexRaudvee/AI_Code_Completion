import spacy

from torchtext.data import Field 
from torchtext.datasets import Multi30k

from config import  from_, to

spacy_lang_in = spacy.load(from_)
spacy_lang_out = spacy.load(to)

def tokenize_lang_in(text):
    return [tok.text for tok in spacy_lang_in.tokenizer(text)]

def tokenize_lang_out(text):
    return [tok.text for tok in spacy_lang_out.tokenizer(text)]

lang_in = Field(tokenize=tokenize_lang_in, lower=True, init_token='<sos>', eos_token='<eos>')
lang_out = Field(tokenize=tokenize_lang_out, lower=True, init_token='<sos>', eos_token='<eos>')

train_data, valid_data, test_data = Multi30k.splits(
    exts=(f'.{from_}', f'.{to}'), fields=(lang_in, lang_out), root='../text_data/.data'
)

lang_in.build_vocab(train_data, max_size=10000, min_freq=2)
lang_out.build_vocab(train_data, max_size=10000, min_freq=2)

