'''from src.data.utils import *
from src.data.slim_pajama import get_slimp_dataset


d = get_slimp_dataset()
#print(d)
print(d['train'][0].shape)
print(d['val'][0].shape)'''

from src.data.wikimulti import download_wikimulti
import tiktoken

tokenizer = tiktoken.get_encoding('gpt2')
for lang in ['fr', 'it', 'de', 'nl']:
    d = download_wikimulti(lang, tokenizer)
    print(d.shape[0])