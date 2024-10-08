from .utils import *
from .wikitext_split import get_split_multi_data
import numpy as np

from scipy.stats import entropy, dirichlet

def make_matrix(len_data, lang_data):
    lang_counters = [0,0,0,0]

    sampler = dirichlet(4*[1])
    sample = sampler.rvs()

    new_size = 0
    for i in range(4):
        new_size += int(sample[0][i]*len_data)

    client_counters = [0,0,0,0]
    ranges = [int(sample[0][i]*len_data) for i in range(4)]
    
    memmap_concat = np.zeros((4,new_size))

    for l,lang in enumerate(lang_data):
        ranges = [int(sample[0][i]*len_data) for i in range(4)]

        for idx, ran in enumerate(ranges):
            memmap_concat[idx, client_counters[idx]:client_counters[idx]+ran] = lang[0][lang_counters[l]:lang_counters[l]+ran]
            lang_counters[l]+=ran
            client_counters[idx]+=ran
        sample = sample[:,[3,0,1,2]]
    return memmap_concat

def get_my_dataset():

    LEN_DATASETS = 84000000
    LEN_TESTS = 16000000

    en_data = get_split_multi_data('en')
    fr_data = get_split_multi_data('fr')
    it_data = get_split_multi_data('it')
    de_data = get_split_multi_data('de')
    
    lang_data = [en_data['train'], fr_data['train'], it_data['train'], de_data['train']]
    eval_data = [en_data['val'], fr_data['val'], it_data['val'], de_data['val']]
    train_matrix = make_matrix(LEN_DATASETS, lang_data)
    test_matrix = make_matrix(LEN_TESTS, eval_data)
    return {
        "train":[train_matrix[i,:] for i in range(4)],
        "val":[test_matrix[i,:] for i in range(4)]
    }
    



