from .utils import *
from .wikitext_split import get_split_multi_data
import numpy as np

from scipy.stats import entropy, dirichlet

def make_matrix(len_data, lang_data,sample):
    lang_counters = [0,0,0,0]
    '''
    sampler = dirichlet(4*[1])
    sample = sampler.rvs()'''


    new_size = 0
    for i in range(4):
        new_size += int(sample[0][i]*len_data)

    client_counters = [0,0,0,0]
    ranges = [int(sample[0][i]*len_data) for i in range(4)]
    
    data_matrix = np.zeros((4,new_size))

    for l,lang in enumerate(lang_data):
        ranges = [int(sample[0][i]*len_data) for i in range(4)]

        for idx, ran in enumerate(ranges):
            data_matrix[idx, client_counters[idx]:client_counters[idx]+ran] = lang[0][lang_counters[l]:lang_counters[l]+ran]
            lang_counters[l]+=ran
            client_counters[idx]+=ran
        sample = sample[:,[3,0,1,2]]
    return data_matrix

def get_wiki_multilingual(dataset_name):

    if dataset_name == 'wiki_multilingual_1':
        sample = np.array([[0.05,0.,0.,0.95]])
    elif dataset_name == 'wiki_multilingual_2':
        sample = np.array([[0.367,0.001,0.002,0.63]])
    elif dataset_name == 'wiki_multilingual_3':
        sample = np.array([[0.054,0.787,0.084,0.075]])
    elif dataset_name == 'wiki_multilingual_4':
        sample = np.array([[0.179,0.056,0.056,0.709]])
    elif dataset_name == 'wiki_multilingual_5':
        sample = np.array([[0.399,0.032,0.247,0.322]])
    elif dataset_name == 'wiki_multilingual_6':
        sample = np.array([[0.24,0.25,0.27,0.24]])
    else:
        raise NotImplementedError("Please choose a valid datset!")

    LEN_DATASETS = 84000000
    LEN_TESTS = 16000000

    en_data = get_split_multi_data('en')
    fr_data = get_split_multi_data('fr')
    it_data = get_split_multi_data('it')
    de_data = get_split_multi_data('de')
    
    lang_data = [en_data['train'], fr_data['train'], it_data['train'], de_data['train']]
    eval_data = [en_data['val'], fr_data['val'], it_data['val'], de_data['val']]
    train_matrix = make_matrix(LEN_DATASETS, lang_data,sample)
    test_matrix = make_matrix(LEN_TESTS, eval_data,sample)
    return {
        "train":[train_matrix[i,:] for i in range(4)],
        "val":[test_matrix[i,:] for i in range(4)]
    }
    



