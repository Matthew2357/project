import json
from collections import Counter
import random
import os
from datasets import load_dataset
import numpy as np
import tiktoken
import random
from .utils import *

# import torchtext


tokenizer = tiktoken.get_encoding("gpt2")

random.seed(42)
np.random.seed(42)

def generate_slimp_dataset(data_path = '/mloscratch/homes/mmeyer/personalized-collaborative-llms/src/data/',
                        num_clients = 4, save_dir = '/mloscratch/homes/mmeyer/personalized-collaborative-llms/src/data/datasets/slimp/'):
    
    text_char_length = 6_000_000

    train_size_large = 1_000_000
    validation_size = 200_000
    test_size = 200_000
    names = save_dir+'names.txt'
    
    used_categories = set(['RedPajamaBook', 'RedPajamaStackExchange', 'RedPajamaGithub', 'RedPajamaArXiv'])
    if not os.path.exists(save_dir):
        dataset = load_dataset("DKYoon/SlimPajama-6B", split="train", cache_dir=data_path, streaming=True)
        loaded_dataset = {}
        token_count = {}
        
        for sample in dataset:
            source = sample['meta']['redpajama_set_name']
            
            current_len = token_count.get(source, 0)
            if current_len < text_char_length:
                new_text = loaded_dataset.get(source, [])
                new_text.append(sample['text'])
                loaded_dataset[source] = new_text
                new_count = token_count.get(source, 0) + len(sample['text'])
                token_count[source] = new_count

            if all(token_count[t] >= text_char_length for t in token_count.keys()):
                break
        os.makedirs(save_dir)
        
        f = open(names,'a')
        
        tokenized = {}
        for dataset_name in token_count.keys():
            if dataset_name not in used_categories:
                continue
            random.shuffle(loaded_dataset[dataset_name])
            train_tokenized = np.array(tokenizer.encode_ordinary(" ".join(loaded_dataset[dataset_name])), dtype=np.uint16)
            tokenized[dataset_name] = train_tokenized
            
            print(f"original size of data of class {dataset_name}:", len(train_tokenized))

            
            np.save(os.path.join(save_dir,'slimpajama_'+dataset_name+'.npy'), train_tokenized)
            f.write(dataset_name)
            f.write('\n')
        f.close()
        return tokenized
    else:
        f = open(names,'r')
        tokenized = {}
       
        for dataset_name in f.read().split('\n'): 
            tokenized[dataset_name] = np.load(os.path.join(save_dir,'slimpajama_'+dataset_name+'.npy'))
        for d in tokenized:
            print(tokenized[d].shape[0])
        return tokenized
    
def separate_data(data, num_clients, num_classes, alpha):
    
    #data is a dictionary, one tokenized dataset for every class
    #of the form "name":np.ndarray() with one dimension

    total_data = sum([data[d].shape[0] for d in data])
    print(total_data)
    least_samples = 500
    i=0
    min_size = 0
    distributions = np.zeros((num_classes, num_clients), dtype=np.uint32)
    sizes = {dataset: data[dataset].shape[0] for dataset in data.keys()}
    while min_size < least_samples:
        i+=1
        
        
        for idx, dataset in enumerate(data.keys()):
           if sizes[dataset] > 100:
                temp = np.random.dirichlet(np.repeat(alpha, num_clients))
                
                temp = np.array([p*(np.sum(distributions, axis=0)[cli]<total_data/num_clients) for p,cli in zip(temp,range(num_clients))])
                temp = temp/temp.sum()
                
                
                temp = (temp*min(sizes[dataset], data[dataset].shape[0]/(num_clients/5))).astype(np.uint32)
                sizes[dataset] -= temp.sum()
                
                distributions[idx,:]+=temp
        min_size = np.sum(distributions, axis=0).min()
        print(min_size)
        print(sizes)
        
    print(np.sum(distributions, axis=1))  
    print(distributions)
    
    distributions = np.cumsum(distributions, axis=1)
    
    

    print(i)
    
    
    
    sep_data = [np.array([]) for _ in range(num_clients)]
    for idx, dataset in enumerate(data.keys()):
        for cli in range(num_clients):
            if cli==0:
                sep_data[cli]=np.concatenate((sep_data[cli],data[dataset][:distributions[idx,cli]]))
            else:
                sep_data[cli]=np.concatenate((sep_data[cli],data[dataset][distributions[idx, cli-1]:distributions[idx,cli]]))
    return sep_data

    
def get_slimp_dataset(alpha, num_clients=4, num_classes=4):
    data = generate_slimp_dataset()
    distributed = separate_data(data, num_clients, num_classes, alpha)
    final_data = {"train":[], "val":[]}
    for cli in distributed:
        cut = int(cli.shape[0]*0.84)
        train = cli[:cut]
        val = cli[cut:]
        final_data['train'].append(train)
        final_data['val'].append(val)
    return final_data



