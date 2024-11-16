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
                        save_dir = '/mloscratch/homes/mmeyer/personalized-collaborative-llms/src/data/datasets/slimp/'):
    
    text_char_length = 10_000_000

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
            if source not in used_categories:
                continue
            current_len = token_count.get(source, 0)
            
            if current_len < text_char_length:
                new_text = loaded_dataset.get(source, [])
                new_text.append(sample['text'])
                loaded_dataset[source] = new_text
                new_count = token_count.get(source, 0) + len(sample['text'].split())
                token_count[source] = new_count

            if all(token_count[t] >= text_char_length for t in used_categories):
                break
        os.makedirs(save_dir)
        
        f = open(names,'a')
        
        tokenized = {}
        for dataset_name in token_count.keys():
            if dataset_name not in used_categories:
                continue
            random.shuffle(loaded_dataset[dataset_name])
            #print(loaded_dataset[dataset_name][:5])
            print(len(loaded_dataset[dataset_name]))
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
            print(dataset_name)
            tokenized[dataset_name] = np.load(os.path.join(save_dir,'slimpajama_'+dataset_name+'.npy'))
        for d in tokenized:
            print(tokenized[d].shape[0])
            tokenized[d] = tokenized[d]
            
        return tokenized


def get_slimp_dataset(alpha:float, num_clients=10, num_classes=4, num_tokens_per_client=500_000, test_ratio = 0.5,
                      save_dir = '/mloscratch/homes/mmeyer/personalized-collaborative-llms/src/data/datasets/slimp/'):
    directory = os.path.join(save_dir,f'slimpajama_{num_classes}_{num_clients}_{num_tokens_per_client}_{test_ratio}_{alpha}')
    if not os.path.isdir(directory):
        os.mkdir(directory)
        #note: num_tokens_per_client is the size of the train set
        data = generate_slimp_dataset()
        #data is a dictionary, one tokenized dataset for every class
        #of the form "name":np.ndarray() with one dimension
        train_data = []
        test_data = []
        used_tokens = {dataset:0 for dataset in data} #we count how much of each category has been used
        datasets = [dataset for dataset in data.keys()]
        tokens_matrix = np.zeros((num_classes, num_clients), dtype = np.uint32)

        

        for cli in range(num_clients):
            if alpha==0:
                distribution=np.zeros(num_clients, dtype=int)
                distribution[cli%num_classes]=1
                cli_train = np.array([], dtype=np.uint16)
                cli_test = np.array([], dtype=np.uint16)
                train_num = num_tokens_per_client
                test_num = int(test_ratio*num_tokens_per_client)
                dataset = datasets[cli%num_classes]
                cli_train = data[dataset][used_tokens[dataset]:used_tokens[dataset]+train_num]
                used_tokens[dataset]+=train_num
                tokens_matrix[datasets.index(dataset),cli]+=train_num
                cli_test = data[dataset][used_tokens[dataset]:used_tokens[dataset]+test_num]
                used_tokens[dataset]+=test_num
                train_data.append(cli_train)
                test_data.append(cli_test)

                
            else:
                sorted_datasets = sorted(used_tokens, key=lambda item: used_tokens[item], reverse=True) #list of the datasets' names in descending order of how much they have been used so far
                distribution = np.sort(np.random.dirichlet(np.repeat(alpha, num_classes)) ) #in ascending order
                cli_train = np.array([], dtype=np.uint16)
                cli_test = np.array([], dtype=np.uint16)
                train_num = num_tokens_per_client
                test_num = int(test_ratio*num_tokens_per_client)
                
                train_tokens = (train_num*distribution).astype(np.uint32)
                
                test_tokens = (test_num*distribution).astype(np.uint32)
                
                for i,dataset in enumerate(sorted_datasets):
                    j = datasets.index(dataset)
                    if train_tokens[i] > 3000:
                        cli_train = np.concatenate((cli_train, data[dataset][used_tokens[dataset]:used_tokens[dataset]+train_tokens[i]]))
                        used_tokens[dataset]+=train_tokens[i]
                        tokens_matrix[j, cli]+=train_tokens[i]
                        cli_test = np.concatenate((cli_test, data[dataset][used_tokens[dataset]:used_tokens[dataset]+test_tokens[i]]))
                        used_tokens[dataset]+=test_tokens[i]
                train_data.append(cli_train)
                test_data.append(cli_test)
            with open(os.path.join(directory,f'{cli}_train.npy'), 'wb') as f:
                np.save(f, cli_train)
            with open(os.path.join(directory,f'{cli}_test.npy'), 'wb') as f:
                np.save(f, cli_test)
            
        print(tokens_matrix)
    else:
        train_data=[]
        test_data = []
        for cli in range(num_clients):
            
            train_data.append(np.load(os.path.join(directory, f'{cli}_train.npy')))
            test_data.append(np.load(os.path.join(directory, f'{cli}_test.npy')))
    
    return {
        "train":train_data,
        "val":test_data
    }





