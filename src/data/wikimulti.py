import os
import numpy as np
import tiktoken
from datasets import load_dataset
def download_wikimulti(lang:str, tokenizer:tiktoken.get_encoding,save_dir = '/mloscratch/homes/mmeyer/personalized-collaborative-llms/src/data/datasets/wikimulti'):
    
    #lang should be one of  'fr', 'it', 'de', 'nl'
    desired_tokens = 10_000_000
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if not os.path.exists(os.path.join(save_dir, f'{lang}.npy')):
        current_token_count = 0
        data = load_dataset("wiki40b", lang ,split='validation', streaming=True)
        all_tokens = np.array([], dtype=np.uint16)
        for sample in data:
            text = sample['text']
            if current_token_count < desired_tokens:
                temp = np.array(tokenizer.encode_ordinary(text), dtype=np.uint16)
                current_token_count += temp.shape[0]
                all_tokens = np.concatenate((all_tokens, temp))
            else:
                break
        with open(os.path.join(save_dir, f'{lang}.npy'), 'wb') as f:
            np.save(f, all_tokens)


        return all_tokens
    else:
        return np.load(os.path.join(save_dir, f'{lang}.npy'))
    
def get_wikimulti(alpha:float, num_clients=10, num_classes=4, num_tokens_per_client=500_000, test_ratio = 0.5,save_dir = '/mloscratch/homes/mmeyer/personalized-collaborative-llms/src/data/datasets/wikimulti'):
    directory = os.path.join(save_dir,f'wikimulti_{num_classes}_{num_clients}_{num_tokens_per_client}_{test_ratio}_{alpha}')
    if not os.path.isdir(directory):
        os.mkdir(directory)
        tokenizer = tiktoken.get_encoding('gpt2')
        #note: num_tokens_per_client is the size of the train set
        langs = ['fr', 'it', 'de', 'nl']
        data = {}
        for lang in langs:
            data[lang] = download_wikimulti(lang, tokenizer)
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





