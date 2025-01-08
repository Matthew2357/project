from datasets import load_dataset
import os
import tiktoken
import numpy as np

def download_fineweb(tokenizer:tiktoken.get_encoding,save_dir = '/mloscratch/homes/mmeyer/personalized-collaborative-llms/src/data/datasets/fineweb'):
    
    # use name="sample-10BT" to use the 10BT sample
    
    desired_tokens = 20_000_000
    if not os.path.exists(os.path.join(save_dir, 'fineweb.npy')):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if not os.path.exists(os.path.join(save_dir, 'fineweb.npy')):
            current_token_count = 0
            fw = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)
            all_tokens = np.array([], dtype=np.uint16)
            for sample in fw:
                
                text = sample['text']
                if current_token_count < desired_tokens:
                    temp = np.array(tokenizer.encode_ordinary(text), dtype=np.uint16)
                    current_token_count += temp.shape[0]
                    all_tokens = np.concatenate((all_tokens, temp))
                else:
                    break
            with open(os.path.join(save_dir, 'fineweb.npy'), 'wb') as f:
                np.save(f, all_tokens)

            return all_tokens
    else:
        return np.load(os.path.join(save_dir, 'fineweb.npy'))       

def get_fineweb(num_clients=12, num_tokens_per_client=500_000, test_ratio = 0.5): 
    train_data = []
    test_data = []
    test_tokens_per_client = int(num_tokens_per_client*(test_ratio/(1-test_ratio)))
    tokenizer = tiktoken.get_encoding('gpt2')  
    data = download_fineweb(tokenizer)
    assert num_clients*num_tokens_per_client/(1-test_ratio) <= data.shape[0], "not enough data available!"
    current_token_count = 0
    for cl in range(num_clients):
        train_data.append(data[current_token_count:current_token_count+num_tokens_per_client])
        current_token_count+= num_tokens_per_client
        test_data.append(data[current_token_count:current_token_count+test_tokens_per_client])
        current_token_count+=test_tokens_per_client
    return {
        "train":train_data,
        "val":test_data
    }





    

    
