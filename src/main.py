import argparse
import copy
import inspect
import json
import os
import random
import sys
from argparse import Namespace

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import List, Dict, Union, Callable
import wandb

import config
import distributed
from data.utils import get_dataset
from models.utils import get_model
from optim.lora import train_lora

from hetlora.clients import Client
from hetlora.distribute import distribute

os.environ["WANDB__SERVICE_WAIT"] = "300"


def redistribute(clients: List[List[nn.Module | Optimizer | LRScheduler]], global_model:List[nn.Module | Optimizer | LRScheduler]) -> None:
    weights = {}
    for name, param in global_model[0].named_parameters():
        if (("lora_A" in name) or ("lora_B" in name)):
            weights[name] = param.data.clone()
    
    for client in clients:
        for name, param in client[0].named_parameters():
            if (("lora_A" in name) or ("lora_B" in name)):
                param.data = weights[name]

def get_args() -> Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_format', choices=config.registered_formats())
    args, rem_args = parser.parse_known_args()
    return config.parse_args_with_format(format=args.config_format, base_parser=parser, args=rem_args, namespace=args)


def get_exp_name(args: Namespace) -> str:
    """ Returns the name of the experiment, used for saving models and wandb. """
    exp_name = f"{args.model}_lr{args.lr}_bs{args.batch_size}x{args.acc_steps}_{args.world_size}nodes"
    if args.wandb_run_prefix != 'none':
        exp_name = args.wandb_run_prefix + '_' + exp_name
    if args.wandb_group != 'none':
        exp_name = args.wandb_group + '_' + exp_name
    if 'lora' in args.model:
        exp_name += f'_lora__rank{args.lora_rank}_alpha{args.lora_alpha}_dropout{args.lora_dropout}'
    exp_name += f'_seed={args.seed}'
    return exp_name

def prepare_model(args: Namespace, distributed_backend, device_type):
        model = get_model(args).to(args.device)
        model = distributed_backend.transform_model(model)

        group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs()
        param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
        optimized_params_cnt = 0
        for g in group_specs:
            params = []
            for p_name in g['params']:
                translated_p_names = distributed_backend.translate_model_parameter_name_for_node(p_name)
                params += [param_name_mapping[p_name] for p_name in translated_p_names]
            g['params'] = params
            optimized_params_cnt += sum([p.numel() for p in g['params']])

        print('number of optimized parameters: %.2fM' % (optimized_params_cnt / 1e6,))

        if args.opt == 'adamw':
            use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
            print(f'using fused AdamW: {use_fused}')
            extra_args = dict(fused=True) if use_fused else dict()
            opt = torch.optim.AdamW(group_specs, lr=args.lr, betas=(args.beta1, args.beta2),
                                    weight_decay=args.weight_decay, **extra_args)
        else:
            opt = torch.optim.SGD(group_specs, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

        if args.scheduler != 'none':
            if args.scheduler in ['cos', 'linear']:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt, max_lr=args.lr,
                                                                total_steps=args.iterations,
                                                                pct_start=args.warmup_percent,
                                                                anneal_strategy=args.scheduler,
                                                                cycle_momentum=False, div_factor=1e2,
                                                                final_div_factor=.05)
            else:
                raise NotImplementedError(f'Unknown scheduler type: {args.scheduler}.')
        else:
            scheduler = None
        return model, opt, scheduler


def main(args: Namespace) -> None:
    print(args)
    torch.backends.cuda.matmul.allow_tf32 = True  # allows us to make sure we're able to use tensor float32 during training
    torch.backends.cudnn.allow_tf32 = True

    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)

    args.device = torch.device(args.device)
    device_type = 'cuda' if 'cuda' in str(args.device) else 'cpu'
    if device_type == 'cuda':
        torch.cuda.set_device(args.device)
    

    print(args.method)
    print(type(args.method))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading dataset '{args.dataset}'")

    data = get_dataset(args)
    print(type(data['train']))
    print(type(data['train'][0]))
    print("dataset loaded!")
    clients = []

    #if args.method in ['hetlora', 'flexlora']:
        #assert len(args.hetlora_ranks)==args.num_clients, "Please provide num_clients lora ranks."
        #args.lora_rank = max(args.hetlora_ranks) #we use this rank to initialize the global model
    
    

    if args.method == 'homogeneous':
        for i in range(args.num_clients):
            clients.append(list(prepare_model(args=args, distributed_backend=distributed_backend, device_type=device_type)))
            global_model=None
    elif args.method in ['hetlora', 'flexlora', 'ffa','ffa_inversed', 'fedavg', 'fedsa']:
        
        
        rankslist = args.hetlora_ranks
        if rankslist == None:
            rankslist = [args.lora_rank]*args.num_clients
        print(rankslist)
        args.lora_rank = max(rankslist)
        args.lora_alpha = 2.0*args.lora_rank
        global_model = list(prepare_model(args=args, distributed_backend=distributed_backend, device_type=device_type))

        if args.method in ['hetlora', 'flexlora']:
            args_dict = vars(args)

            configslist = []
            for i in range(args.num_clients):
                new_args = Namespace(**args_dict)
                new_args.lora_rank = rankslist[i]
                new_args.lora_alpha = rankslist[i]*2.0
                configslist.append(new_args)
                
            print(configslist)

            for i in range(args.num_clients):
                clients.append(list(prepare_model(args=configslist[i], distributed_backend=distributed_backend, device_type=device_type)))
        else:
            for i in range(args.num_clients):
                clients.append(list(prepare_model(args=args, distributed_backend=distributed_backend, device_type=device_type)))
        if args.method in ['ffa','ffa_inversed']:
            global_model[0].reset_parameters_lora()
            for client in clients:
                client[0].reset_parameters_lora()
        if args.method in ['ffa', 'ffa_inversed','fedsa']:
            #initialize all lora weights in exactly the same way for ffa
            redistribute(clients, global_model)
        
        global_model[0] = torch.compile(global_model[0], dynamic=True)
    else:
        raise NotImplementedError(f"No training method implemented for model type '{args.model}'.")
    args.world_size = distributed_backend.get_world_size()
    exp_name = get_exp_name(args)
    if distributed_backend.is_master_process() and args.wandb:
        params_copy = copy.deepcopy(vars(args))
        del params_copy['device']
        wandb.init(project=args.wandb_project, name=args.wandb_name, group=args.wandb_group, config=params_copy)

    ckpt_path = os.path.join(args.results_base_folder, args.dataset, args.model, exp_name)
    if not os.path.exists(ckpt_path):
        if distributed_backend.is_master_process():
            os.makedirs(ckpt_path)
    elif os.path.isfile(os.path.join(ckpt_path, 'summary.json')):  # the experiment was already completed
        print(f"Already found experiment '{ckpt_path}'.\nSkipping.")
        #sys.exit(0)

    if args.model == 'lora':
        train = train_lora
    else:
        raise NotImplementedError(f"No training method implemented for model type '{args.model}'.")

    print(f'\nTraining model={args.model} \n{vars(args)}\n')

    stats = train(clients, data, args.iterations, args.acc_steps, args.batch_size, args.sequence_length,
                  eval_freq=args.eval_freq, method=args.method,
                  distributed_backend=distributed_backend,
                  extra_args=args, global_model=global_model) #TODO: make sure the correct method is being used, I feel like something might be wrong

    args.device = None
    args.dtype = None
    stats['args'] = vars(args)
    if distributed_backend.is_master_process():
        with open(f'{ckpt_path}/summary.json', 'w') as fs:
            json.dump(stats, fs)
    distributed_backend.finalize()


if __name__ == '__main__':
    args = get_args()
    main(args)
