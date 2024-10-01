import torch

from models.lora import GPTLoRA

from .clients import Client

def distribute(global_model: GPTLoRA, hetlora_ranks: list[int], opt, scheduler):
    max_rank = max(hetlora_ranks)
    return [[global_model.truncate(rank, max_rank), opt, scheduler] for rank in hetlora_ranks] #TODO: fix this because the model is currently a none type