import torch

from models.lora import GPTLoRA

from .clients import Client

def distribute(global_model: GPTLoRA, hetlora_ranks: list[int], opt, scheduler) -> list[Client]:
    return [[global_model.truncate(rank), opt, scheduler] for rank in hetlora_ranks]