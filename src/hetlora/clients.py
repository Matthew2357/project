from argparse import Namespace

class Client():
    def __init__(self, args: Namespace, client_rank: int):
        assert isinstance(int, client_rank), "LoRA rank must be an integer."
        assert client_rank >= 1, "LoRA rank must be positive."
        self.args = args
        self.args.lora_rank = client_rank
            

