from dataclasses import dataclass

@dataclass
class GPTconfig:
    vocab_size: int = 50257
    n_embd: int = 768 #d_model
    n_head: int = 12
    n_layer: int = 12
    block_size: int = 1024 #context length
    dropout: float = 0.1
    batch_size: int = 8
    
    