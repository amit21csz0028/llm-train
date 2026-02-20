"""_summary_
    
"""


import torch
import torch.nn as nn
from config import GPTconfig

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. Token embedding table: vocab_size → n_embd
        # 2. Position embedding table: block_size → n_embd
        # 3. List of transformer blocks (n_layer of them)
        # 4. Final layer norm
        # 5. Output head: n_embd → vocab_size (to predict next token)

    def forward(self, idx):
        # idx is (batch, sequence_length) of token IDs
        # 1. Get token embeddings + position embeddings
        # 2. Pass through all transformer blocks
        # 3. Final layer norm
        # 4. Project to vocab_size → logits
        pass