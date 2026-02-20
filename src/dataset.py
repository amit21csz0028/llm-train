"""
The Big Picture: How LLMs Actually Work
Before we write dataset.py, let's understand what we're actually training the model to do. This is the most important concept:

LLMs are next-token predictors
The ENTIRE job of a GPT model is:


Given: "The cat sat on the"
Predict: "mat" (the next token)
That's it. The magic comes from doing this extremely well over billions of examples.

How training works (the data flow):

Step 1: Raw text
  "The cat sat on the mat"

Step 2: Tokenize
  [464, 3797, 3332, 319, 262, 2603]

Step 3: Create input/target pairs (this is key!)
  Input:  [464, 3797, 3332, 319, 262]    → "The cat sat on the"
  Target: [3797, 3332, 319, 262, 2603]    → "cat sat on the mat"
         ↑ shifted by one position!

Step 4: Model predicts next token at EVERY position simultaneously
  Position 0: given "The"           → predict "cat"
  Position 1: given "The cat"       → predict "sat"
  Position 2: given "The cat sat"   → predict "on"
  ... and so on

Step 5: Loss = how wrong were the predictions?
  (compare predictions vs targets using Cross-Entropy Loss)

Step 6: Backprop → update weights → repeat
Key insight: The input and target are the same sequence shifted by 1 position. This is called causal language modeling. 
"""

import torch
from torch.utils.data import Dataset
from tokenizer import Tokenizer
from config import GPTconfig

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
      self.block_size = block_size
      # 1. Encode the entire text into tokens
      self.tokens = tokenizer.encode(text)
      # 2. Store the tokens as a tensor
      self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        
    
    def __len__(self):
        # How many chunks of block_size can we extract?
        return len(self.tokens) // self.block_size
    
    def __getitem__(self, idx):
        # Get a chunk starting at position idx * block_size
        start = idx * self.block_size
        chunk = self.tokens[start: start + self.block_size +1] # block_size+1
        # x = chunk[:-1]   (all except last)
        # y = chunk[1:]    (all except first)  ← shifted by 1!
        # return x, y
        x = chunk[:-1]   #(all except last)
        y = chunk[1:]    #(all except first)  ← shifted by 1!
        return x, y