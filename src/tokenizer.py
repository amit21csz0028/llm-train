"""
Tokenizer — converts text <-> token IDs (integers)
- Uses tiktoken with GPT-2's BPE (Byte Pair Encoding) vocabulary
- BPE splits text into subwords, not full words
  e.g. "unhappiness" → ["un", "happ", "iness"] → [388, 31180, 1108]
- vocab_size = 50257 subword tokens in GPT-2's vocabulary

IMPORTANT distinction:
  Tokenizer: text → token IDs (integers)      ← this file
  Embedding: token IDs → vectors (floats)      ← model's job (learned during training)
"""
import tiktoken

class Tokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
    
    def encode(self, text):
        return self.enc.encode(text)
    
    def decode(self, tokens):
        return self.enc.decode(tokens)