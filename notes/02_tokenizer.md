# 02 - Tokenizer

## What a Tokenizer Does
- Converts text (strings) → token IDs (integers)
- Does NOT create embeddings — that's the model's job

```
Text → [Tokenizer] → Token IDs (integers)       ← tiktoken does this
Token IDs → [Embedding Layer] → Embeddings        ← model does this
```

## BPE (Byte Pair Encoding)
- GPT-2 uses BPE tokenization
- Splits text into subwords, not full words
- Common subwords get their own token
- Example: "unhappiness" → ["un", "happ", "iness"] → [388, 31180, 1108]
- vocab_size = 50257 comes from GPT-2's BPE vocabulary

## Why tiktoken?
- OpenAI's fast BPE tokenizer (used by GPT-2/3/4)
- We use the "gpt2" encoding to match our model's vocab_size
- Simple API: `encode(text) → [int]`, `decode([int]) → text`
