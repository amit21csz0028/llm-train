# 03 - Dataset & How LLM Training Data Works

## Core Concept: LLMs are Next-Token Predictors
The ENTIRE job of a GPT model is:
```
Given: "The cat sat on the"
Predict: "mat" (the next token)
```
The magic comes from doing this extremely well over billions of examples.

## Training Data Flow

### Step 1: Raw text
```
"The cat sat on the mat"
```

### Step 2: Tokenize
```
[464, 3797, 3332, 319, 262, 2603]
```

### Step 3: Create input/target pairs (the key insight!)
```
Input:  [464, 3797, 3332, 319, 262]    → "The cat sat on the"
Target: [3797, 3332, 319, 262, 2603]    → "cat sat on the mat"
         ↑ shifted by one position!
```

### Step 4: Model predicts at every position simultaneously
```
Position 0: given "The"           → predict "cat"
Position 1: given "The cat"       → predict "sat"
Position 2: given "The cat sat"   → predict "on"
```

### Step 5: Loss (Cross-Entropy)
Compare predictions vs targets — how wrong were we?

### Step 6: Backprop → update weights → repeat

## Why block_size + 1?
- We need `block_size` tokens for input AND `block_size` tokens for target
- Since target is just input shifted by 1, we grab `block_size + 1` tokens total
- chunk[:-1] = input (first block_size tokens)
- chunk[1:] = target (last block_size tokens)

## Causal Language Modeling vs Autoregressive

| Term | Meaning |
|---|---|
| **Causal LM** | Training objective — predict next token, each position can only see past tokens (not future) |
| **Autoregressive** | Generation method — generate one token at a time, feed output back as input |

- **Training**: Causal LM (all positions predicted in parallel, masked from future)
- **Inference**: Autoregressive (one token at a time, sequential)
- GPT-2 uses both
- BERT is different — Masked LM, can see both past and future

```
GPT (Causal):    "The [cat] sat ___"  → can only look left ←
BERT (Masked):   "The ___ sat on"     → can look both ways ← →
```
