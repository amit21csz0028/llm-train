# 01 - Config & Project Setup

## Why @dataclass for config?
- When a class is just holding data/config with no methods, `@dataclass` is clean and Pythonic
- Auto-generates `__init__`, `__repr__`, etc.

## GPT-2 Small Hyperparameters
| Parameter | Value | Meaning |
|---|---|---|
| `vocab_size` | 50257 | Number of tokens in GPT-2 BPE vocabulary |
| `n_embd` | 768 | Embedding dimension (also called `d_model`) |
| `n_head` | 12 | Number of attention heads |
| `n_layer` | 12 | Number of transformer blocks |
| `block_size` | 1024 | Max sequence length (context window) |
| `dropout` | 0.1 | Regularization to prevent overfitting |
| `batch_size` | 8 | Number of sequences processed simultaneously |

## Key Concept
- Config is the **single source of truth** — every other file imports from here
- Keeps hyperparameters in one place, easy to experiment by changing values
