import torch

# Hyperparameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
BATCH_SIZE = 64 # how many independent sequences will we process in parallel?
BLOCK_SIZE = 256 # what is the maximum context length for predictions?
EVAL_ITERS = 50
N_EMBED = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2