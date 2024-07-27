import torch
from config import BLOCK_SIZE, BATCH_SIZE, DEVICE

def encode(s):
    # encoder: take a string, output a list of integers
    return [STOI[c] for c in s]

def decode(l):
    # decoder: take a list of integers, output a string
    return ''.join([ITOS[i] for i in l])


def pre_process_data(text):
    # all the unique characters in the text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # create a mapping from characters to integers
    global STOI, ITOS
    STOI = { ch:i for i,ch in enumerate(chars) }
    ITOS = { i:ch for i,ch in enumerate(chars) }

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val

    global TRAIN_DATA, VAL_DATA
    TRAIN_DATA = data[:n]
    VAL_DATA = data[n:]

    return vocab_size

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = TRAIN_DATA if split == 'train' else VAL_DATA
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y