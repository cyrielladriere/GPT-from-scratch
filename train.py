from data_processing import pre_process_data
from model import GPTLanguageModel, estimate_loss, get_batch
import torch
import time
from config import MAX_ITERS, LEARNING_RATE, EVAL_INTERVAL, DEVICE

with open('artifacts/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocab_size = pre_process_data(text)

model = GPTLanguageModel(vocab_size)
model.to(DEVICE)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

start_eval_time = time.time()
for iter in range(MAX_ITERS):

    # every eval_interval epochs evaluate the loss on train and val sets
    if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
        losses = estimate_loss(model)
        eval_time = f"[{(time.time()-start_eval_time)// 60:.0f}m {(time.time()-start_eval_time) % 60:.0f}s]"
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} {eval_time}")
        start_eval_time = time.time()

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "artifacts/model.pt")