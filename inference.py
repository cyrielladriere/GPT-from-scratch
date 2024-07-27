import torch
from data_processing import decode, pre_process_data
from model import GPTLanguageModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'


with open('artifacts/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocab_size = pre_process_data(text)

model = GPTLanguageModel(vocab_size)
model.load_state_dict(torch.load("artifacts/model.pt"))
model.to(device)
model.eval()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

# write to file
open('artifacts/output.txt', 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))