import os
import requests
import tiktoken
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# read the dataset

# input_file_path = os.path.join(os.path.dirname('.'), 'input.txt')
# if not os.path.exists(input_file_path):
#     data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
#     with open(input_file_path, 'w') as f:
#         f.write(requests.get(data_url).text)

# hyperparameters
batch_size = 4 # how many independent sequences to be processed in parallel
block_size = 8 # maximum context length for predictions
max_iters = 3000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-2
device = 'mps' if torch.cuda.is_available() else 'cpu'

# set seed 
torch.manual_seed(42)

# read it to inspect it
with open('kz_nagyz.txt','r',encoding='UTF-8') as f:
    text = f.read()

# unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# encode the entire text and store it into a torch.Tensor
data = torch.tensor(encode(text),dtype=torch.long)

# let's split the data into train and val datasets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# generate a batch of data of inputs x and targets y
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# bigram language model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits of the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self, idx, targets=None):
        # idx, targets are both (B,T) tensors of integers
        logits = self.token_embedding_table(idx) # B,T,C
        
        if targets is None:
            loss = None
        else:            
            # torch.nn.functional cross_entropy wants BCT instead of BTC, C as a second dimension
            B, T, C = logits.shape
            # let's just stretch all the arrays and reshape the logits to 2D matrix
            logits = logits.view(B*T, C)
            # don't forget the targets too because this transformation should affect the targets equally
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # obtain predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # randomly sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx,idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create an optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# training loop
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')
    
    # evaluate the loss
    logits, loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context,max_new_tokens=500)[0].tolist()))