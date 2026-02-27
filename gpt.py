import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os
import urllib.request

random.seed(42)

# ----------------------------
# dataset
# ----------------------------
if not os.path.exists('input.txt'):
    url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(url, 'input.txt')

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

chars = sorted(set(''.join(docs)))
BOS = len(chars)
vocab_size = len(chars) + 1
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}
print(f"vocab size: {vocab_size}")

# ----------------------------
# model
# ----------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean() + self.eps)

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=16, n_head=4):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(16, n_embd)

        self.attn_wq = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_wk = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_wv = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_wo = nn.Linear(n_embd, n_embd, bias=False)

        self.fc1 = nn.Linear(n_embd, 4*n_embd, bias=False)
        self.fc2 = nn.Linear(4*n_embd, n_embd, bias=False)

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.norm1 = RMSNorm(n_embd)
        self.norm2 = RMSNorm(n_embd)

    def forward_step(self, token_id, pos_id, keys, values):
        dev = self.wte.weight.device
        # token + pos embedding
        x = self.wte(torch.tensor(token_id, device=dev)) + self.wpe(torch.tensor(pos_id, device=dev))
        x = self.norm1(x)

        # attention
        x_res = x
        x = self.norm1(x)

        q = self.attn_wq(x)
        k = self.attn_wk(x)
        v = self.attn_wv(x)

        keys.append(k)
        values.append(v)

        attn_out = []

        for h in range(self.n_head):
            hs = h * self.head_dim
            q_h = q[hs:hs+self.head_dim]

            k_h = torch.stack([kk[hs:hs+self.head_dim] for kk in keys])
            v_h = torch.stack([vv[hs:hs+self.head_dim] for vv in values])

            att = (k_h @ q_h) / math.sqrt(self.head_dim)
            w = F.softmax(att, dim=0)

            head = (w.unsqueeze(1) * v_h).sum(dim=0)
            attn_out.append(head)

        x_attn = torch.cat(attn_out)
        x = self.attn_wo(x_attn)
        x = x + x_res

        # MLP
        x_res = x
        x = self.norm2(x)
        x = self.fc2(F.relu(self.fc1(x)))
        x = x + x_res

        logits = self.lm_head(x)
        return logits

# ----------------------------
# training
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MiniGPT(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.85, 0.99), eps=1e-8)
num_params = sum(p.numel() for p in model.parameters())
print(f"num params: {num_params}")

num_steps = 1000

for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [stoi[c] for c in doc] + [BOS]
    n = min(16, len(tokens) - 1)

    keys, values = [], []
    losses = []

    for pos in range(n):
        logits = model.forward_step(tokens[pos], pos, keys, values)
        loss = F.cross_entropy(
            logits.unsqueeze(0),
            torch.tensor([tokens[pos+1]], device=device)
        )
        losses.append(loss)

    loss = torch.stack(losses).mean()

    lr_t = 0.01 * (1 - step / num_steps)
    for g in optimizer.param_groups:
        g['lr'] = lr_t

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.item():.4f}", end='\r')

# ----------------------------
# inference
# ----------------------------
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [], []
    token = BOS
    sample = []

    for pos in range(16):
        logits = model.forward_step(token, pos, keys, values)
        probs = F.softmax(logits / 0.5, dim=0)
        token = torch.multinomial(probs, 1).item()

        if token == BOS:
            break

        sample.append(itos[token])

    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
