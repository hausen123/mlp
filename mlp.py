import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================================================
# Argument
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--corpus", type=str, default=None)

    parser.add_argument("--save_prefix", type=str, default="datastore")

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)

    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--tau", type=float, default=10.0)

    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--lambda_interp", type=float, default=0.45)

    parser.add_argument("--mode", type=str, default="full",
                        choices=["build", "knn", "train", "infer", "full"])

    parser.add_argument("--prompt", type=str,
                        default="Transformerの仕組みを説明してください。")

    parser.add_argument("--max_new_tokens", type=int, default=1024)

    return parser.parse_args()


# =========================================================
# Corpus
# =========================================================

def load_text_corpus(path, tokenizer, max_length):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenizer(text, return_tensors="pt",
                       truncation=False)["input_ids"][0]

    chunks = []
    for i in range(0, len(tokens) - max_length - 1, max_length):
        chunks.append(tokens[i:i+max_length+1])

    return chunks


# =========================================================
# Datastore Build
# =========================================================

def build_datastore(model, tokenizer, text_path,
                    target_layer_index, device,
                    save_prefix, max_length):

    model.eval()

    chunks = load_text_corpus(text_path, tokenizer, max_length)

    all_keys = []
    all_vals = []

    with torch.no_grad():
        for chunk in tqdm(chunks, desc="Building datastore"):

            chunk = chunk.unsqueeze(0).to(device)

            outputs = model(
                input_ids=chunk[:, :-1],
                output_hidden_states=True
            )

            hidden = outputs.hidden_states[target_layer_index]

            keys = hidden.reshape(-1, hidden.size(-1)).cpu()
            vals = chunk[:, 1:].reshape(-1).cpu()

            all_keys.append(keys)
            all_vals.append(vals)

    keys = torch.cat(all_keys).numpy().astype("float32")
    vals = torch.cat(all_vals).numpy().astype("int64")

    np.save(save_prefix + "_keys.npy", keys)
    np.save(save_prefix + "_vals.npy", vals)

    print("Datastore saved.")


# =========================================================
# kNN Target Build
# =========================================================

def compute_knn_targets(save_prefix, K, tau, batch_size=512):

    keys = np.load(save_prefix + "_keys.npy")
    vals = np.load(save_prefix + "_vals.npy")

    dim = keys.shape[1]
    res = faiss.StandardGpuResources()
    index_cpu = faiss.IndexFlatL2(dim)
    index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    index.add(keys)

    all_targets = []

    for start in tqdm(range(0, len(keys), batch_size), desc="Computing kNN targets"):
        end = min(start + batch_size, len(keys))
        batch_keys = keys[start:end]
        D, I = index.search(batch_keys, K + 1)

        for bi in range(end - start):
            gi = start + bi
            ii = I[bi]
            dd = D[bi]
            mask = ii != gi
            ii = ii[mask][:K]
            dd = dd[mask][:K]
            weights = np.exp(-dd / tau)
            token_probs = {}
            for idx, w in zip(ii, weights):
                token = int(vals[idx])
                token_probs[token] = token_probs.get(token, 0) + float(w)
            total = sum(token_probs.values())
            if total == 0:
                all_targets.append({})
                continue
            for k in token_probs:
                token_probs[k] /= total
            all_targets.append(token_probs)

    np.save(save_prefix + "_targets.npy",
            np.array(all_targets, dtype=object),
            allow_pickle=True)

    print("kNN targets saved.")


# =========================================================
# Dataset
# =========================================================

class MLPMemoryDataset(Dataset):

    def __init__(self, save_prefix):

        self.keys = torch.tensor(
            np.load(save_prefix + "_keys.npy"),
            dtype=torch.float32
        )

        self.vals = torch.tensor(
            np.load(save_prefix + "_vals.npy"),
            dtype=torch.long
        )

        self.targets = np.load(
            save_prefix + "_targets.npy",
            allow_pickle=True
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return self.keys[idx], self.vals[idx], self.targets[idx]


# =========================================================
# MLP Memory
# =========================================================

class MLPMemory(nn.Module):

    def __init__(self, hidden_dim, embed_weight):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.embed_weight = embed_weight.float()  # frozen, float32

    def forward(self, h):
        h = h.float()
        h = self.mlp(h)
        logits = torch.matmul(h, self.embed_weight.T)
        return logits


# =========================================================
# Training
# =========================================================

def collate_fn(batch):
    keys = torch.stack([b[0] for b in batch])
    vals = torch.stack([b[1] for b in batch])
    targets = [b[2] for b in batch]
    return keys, vals, targets

def train_mlp(model, save_prefix,
              alpha, batch_size,
              epochs, device, config=None):

    dataset = MLPMemoryDataset(save_prefix)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=collate_fn)

    hidden_dim = model.config.hidden_size
    embed_weight = model.get_input_embeddings().weight.detach()
    embed_weight.requires_grad = False

    mlp = MLPMemory(hidden_dim, embed_weight).to(device)

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=4e-4)

    mlp.train()

    for epoch in range(epochs):

        total_loss = 0

        for keys, true_token, target_dicts in tqdm(loader, desc=f"Epoch {epoch+1}"):

            keys = keys.to(device)
            true_token = true_token.to(device)

            logits = mlp(keys)
            log_probs = F.log_softmax(logits, dim=-1)

            ce_loss = F.nll_loss(log_probs, true_token)

            kl_loss = torch.tensor(0.0, device=device)

            for i, target_dict in enumerate(target_dicts):
                if not target_dict:
                    continue
                tokens = torch.tensor(
                    list(target_dict.keys()),
                    dtype=torch.long, device=device
                )
                probs = torch.tensor(
                    list(target_dict.values()),
                    dtype=torch.float32, device=device
                )
                log_p_knn = torch.log(probs.clamp(min=1e-30))
                log_p_mlp = log_probs[i, tokens]
                kl_loss = kl_loss + (probs * (log_p_knn - log_p_mlp)).sum()

            kl_loss = kl_loss / len(target_dicts)

            loss = alpha * kl_loss + (1 - alpha) * ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: {total_loss/len(loader)}")

    torch.save({
        "state_dict": mlp.state_dict(),
        "config": config or {},
    }, "mlp_memory.pt")
    print("MLP saved.")

    return mlp


# =========================================================
# Inference (Base)
# =========================================================

def inference(model, tokenizer, prompt,
              max_new_tokens, device):

    model.eval()

    inputs = tokenizer(prompt,
                       return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )


# =========================================================
# Inference with MLP
# =========================================================

def inference_mlp(model, tokenizer, mlp,
                  prompt, target_layer_index,
                  lambda_interp,
                  max_new_tokens,
                  device):

    model.eval()
    mlp.eval()

    inputs = tokenizer(prompt,
                       return_tensors="pt").to(device)

    input_ids = inputs["input_ids"]
    past_key_values = None

    generated = input_ids

    for _ in range(max_new_tokens):

        with torch.no_grad():
            outputs = model(
                input_ids=generated[:, -1:]
                if past_key_values else generated,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True
            )

        past_key_values = outputs.past_key_values

        lm_logits = outputs.logits[:, -1, :]
        p_lm = F.softmax(lm_logits, dim=-1)

        hidden = outputs.hidden_states[
            target_layer_index
        ][:, -1, :]

        mlp_logits = mlp(hidden)
        p_mlp = F.softmax(mlp_logits, dim=-1)

        p_final = lambda_interp * p_mlp + \
            (1 - lambda_interp) * p_lm

        next_token = torch.argmax(
            p_final,
            dim=-1,
            keepdim=True
        )

        generated = torch.cat(
            [generated, next_token],
            dim=-1
        )

    return tokenizer.decode(
        generated[0],
        skip_special_tokens=True
    )


# =========================================================
# Main
# =========================================================

def main():

    args = parse_args()

    if args.mode in ["build", "full"] and args.corpus is None:
        raise ValueError("--corpus is required for mode 'build' or 'full'")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float16
        if device == "cuda" else torch.float32,
        device_map="auto"
    )

    model.eval()

    num_layers = model.config.num_hidden_layers
    target_layer_index = int(num_layers * 0.7)

    print("Target layer:", target_layer_index)

    if args.mode in ["build", "full"]:
        build_datastore(
            model,
            tokenizer,
            args.corpus,
            target_layer_index,
            device,
            args.save_prefix,
            args.max_length
        )

    if args.mode in ["knn", "full"]:
        compute_knn_targets(
            args.save_prefix,
            args.K,
            args.tau
        )

    mlp = None

    if args.mode in ["train", "full"]:
        config = {
            "model_name": args.model_name,
            "hidden_dim": model.config.hidden_size,
            "target_layer_index": target_layer_index,
            "save_prefix": args.save_prefix,
            "max_length": args.max_length,
            "K": args.K,
            "tau": args.tau,
            "alpha": args.alpha,
            "lambda_interp": args.lambda_interp,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        }
        mlp = train_mlp(
            model,
            args.save_prefix,
            args.alpha,
            args.batch_size,
            args.epochs,
            device,
            config=config,
        )

    if args.mode in ["infer", "full"]:

        if mlp is None:
            checkpoint = torch.load("mlp_memory.pt", weights_only=False)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                config = checkpoint["config"]
                print("Loaded config:", config)
                hidden_dim = config["hidden_dim"]
                target_layer_index = config["target_layer_index"]
                state_dict = checkpoint["state_dict"]
            else:
                hidden_dim = model.config.hidden_size
                state_dict = checkpoint
            embed_weight = model.get_input_embeddings().weight.detach()
            mlp = MLPMemory(hidden_dim, embed_weight).to(device)
            mlp.load_state_dict(state_dict)

        print("\n=== Base LM ===")
        print(inference(model, tokenizer,
                        args.prompt, args.max_new_tokens, device))

        print("\n=== MLP Memory ===")
        print(inference_mlp(model, tokenizer,
                            mlp,
                            args.prompt,
                            target_layer_index,
                            args.lambda_interp,
                            args.max_new_tokens,
                            device))


if __name__ == "__main__":
    main()
