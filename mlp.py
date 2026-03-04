import os
import re
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file, load_file
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig

# =========================================================
# Defaults
# =========================================================

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MAX_LENGTH = 2048
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 20
DEFAULT_K = 64
DEFAULT_TAU = 10.0
DEFAULT_ALPHA = 0.4
DEFAULT_LAMBDA_INTERP = 0.45
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_NUM_LAYERS = 22
DEFAULT_RAG_K = 3

# =========================================================
# Argument
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="MLP Memory: kNN-LM / QA-based training and inference"
    )
    parser.add_argument("--mode", type=str, default="infer",
                        choices=["build", "knn", "train", "infer", "full",
                                 "qa-build", "qa-train", "qa-full",
                                 "rag-build", "rag-infer"],
                        help=(
                            "build/knn/train/infer/full: kNN workflow. "
                            "qa-build/qa-train/qa-full: QA workflow."
                        ))
    # model
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model_dir", type=str, default=None,
                        help="保存済み MLPMemory ディレクトリ (infer 時に必須)")
    # kNN datastore
    parser.add_argument("--corpus", type=str, default=None,
                        help="コーパステキストファイル (build/full 時に必須)")
    parser.add_argument("--save_prefix", type=str, default="datastore",
                        help="kNN datastore の保存プレフィックス")
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    # QA datastore
    parser.add_argument("--qa_path", type=str, default=None,
                        help="QA JSONL ファイル (qa-build/qa-full 時に必須)")
    parser.add_argument("--qa_prefix", type=str, default="qa_ds",
                        help="QA datastore の保存プレフィックス")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="QA サンプル数の上限 (省略時は全件)")
    # training
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lambda_interp", type=float, default=DEFAULT_LAMBDA_INTERP)
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--max_tokens_per_step", type=int, default=2048,
                        help="MLP forward/backward のトークン上限（QA学習時のOOM防止）")
    parser.add_argument("--checkpoint_every", type=int, default=3,
                        help="N エポックごとにチェックポイント上書き保存（0で無効）")
    # inference
    parser.add_argument("--prompt", type=str,
                        default="Transformerの仕組みを説明してください。")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--skip_base_lm", action="store_true",
                        help="推論時に Base LM の出力をスキップする")
    parser.add_argument("--use_final_layer", action="store_true",
                        help="70%層に加えて最終隠れ層も足し合わせて MLP に入力する")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="継続学習するモデルディレクトリ（アーキテクチャが一致しない場合はエラー）")
    parser.add_argument("-m", "--comment", type=str, default=None,
                        help="モデル保存時のコメント（train/qa-train/full/qa-full 時は必須）")
    parser.add_argument("--rag_k", type=int, default=DEFAULT_RAG_K,
                        help="RAG 検索時の top-k 件数")
    return parser.parse_args()


# =========================================================
# Corpus
# =========================================================

def split_text(input_str, max_length):
    result = []
    current_chunk = ""
    for char in input_str:
        current_chunk += char
        if char == '。':
            if len(current_chunk) >= max_length:
                result.append(current_chunk)
                current_chunk = ""
        elif len(current_chunk) >= max_length:
            last_period_index = current_chunk.rfind('。')
            if last_period_index != -1:
                result.append(current_chunk[:last_period_index + 1])
                current_chunk = current_chunk[last_period_index + 1:]
    if current_chunk:
        result.append(current_chunk)
    return result

def load_text_corpus(path, tokenizer, max_length=DEFAULT_MAX_LENGTH):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text_chunks = split_text(text, max_length)
    eos = torch.tensor([tokenizer.eos_token_id])
    chunks = []
    for tc in text_chunks:
        ids = tokenizer(tc, return_tensors="pt",
                        truncation=False)["input_ids"][0]
        ids = torch.cat([ids, eos])
        if len(ids) >= 2:
            chunks.append(ids)
    return chunks


# =========================================================
# QA Datastore Build (tokenization only, no LM pass)
# =========================================================

def build_qa_datastore(tokenizer, qa_path, save_prefix,
                       max_seq_len=2048, max_samples=None):
    """Tokenize QA pairs from JSONL and save as token sequences.
    No LM pass required. Saves:
      save_prefix_qa_plens.npy  - prompt lengths (int32)
      save_prefix_qa_ids.npy   - full token id arrays (object array)
    c_0 = chat-template prefix up to add_generation_prompt end.
    w_0 = first token of output.
    """
    sequences = []
    skipped = 0
    with open(qa_path, encoding="utf-8") as f:
        lines = f.readlines()
    if max_samples is not None:
        lines = lines[:max_samples]
    for line in tqdm(lines, desc="Tokenizing QA pairs"):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        instruction = item["instruction"]
        output_text = item["output"]
        messages = [{"role": "user", "content": instruction}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        output_ids = tokenizer(
            output_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        eos = torch.tensor([tokenizer.eos_token_id])
        output_ids = torch.cat([output_ids, eos])
        P = len(prompt_ids)
        T = len(output_ids)
        if P + T > max_seq_len:
            skipped += 1
            continue
        full_ids = torch.cat([prompt_ids, output_ids]).numpy().astype("int32")
        sequences.append((P, full_ids))
    print(f"Tokenized {len(sequences)} QA pairs, skipped {skipped}")
    np.save(
        save_prefix + "_qa_plens.npy",
        np.array([s[0] for s in sequences], dtype=np.int32)
    )
    np.save(
        save_prefix + "_qa_ids.npy",
        np.array([s[1] for s in sequences], dtype=object),
        allow_pickle=True
    )
    print(f"Saved to {save_prefix}_qa_plens.npy / _qa_ids.npy")


# =========================================================
# Datastore Build
# =========================================================

def build_datastore(model, tokenizer, text_path,
                    target_layer_index, device,
                    save_prefix, max_length=DEFAULT_MAX_LENGTH,
                    min_ctx=1, use_final_layer=False):
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
            if use_final_layer:
                hidden = hidden + outputs.hidden_states[-1]
            keys = hidden[:, min_ctx:, :].reshape(-1, hidden.size(-1)).cpu()
            vals = chunk[:, min_ctx + 1:].reshape(-1).cpu()
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

def compute_knn_targets(save_prefix, K=DEFAULT_K, tau=DEFAULT_TAU, batch_size=512):
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
# MLPMemory Config
# =========================================================

class MLPMemoryConfig(PretrainedConfig):
    model_type = "mlp_memory"
    def __init__(
        self,
        base_model_name=DEFAULT_MODEL_NAME,
        hidden_dim=896,
        num_layers=DEFAULT_NUM_LAYERS,
        target_layer_index=16,
        lambda_interp=DEFAULT_LAMBDA_INTERP,
        use_final_layer=False,
        training=None,
        comment="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.target_layer_index = target_layer_index
        self.lambda_interp = lambda_interp
        self.use_final_layer = use_final_layer
        self.training = training or {}
        self.comment = comment


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

def _validate_resume(resume_dir, hidden_dim, num_layers, target_layer_index, use_final_layer):
    """保存済みモデルのアーキテクチャが現在の設定と一致するか検証する。"""
    saved = MLPMemoryConfig.from_pretrained(resume_dir)
    errors = []
    if saved.hidden_dim != hidden_dim:
        errors.append(f"  hidden_dim:         saved={saved.hidden_dim}, current={hidden_dim}")
    if saved.num_layers != num_layers:
        errors.append(f"  num_layers:         saved={saved.num_layers}, current={num_layers}")
    if saved.target_layer_index != target_layer_index:
        errors.append(f"  target_layer_index: saved={saved.target_layer_index}, current={target_layer_index}")
    if getattr(saved, "use_final_layer", False) != use_final_layer:
        errors.append(f"  use_final_layer:    saved={getattr(saved, 'use_final_layer', False)}, current={use_final_layer}")
    if errors:
        print("Error: incompatible model config in --resume_from:")
        for e in errors:
            print(e)
        sys.exit(1)

def _make_ckpt_dir(model_name):
    name = model_name.split("/")[-1]
    slug = re.sub(r'\.', '', name.lower())
    slug = re.sub(r'[^a-z0-9]+', '-', slug).strip('-')
    return os.path.join("model", f"ckpt_{slug}")

def _make_save_dir(model_name):
    name = model_name.split("/")[-1]
    slug = re.sub(r'\.', '', name.lower())
    slug = re.sub(r'[^a-z0-9]+', '-', slug).strip('-')
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    return os.path.join("model", f"{timestamp}_{slug}")

class MLPBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
    def forward(self, x):
        return x + self.ff(self.norm(x))

class MLPMemory(nn.Module):
    def __init__(self, config: MLPMemoryConfig, embed_weight):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([
            MLPBlock(config.hidden_dim)
            for _ in range(config.num_layers)
        ])
        self.register_buffer("embed_weight", embed_weight.float())
    def forward(self, h):
        h = h.float()
        for block in self.blocks:
            h = block(h)
        logits = torch.matmul(h, self.embed_weight.T)
        return logits
    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.config.save_pretrained(save_dir)
        sd = {k: v.contiguous() for k, v in self.state_dict().items()
              if k != "embed_weight"}
        save_file(sd, os.path.join(save_dir, "model.safetensors"))
        print(f"Saved to {save_dir}")
    @classmethod
    def from_pretrained(cls, save_dir, embed_weight):
        config = MLPMemoryConfig.from_pretrained(save_dir)
        model = cls(config, embed_weight)
        state_dict = load_file(os.path.join(save_dir, "model.safetensors"))
        model.load_state_dict(state_dict, strict=False)
        return model


# =========================================================
# Training
# =========================================================

def collate_fn(batch):
    keys = torch.stack([b[0] for b in batch])
    vals = torch.stack([b[1] for b in batch])
    targets = [b[2] for b in batch]
    return keys, vals, targets

def train_mlp(model, save_prefix, device,
              model_name=None,
              target_layer_index=None,
              alpha=DEFAULT_ALPHA,
              batch_size=DEFAULT_BATCH_SIZE,
              epochs=DEFAULT_EPOCHS,
              lambda_interp=DEFAULT_LAMBDA_INTERP,
              K=DEFAULT_K,
              tau=DEFAULT_TAU,
              max_length=DEFAULT_MAX_LENGTH,
              num_layers=DEFAULT_NUM_LAYERS,
              use_final_layer=False,
              resume_from=None,
              checkpoint_every=3,
              comment=""):
    dataset = MLPMemoryDataset(save_prefix)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=collate_fn)
    hidden_dim = model.config.hidden_size
    embed_weight = model.get_input_embeddings().weight.detach()
    embed_weight.requires_grad = False
    config = MLPMemoryConfig(
        base_model_name=model_name or "",
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        target_layer_index=target_layer_index or 0,
        lambda_interp=lambda_interp,
        use_final_layer=use_final_layer,
        training={
            "K": K,
            "tau": tau,
            "alpha": alpha,
            "batch_size": batch_size,
            "epochs": epochs,
            "save_prefix": save_prefix,
            "max_length": max_length,
        },
        comment=comment,
    )
    if resume_from:
        _validate_resume(resume_from, hidden_dim, num_layers,
                         target_layer_index or 0, use_final_layer)
        mlp = MLPMemory.from_pretrained(resume_from, embed_weight).to(device)
        print(f"Resumed from {resume_from}")
    else:
        mlp = MLPMemory(config, embed_weight).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=4e-4)
    ckpt_dir = _make_ckpt_dir(model_name or "mlp-memory") if checkpoint_every > 0 else None
    if ckpt_dir:
        print(f"Checkpoint dir: {ckpt_dir}")
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
        if ckpt_dir and (epoch + 1) % checkpoint_every == 0:
            mlp.save_pretrained(ckpt_dir)
            print(f"Checkpoint saved to {ckpt_dir}")
    save_dir = _make_save_dir(model_name or "mlp-memory")
    mlp.save_pretrained(save_dir)
    return mlp, save_dir


# =========================================================
# QA Training (CE loss only, no kNN / FAISS)
# =========================================================

def train_mlp_qa(model, save_prefix, device,
                 model_name=None,
                 target_layer_index=None,
                 batch_size=DEFAULT_BATCH_SIZE,
                 epochs=DEFAULT_EPOCHS,
                 num_layers=DEFAULT_NUM_LAYERS,
                 max_tokens_per_step=2048,
                 use_final_layer=False,
                 resume_from=None,
                 checkpoint_every=3,
                 comment=""):
    """Train MLP Memory on QA pairs with online LM inference and CE loss.
    For each batch:
      1. Pad QA token sequences and run the frozen LM (teacher forcing).
      2. Extract hidden_states[target_layer_index] at output positions.
         key at position t = h_{P-1+t}, target = output_ids[t]
      3. Compute CE(MLP(keys), targets) and update only the MLP.
    No kNN / FAISS required.
    """
    prompt_lens = np.load(save_prefix + "_qa_plens.npy")
    full_ids_arr = np.load(save_prefix + "_qa_ids.npy", allow_pickle=True)
    sequences = list(zip(prompt_lens.tolist(), full_ids_arr))
    hidden_dim = model.config.hidden_size
    embed_weight = model.get_input_embeddings().weight.detach()
    embed_weight.requires_grad = False
    config = MLPMemoryConfig(
        base_model_name=model_name or "",
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        target_layer_index=target_layer_index or 0,
        use_final_layer=use_final_layer,
        comment=comment,
    )
    if resume_from:
        _validate_resume(resume_from, hidden_dim, num_layers,
                         target_layer_index or 0, use_final_layer)
        mlp = MLPMemory.from_pretrained(resume_from, embed_weight).to(device)
        print(f"Resumed from {resume_from}")
    else:
        mlp = MLPMemory(config, embed_weight).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=4e-4)
    ckpt_dir = _make_ckpt_dir((model_name or "mlp-memory") + "-qa") if checkpoint_every > 0 else None
    if ckpt_dir:
        print(f"Checkpoint dir: {ckpt_dir}")
    indices = list(range(len(sequences)))
    for epoch in range(epochs):
        random.shuffle(indices)
        total_loss = 0.0
        n_batches = 0
        model.eval()
        mlp.train()
        for batch_start in tqdm(
            range(0, len(indices), batch_size),
            desc=f"Epoch {epoch+1}"
        ):
            batch_idx = indices[batch_start:batch_start + batch_size]
            batch_seqs = [sequences[i] for i in batch_idx]
            # Sort longest-first to minimize padding waste
            batch_seqs = sorted(batch_seqs, key=lambda x: len(x[1]), reverse=True)
            max_len = len(batch_seqs[0][1])
            input_ids_list = []
            attn_mask_list = []
            for P, full_ids in batch_seqs:
                ids = torch.tensor(full_ids, dtype=torch.long)
                pad_len = max_len - len(ids)
                padded = F.pad(ids, (0, pad_len), value=0)
                mask = torch.cat([
                    torch.ones(len(ids), dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long),
                ])
                input_ids_list.append(padded)
                attn_mask_list.append(mask)
            input_ids_batch = torch.stack(input_ids_list).to(device)
            attn_mask_batch = torch.stack(attn_mask_list).to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids_batch,
                    attention_mask=attn_mask_batch,
                    output_hidden_states=True,
                )
            hidden = outputs.hidden_states[target_layer_index].float()  # [B, max_len, H]
            if use_final_layer:
                hidden = hidden + outputs.hidden_states[-1].float()
            del outputs
            torch.cuda.empty_cache()
            all_keys = []
            all_vals = []
            for i, (P, full_ids) in enumerate(batch_seqs):
                T = len(full_ids) - P
                if T < 1:
                    continue
                # h_{P-1} ... h_{P+T-2} → predict o_0 ... o_{T-1}
                keys = hidden[i, P - 1:P + T - 1, :]
                vals = torch.tensor(full_ids[P:P + T], dtype=torch.long)
                all_keys.append(keys)
                all_vals.append(vals)
            del hidden
            if not all_keys:
                continue
            keys_cat = torch.cat(all_keys, dim=0)
            vals_cat = torch.cat(all_vals, dim=0).to(device)
            # Sub-batch MLP forward/backward to cap peak VRAM
            # logits=[N, vocab] can OOM when N is large (long output seqs)
            N = len(keys_cat)
            optimizer.zero_grad()
            accum_loss = 0.0
            for t in range(0, N, max_tokens_per_step):
                k = keys_cat[t:t + max_tokens_per_step]
                v = vals_cat[t:t + max_tokens_per_step]
                sub_loss = F.cross_entropy(mlp(k), v) * (len(k) / N)
                sub_loss.backward()
                accum_loss += sub_loss.item()
            optimizer.step()
            total_loss += accum_loss
            n_batches += 1
        print(f"Epoch {epoch+1}: {total_loss / max(n_batches, 1):.4f}")
        if ckpt_dir and (epoch + 1) % checkpoint_every == 0:
            mlp.save_pretrained(ckpt_dir)
            print(f"Checkpoint saved to {ckpt_dir}")
    save_dir = _make_save_dir((model_name or "mlp-memory") + "-qa")
    mlp.save_pretrained(save_dir)
    return mlp, save_dir


# =========================================================
# RAG
# =========================================================

def build_rag_index(model, tokenizer, qa_path, rag_prefix, device, batch_size=512):
    """QA JSONL の instruction を token embedding mean-pool で埋め込み FAISS インデックスを構築する。
    保存: {rag_prefix}_rag.index, {rag_prefix}_rag_meta.npy
    """
    records = []
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    embed = model.get_input_embeddings().weight.detach().float()
    dim = embed.shape[1]
    vecs = np.zeros((len(records), dim), dtype=np.float32)
    for i in tqdm(range(0, len(records), batch_size), desc="Building RAG index"):
        batch = records[i:i + batch_size]
        for j, r in enumerate(batch):
            ids = tokenizer.encode(
                r["instruction"], max_length=128, truncation=True, add_special_tokens=False
            )
            if not ids:
                continue
            ids_t = torch.tensor(ids, device=device)
            vecs[i + j] = embed[ids_t].mean(0).cpu().numpy()
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, rag_prefix + "_rag.index")
    np.save(rag_prefix + "_rag_meta.npy", np.array(records, dtype=object))
    print(f"RAG index built: {len(records)} QA pairs → {rag_prefix}_rag.index")

def inference_rag(model, tokenizer, query, rag_prefix,
                  k=DEFAULT_RAG_K,
                  max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                  device="cpu"):
    """RAG 推論: top-k QA ペアを検索してシステムプロンプトに付与し生成する。"""
    index = faiss.read_index(rag_prefix + "_rag.index")
    records = np.load(rag_prefix + "_rag_meta.npy", allow_pickle=True)
    embed = model.get_input_embeddings().weight.detach().float()
    ids = tokenizer.encode(query, max_length=128, truncation=True, add_special_tokens=False)
    ids_t = torch.tensor(ids, device=device)
    q_vec = embed[ids_t].mean(0).cpu().numpy().reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(q_vec)
    _, I = index.search(q_vec, k)
    retrieved = [records[i] for i in I[0]]
    context = "\n\n".join(
        f"Q: {r['instruction']}\nA: {r['output']}" for r in retrieved
    )
    messages = [
        {"role": "system", "content": f"以下の参考情報を踏まえて回答してください。\n\n{context}"},
        {"role": "user", "content": query},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.1,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True), retrieved

# =========================================================
# Inference (Base)
# =========================================================

def inference(model, tokenizer, prompt,
              max_new_tokens=DEFAULT_MAX_NEW_TOKENS, device="cpu"):
    model.eval()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.1,
        )
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(
        output[0][prompt_len:],
        skip_special_tokens=True
    )


# =========================================================
# Inference with MLP
# =========================================================

def inference_mlp(model, tokenizer, mlp,
                  prompt, target_layer_index,
                  lambda_interp=DEFAULT_LAMBDA_INTERP,
                  max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                  device="cpu",
                  repetition_penalty=1.1,
                  temperature=0.7,
                  top_p=0.8,
                  top_k=20,
                  use_final_layer=False):
    model.eval()
    mlp.eval()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
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
        h = outputs.hidden_states[target_layer_index][:, -1, :]
        if use_final_layer:
            h = h + outputs.hidden_states[-1][:, -1, :]
        mlp_logits = mlp(h)
        if repetition_penalty != 1.0:
            for token_id in set(generated[0].tolist()):
                lm_logits[0, token_id] = (
                    lm_logits[0, token_id] / repetition_penalty
                    if lm_logits[0, token_id] > 0
                    else lm_logits[0, token_id] * repetition_penalty
                )
                mlp_logits[0, token_id] = (
                    mlp_logits[0, token_id] / repetition_penalty
                    if mlp_logits[0, token_id] > 0
                    else mlp_logits[0, token_id] * repetition_penalty
                )
        p_lm = F.softmax(lm_logits, dim=-1)
        p_mlp = F.softmax(mlp_logits, dim=-1)
        p_final = lambda_interp * p_mlp + (1 - lambda_interp) * p_lm
        logits = torch.log(p_final.clamp(min=1e-10)) / temperature
        if top_k > 0:
            top_k_vals = torch.topk(logits, top_k)[0]
            logits = logits.masked_fill(logits < top_k_vals[:, -1:], float('-inf'))
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = (cum_probs - F.softmax(sorted_logits, dim=-1)) >= top_p
            sorted_logits[remove] = float('-inf')
            logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    prompt_len = input_ids.shape[1]
    return tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)


# =========================================================
# Main
# =========================================================

def main():
    args = parse_args()
    if args.mode in ["build", "full"] and args.corpus is None:
        raise ValueError("--corpus is required for mode 'build' or 'full'")
    if args.mode in ["qa-build", "qa-full"] and args.qa_path is None:
        raise ValueError("--qa_path is required for mode 'qa-build' or 'qa-full'")
    if args.mode in ["train", "full", "qa-train", "qa-full"] and not args.comment:
        raise ValueError("-m/--comment is required for training modes")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    # qa-build はモデル不要
    need_model = args.mode not in ("qa-build",)
    model = None
    target_layer_index = None
    if need_model:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
        )
        model.eval()
        num_layers = model.config.num_hidden_layers
        target_layer_index = int(num_layers * 0.7)
        print(f"Target layer: {target_layer_index} / {num_layers}")
    # --- kNN workflow ---
    if args.mode in ["build", "full"]:
        build_datastore(
            model, tokenizer, args.corpus,
            target_layer_index, device,
            args.save_prefix, args.max_length,
            use_final_layer=args.use_final_layer,
        )
    if args.mode in ["knn", "full"]:
        compute_knn_targets(args.save_prefix, args.K, args.tau)
    mlp = None
    save_dir = args.model_dir
    if args.mode in ["train", "full"]:
        mlp, save_dir = train_mlp(
            model, args.save_prefix, device,
            model_name=args.model_name,
            target_layer_index=target_layer_index,
            alpha=args.alpha,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lambda_interp=args.lambda_interp,
            K=args.K,
            tau=args.tau,
            max_length=args.max_length,
            num_layers=args.num_layers,
            use_final_layer=args.use_final_layer,
            resume_from=args.resume_from,
            checkpoint_every=args.checkpoint_every,
            comment=args.comment,
        )
    # --- QA workflow ---
    if args.mode in ["qa-build", "qa-full"]:
        build_qa_datastore(
            tokenizer, args.qa_path, args.qa_prefix,
            max_samples=args.max_samples,
        )
    if args.mode in ["qa-train", "qa-full"]:
        mlp, save_dir = train_mlp_qa(
            model, args.qa_prefix, device,
            model_name=args.model_name,
            target_layer_index=target_layer_index,
            batch_size=args.batch_size,
            epochs=args.epochs,
            num_layers=args.num_layers,
            max_tokens_per_step=args.max_tokens_per_step,
            use_final_layer=args.use_final_layer,
            resume_from=args.resume_from,
            checkpoint_every=args.checkpoint_every,
            comment=args.comment,
        )
    # --- RAG workflow ---
    if args.mode == "rag-build":
        if args.qa_path is None:
            raise ValueError("--qa_path is required for mode 'rag-build'")
        build_rag_index(model, tokenizer, args.qa_path, args.qa_prefix, device)
    if args.mode == "rag-infer":
        print("\n=== RAG ===")
        result, retrieved = inference_rag(
            model, tokenizer, args.prompt, args.qa_prefix,
            k=args.rag_k,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        print(result)
        print("\n--- Retrieved QA pairs ---")
        for i, r in enumerate(retrieved, 1):
            print(f"[{i}] Q: {r['instruction']}")
            print(f"    A: {r['output'][:120]}{'...' if len(r['output']) > 120 else ''}")
    # --- inference ---
    if args.mode in ["infer", "full", "qa-full"]:
        if mlp is None:
            if save_dir is None:
                raise ValueError(
                    "--model_dir is required for infer mode without prior training"
                )
            embed_weight = model.get_input_embeddings().weight.detach()
            mlp = MLPMemory.from_pretrained(save_dir, embed_weight).to(device)
            target_layer_index = mlp.config.target_layer_index
            lambda_interp = mlp.config.lambda_interp
        else:
            lambda_interp = args.lambda_interp
        if not args.skip_base_lm:
            print("\n=== Base LM ===")
            print(inference(model, tokenizer,
                            args.prompt, args.max_new_tokens, device))
        use_final_layer = getattr(mlp.config, "use_final_layer", False)
        print("\n=== MLP Memory ===")
        print(inference_mlp(model, tokenizer,
                            mlp, args.prompt,
                            target_layer_index, lambda_interp,
                            args.max_new_tokens, device,
                            use_final_layer=use_final_layer))


if __name__ == "__main__":
    main()
