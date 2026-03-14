"""VRAM measurement: output_hidden_states=True vs forward hook (Qwen2.5-0.5B-Instruct)"""
import sys, json
import torch
import torch.nn.functional as F
sys.path.insert(0, "/home/ryo/2026/mlp")
from transformers import AutoTokenizer, AutoModelForCausalLM
from mlp import MLPMemory, MLPMemoryConfig, _extract_hidden_hook

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MLP_CKPT   = "model/202603131624_qwen25-05b-instruct-qa"
BATCH_SIZE = 64
SEQ_LEN    = 512   # padded sequence length (simulate)
MAX_TOKENS_PER_STEP = 2048
TARGET_LAYER_INDEX  = 16
DEVICE = "cuda"

def gb(n): return n / 1e9
def snap(label):
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated()
    p = torch.cuda.max_memory_allocated()
    print(f"  [{label:<40}] alloc={gb(a):.3f}GB  peak={gb(p):.3f}GB")
    return a, p

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
print("=== VRAM measurement: Qwen2.5-0.5B qa-train ===\n")

# 1. Load base model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype="auto", device_map=DEVICE
)
model.eval()
torch.cuda.reset_peak_memory_stats()
snap("1. base model loaded")

# 2. Load MLP
with open(f"{MLP_CKPT}/config.json") as f:
    cfg = json.load(f)
embed_weight = model.get_input_embeddings().weight.detach()
embed_weight.requires_grad = False
config = MLPMemoryConfig(
    base_model_name=MODEL_NAME,
    hidden_dim=cfg["hidden_dim"],
    num_layers=cfg["num_layers"],
    target_layer_index=cfg["target_layer_index"],
    use_final_layer=cfg["use_final_layer"],
    training={},
)
mlp = MLPMemory(config, embed_weight).to(DEVICE)
optimizer = torch.optim.AdamW(mlp.parameters(), lr=4e-4)
torch.cuda.reset_peak_memory_stats()
snap("2. MLP + optimizer loaded")

# 3. Simulate batch: bs=64, seq_len=512
input_ids  = torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long, device=DEVICE)
attn_mask  = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long, device=DEVICE)

print("\n--- [旧: output_hidden_states=True] ---")
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        output_hidden_states=True,
        logits_to_keep=1,
    )
snap("3a. base fwd hs=True")
print(f"     hidden_states: {len(outputs.hidden_states)} tensors × {list(outputs.hidden_states[0].shape)}")
hs_bytes = sum(h.numel() * h.element_size() for h in outputs.hidden_states)
print(f"     hidden_states total: {hs_bytes/1e9:.3f}GB")
torch.cuda.reset_peak_memory_stats()
hidden_old = outputs.hidden_states[TARGET_LAYER_INDEX].float()
hidden_old = hidden_old + outputs.hidden_states[-1].float()
del outputs
torch.cuda.empty_cache()
snap("3b. after del outputs (hs=True)")

print("\n--- [新: forward hook] ---")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
hidden_new = _extract_hidden_hook(model, input_ids, attn_mask, TARGET_LAYER_INDEX, use_final_layer=True)
snap("4a. hook forward (2層のみ保持)")
print(f"     hidden shape: {list(hidden_new.shape)}  ({hidden_new.numel()*4/1e6:.1f}MB fp32)")

# verify outputs match
diff = (hidden_old - hidden_new).abs().max().item()
print(f"     max diff vs hs=True: {diff:.2e}  {'✓ 一致' if diff < 1e-4 else '✗ 不一致'}")
del hidden_old
hidden = hidden_new

# simulate keys extraction (output positions only, ~half seq)
OUT_LEN = SEQ_LEN // 2
keys_cat = hidden[:, :OUT_LEN, :].reshape(-1, cfg["hidden_dim"]).detach()
vals_cat = torch.zeros(BATCH_SIZE * OUT_LEN, dtype=torch.long, device=DEVICE)
del hidden
N = len(keys_cat)
print(f"     N tokens total: {N}")
torch.cuda.reset_peak_memory_stats()
snap("5. after keys extraction (pre-MLP)")

# MLP forward + backward (chunked)
optimizer.zero_grad()
mlp.train()
for t in range(0, N, MAX_TOKENS_PER_STEP):
    k = keys_cat[t:t + MAX_TOKENS_PER_STEP]
    v = vals_cat[t:t + MAX_TOKENS_PER_STEP]
    logits = mlp(k)
    if t == 0:
        snap(f"6a. MLP forward chunk0 logits {list(logits.shape)}")
        print(f"      logits size: {logits.numel()*4/1e6:.1f}MB fp32")
    sub_loss = F.cross_entropy(logits, v) * (len(k) / N)
    sub_loss.backward()
    if t == 0:
        snap("6b. after backward chunk0")

optimizer.step()
snap("7. after optimizer.step()")
