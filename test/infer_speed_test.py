"""Inference speed & VRAM test: Qwen2.5-0.5B vs Qwen3-4B"""
import sys
import time
import torch
sys.path.insert(0, "/home/ryo/2026/mlp")
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT = "原子力発電所の耐震設計において、基準地震動の策定方法について説明してください。"
MAX_NEW_TOKENS = 200
WARMUP_TOKENS = 50

MODELS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "auto"),
    ("Qwen/Qwen3-4B-Instruct-2507", "auto"),
]

def measure(model_name, torch_dtype):
    print(f"\n{'='*60}")
    print(f"Model: {model_name}, dtype: {torch_dtype}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vram_before = torch.cuda.memory_allocated() / 1e9
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map="cuda"
    )
    model.eval()
    load_time = time.time() - t0
    vram_model = torch.cuda.memory_allocated() / 1e9
    print(f"Load time: {load_time:.1f}s, VRAM after load: {vram_model:.2f} GB")
    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    # warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=WARMUP_TOKENS, do_sample=False)
    torch.cuda.synchronize()
    vram_peak_warmup = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    # actual measurement
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.time() - t1
    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    tps = n_new / elapsed
    vram_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"Generated tokens: {n_new}, Time: {elapsed:.2f}s")
    print(f"Speed: {tps:.1f} tokens/sec")
    print(f"VRAM peak (generate): {vram_peak:.2f} GB")
    result_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Output (first 200 chars): {result_text[:200]}")
    del model
    torch.cuda.empty_cache()
    return tps, vram_peak

results = []
for model_name, dtype in MODELS:
    try:
        tps, vram = measure(model_name, dtype)
        results.append((model_name, tps, vram))
    except Exception as e:
        print(f"ERROR: {e}")
        results.append((model_name, 0, 0))

print(f"\n{'='*60}")
print("Summary:")
for model_name, tps, vram in results:
    print(f"  {model_name}: {tps:.1f} tok/s, VRAM={vram:.2f} GB")
