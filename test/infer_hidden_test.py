"""Inference speed test: plain generate vs token-by-token vs hidden-state extraction"""
import sys
import time
import torch
import torch.nn.functional as F
sys.path.insert(0, "/home/ryo/2026/mlp")
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT = "原子力発電所の耐震設計において、基準地震動の策定方法について説明してください。"
MAX_NEW_TOKENS = 200
WARMUP_TOKENS = 30

MODELS = [
    {"name": "Qwen/Qwen2.5-0.5B-Instruct", "target_layer_index": 16},
    {"name": "Qwen/Qwen3-4B-Instruct-2507", "target_layer_index": 25},
]

def token_by_token(model, tokenizer, inputs, max_new_tokens, output_hidden_states=False):
    generated = inputs["input_ids"]
    past_key_values = None
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=generated[:, -1:] if past_key_values else generated,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=output_hidden_states,
            )
        past_key_values = outputs.past_key_values
        probs = F.softmax(outputs.logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return generated

def run_test(cfg):
    model_name = cfg["name"]
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="cuda"
    )
    model.eval()
    vram_model = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM (model): {vram_model:.2f} GB")
    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    # warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=WARMUP_TOKENS, do_sample=False)
    torch.cuda.synchronize()
    def measure(label, fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        alloc_before = torch.cuda.memory_allocated()
        t = time.time()
        fn()
        torch.cuda.synchronize()
        elapsed = time.time() - t
        alloc_after = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        delta = peak - alloc_before
        print(f"[{label:<22}]  {MAX_NEW_TOKENS/elapsed:5.1f} tok/s"
              f"  base={alloc_before/1e9:.2f}GB"
              f"  peak={peak/1e9:.2f}GB"
              f"  delta={delta/1e6:+.0f}MB")
    # ---- 1) plain model.generate ----
    def f1():
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    measure("plain generate", f1)
    # ---- 2) token-by-token, no hidden states ----
    measure("token-by-token", lambda: token_by_token(
        model, tokenizer, inputs, MAX_NEW_TOKENS, output_hidden_states=False))
    # ---- 3) token-by-token + output_hidden_states=True ----
    measure("hidden states=True", lambda: token_by_token(
        model, tokenizer, inputs, MAX_NEW_TOKENS, output_hidden_states=True))
    del model
    torch.cuda.empty_cache()

for cfg in MODELS:
    try:
        run_test(cfg)
    except Exception as e:
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()
