"""
Lambda interpolation survey.
Runs inference_mlp over lambda values [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
and writes all outputs to test/lambda_survey_YYYYMMDDHHММ.txt.
"""
import sys
import glob
import torch
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from transformers import AutoTokenizer, AutoModelForCausalLM
from mlp import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MAX_NEW_TOKENS,
    inference_mlp,
    MLPMemory,
)

PROMPT = "基準地震動の策定方法について教えてください。"
LAMBDA_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_NAME,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()
    model_dirs = sorted(glob.glob("model/*/"), reverse=True)
    if not model_dirs:
        raise RuntimeError("No saved model found in model/. Run step 4 first.")
    save_dir = model_dirs[0].rstrip("/")
    print(f"Loading MLP from {save_dir}")
    embed_weight = model.get_input_embeddings().weight.detach()
    mlp = MLPMemory.from_pretrained(save_dir, embed_weight).to(device)
    target_layer_index = mlp.config.target_layer_index
    use_final_layer = getattr(mlp.config, "use_final_layer", False)
    print(f"Target layer: {target_layer_index}, use_final_layer: {use_final_layer}")
    out_path = Path(__file__).parent / f"lambda_survey_{datetime.now().strftime('%Y%m%d%H%M')}.txt"
    lines = []
    lines.append(f"Lambda survey — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Model: {save_dir}")
    lines.append(f"Prompt: {PROMPT}")
    lines.append("=" * 60)
    for lam in LAMBDA_VALUES:
        print(f"\n--- lambda={lam} ---")
        result = inference_mlp(
            model, tokenizer, mlp,
            PROMPT, target_layer_index,
            lambda_interp=lam,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            device=device,
            use_final_layer=use_final_layer,
        )
        print(result)
        lines.append(f"\n[lambda={lam}]")
        lines.append(result)
        lines.append("-" * 60)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
