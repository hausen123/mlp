"""
Lambda interpolation survey.
Runs inference_mlp over lambda values and writes all outputs to test/lambda_survey_YYYYMMDDHHММ.txt.
"""
import hashlib
import json
import os
import sys
import glob
import argparse
import tempfile
import torch
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from transformers import AutoTokenizer, AutoModelForCausalLM
from mlp import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MAX_NEW_TOKENS,
    inference_mlp,
    inference_rag,
    build_rag_index,
    MLPMemory,
)

DEFAULT_LAMBDA_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="基準地震動の策定方法について教えてください。")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--lambdas", type=float, nargs="+", default=None)
    parser.add_argument("--rag_prefix", type=str, default=None, help="RAGインデックスのプレフィックス（省略時は --qa_path から自動生成）")
    parser.add_argument("--qa_path", type=str, default=None, help="RAGインデックス構築用 QA JSONL（--rag_prefix 省略時に自動構築）")
    parser.add_argument("--max_chunks", type=int, default=None, help="RAG構築に使用する最大chunk_id数（省略時はモデルconfigから自動取得）")
    parser.add_argument("--output", "-o", type=str, default=None, help="出力ファイルパス（省略時は自動生成）")
    args = parser.parse_args()
    lambda_values = args.lambdas if args.lambdas is not None else DEFAULT_LAMBDA_VALUES
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_NAME,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()
    if args.model_dir:
        save_dir = args.model_dir
    else:
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
    max_chunks = args.max_chunks
    if max_chunks is None:
        max_chunks = (mlp.config.training or {}).get("max_chunks")
    out_path = Path(args.output) if args.output else Path(__file__).parent / f"lambda_survey_{datetime.now().strftime('%Y%m%d%H%M')}.txt"
    lines = []
    lines.append(f"Lambda survey — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Model: {save_dir}")
    lines.append(f"Prompt: {args.prompt}")
    lines.append("=" * 60)
    rag_prefix = args.rag_prefix
    _tmp_jsonl = None
    if args.qa_path is not None:
        qa_path_for_rag = args.qa_path
        if max_chunks is not None:
            with open(args.qa_path, encoding="utf-8") as f:
                all_lines = [json.loads(l) for l in f if l.strip()]
            seen, ordered_ids = set(), []
            for item in all_lines:
                cid = item.get("chunk_id")
                if cid is not None and cid not in seen:
                    seen.add(cid)
                    ordered_ids.append(cid)
            valid_ids = set(ordered_ids[:max_chunks])
            filtered = [item for item in all_lines if item.get("chunk_id") in valid_ids]
            _tmp_fd, _tmp_jsonl = tempfile.mkstemp(suffix=".jsonl", dir="tmp")
            with os.fdopen(_tmp_fd, "w", encoding="utf-8") as f:
                for item in filtered:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            qa_path_for_rag = _tmp_jsonl
            print(f"RAG: using {len(filtered)} entries from {len(valid_ids)} chunks (max_chunks={max_chunks})")
        if rag_prefix is None:
            _key = args.qa_path + (f"_c{max_chunks}" if max_chunks else "")
            _h = hashlib.md5(_key.encode()).hexdigest()[:8]
            rag_prefix = f"tmp/rag_{_h}"
    if rag_prefix:
        index_path = rag_prefix + "_rag.index"
        if not os.path.exists(index_path):
            if args.qa_path is None:
                raise ValueError(f"RAG index not found: {index_path}. Specify --qa_path to build it.")
            print(f"RAG index not found. Building from {qa_path_for_rag} ...")
            build_rag_index(model, tokenizer, qa_path_for_rag, rag_prefix, device)
        if _tmp_jsonl and os.path.exists(_tmp_jsonl):
            os.remove(_tmp_jsonl)
        print("\n--- RAG baseline ---")
        try:
            rag_out = inference_rag(
                model, tokenizer, args.prompt, rag_prefix,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                device=device,
            )
            rag_result = rag_out[0] if isinstance(rag_out, tuple) else rag_out
            retrieved = rag_out[1] if isinstance(rag_out, tuple) else []
            print(rag_result)
            print("\n--- Retrieved QA pairs ---")
            for i, r in enumerate(retrieved, 1):
                print(f"[{i}] Q: {r['instruction']}")
                print(f"    A: {r['output'][:120]}{'...' if len(r['output']) > 120 else ''}")
            lines.append("\n[RAG baseline]")
            lines.append(rag_result)
            lines.append("\n[Retrieved QA pairs]")
            for i, r in enumerate(retrieved, 1):
                lines.append(f"[{i}] Q: {r['instruction']}")
                lines.append(f"    A: {r['output'][:200]}{'...' if len(r['output']) > 200 else ''}")
            lines.append("-" * 60)
        except Exception as e:
            print(f"RAG inference failed: {e}")
    for lam in lambda_values:
        print(f"\n--- lambda={lam} ---")
        result = inference_mlp(
            model, tokenizer, mlp,
            args.prompt, target_layer_index,
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
