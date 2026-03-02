import argparse
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mlp import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MAX_NEW_TOKENS,
    build_datastore,
    compute_knn_targets,
    train_mlp,
    build_qa_datastore,
    train_mlp_qa,
    inference,
    inference_mlp,
    MLPMemory,
)

CORPUS_PATH = "kangaekata.txt"
SAVE_PREFIX = "test_ds"
QA_PATH = "data/202510301810_250304_kangaekata_rag_workflow.jsonl"
QA_SAVE_PREFIX = "qa_ds"
PROMPT = "原子力発電所の耐震設計において"

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_NAME,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer, device

def step1(model, tokenizer, device, prompt):
    print("\n=== Step 1: Base inference ===")
    print(inference(model, tokenizer, prompt, device=device))

def step2(model, tokenizer, device, target_layer_index):
    print("\n=== Step 2: Datastore build ===")
    build_datastore(model, tokenizer, CORPUS_PATH, target_layer_index, device, SAVE_PREFIX)

def step3():
    print("\n=== Step 3: kNN targets ===")
    compute_knn_targets(SAVE_PREFIX)

def step4(model, device, target_layer_index):
    print("\n=== Step 4: Train MLP (kNN) ===")
    return train_mlp(
        model, SAVE_PREFIX, device,
        model_name=DEFAULT_MODEL_NAME,
        target_layer_index=target_layer_index,
    )

def step5(model, tokenizer, mlp, device, target_layer_index, prompt):
    print("\n=== Step 5: Inference with MLP ===")
    print(inference_mlp(
        model, tokenizer, mlp,
        prompt, target_layer_index,
        mlp.config.lambda_interp,
        device=device,
    ))

def step6(tokenizer, qa_limit=None):
    print("\n=== Step 6: Build QA datastore ===")
    build_qa_datastore(tokenizer, QA_PATH, QA_SAVE_PREFIX, max_samples=qa_limit)

def step7(model, device, target_layer_index, epochs=None):
    print("\n=== Step 7: Train MLP (QA) ===")
    kwargs = dict(
        model_name=DEFAULT_MODEL_NAME,
        target_layer_index=target_layer_index,
    )
    if epochs is not None:
        kwargs["epochs"] = epochs
    return train_mlp_qa(model, QA_SAVE_PREFIX, device, **kwargs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=0,
                        help="実行するステップ (1-7)。省略時は全ステップ実行")
    parser.add_argument("--prompt", type=str, default=PROMPT,
                        help="推論プロンプト (step 1, 5 で使用)")
    parser.add_argument("--qa_limit", type=int, default=None,
                        help="step 6 で使用するQAサンプル数の上限 (テスト用)")
    args = parser.parse_args()
    model, tokenizer, device = load_model()
    num_layers = model.config.num_hidden_layers
    target_layer_index = int(num_layers * 0.7)
    print(f"Target layer: {target_layer_index} / {num_layers}")
    run_all = args.step == 0
    if run_all or args.step == 1:
        step1(model, tokenizer, device, args.prompt)
    if run_all or args.step == 2:
        step2(model, tokenizer, device, target_layer_index)
    if run_all or args.step == 3:
        step3()
    mlp = None
    save_dir = None
    if run_all or args.step == 4:
        mlp, save_dir = step4(model, device, target_layer_index)
    if run_all or args.step == 5:
        if mlp is None:
            model_dirs = sorted(glob.glob("model/*/"), reverse=True)
            if not model_dirs:
                raise RuntimeError("No saved model found in model/. Run step 4 first.")
            save_dir = model_dirs[0].rstrip("/")
            print(f"Loading from {save_dir}")
            embed_weight = model.get_input_embeddings().weight.detach()
            mlp = MLPMemory.from_pretrained(save_dir, embed_weight).to(device)
            target_layer_index = mlp.config.target_layer_index
        step5(model, tokenizer, mlp, device, target_layer_index, args.prompt)
    if run_all or args.step == 6:
        step6(tokenizer, args.qa_limit)
    if run_all or args.step == 7:
        mlp, save_dir = step7(model, device, target_layer_index)
        if run_all:
            step5(model, tokenizer, mlp, device, target_layer_index, args.prompt)

if __name__ == "__main__":
    main()
