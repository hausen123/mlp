import argparse
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mlp import (
    build_datastore,
    compute_knn_targets,
    train_mlp,
    inference,
    inference_mlp,
    MLPMemory,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
CORPUS_PATH = "kangaekata.txt"
SAVE_PREFIX = "test_ds"
PROMPT = "原子力発電所の耐震設計において"
MAX_NEW_TOKENS = 1024
MAX_LENGTH = 256
K = 64
TAU = 10.0
ALPHA = 0.4
LAMBDA_INTERP = 0.45
BATCH_SIZE = 64
EPOCHS = 3

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer, device

def step1(model, tokenizer, device, prompt):
    print("\n=== Step 1: Base inference ===")
    out = inference(model, tokenizer, prompt, MAX_NEW_TOKENS, device)
    print(out)

def step2(model, tokenizer, device, target_layer_index):
    print("\n=== Step 2: Datastore build ===")
    build_datastore(
        model, tokenizer, CORPUS_PATH,
        target_layer_index, device,
        SAVE_PREFIX, MAX_LENGTH,
    )

def step3():
    print("\n=== Step 3: kNN targets ===")
    compute_knn_targets(SAVE_PREFIX, K, TAU)

def step4(model, device, target_layer_index):
    print("\n=== Step 4: Train MLP ===")
    return train_mlp(
        model, SAVE_PREFIX, ALPHA, BATCH_SIZE, EPOCHS, device,
        model_name=MODEL_NAME,
        target_layer_index=target_layer_index,
        lambda_interp=LAMBDA_INTERP,
        K=K, tau=TAU, max_length=MAX_LENGTH,
    )

def step5(model, tokenizer, mlp, device, target_layer_index, prompt):
    print("\n=== Step 5: Inference with MLP ===")
    out = inference_mlp(
        model, tokenizer, mlp,
        prompt, target_layer_index,
        mlp.config.lambda_interp, MAX_NEW_TOKENS, device,
    )
    print(out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=0,
                        help="実行するステップ (1-5)。省略時は全ステップ実行")
    parser.add_argument("--prompt", type=str, default=PROMPT,
                        help="推論プロンプト (step 1, 5 で使用)")
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

if __name__ == "__main__":
    main()
