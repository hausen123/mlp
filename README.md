# MLP Memory

kNN-LM の知識を MLP に蒸留、または QA ペアを教師信号として MLP を訓練し、推論時に LM の出力と補間するシステム。

## 概要

```
p_final = λ * p_MLP + (1 - λ) * p_LM
```

2つの学習方式を実装し比較できる。

| 方式 | 教師信号 | 特徴 |
|------|---------|------|
| **kNN** | コーパスから構築した kNN トークン分布 | FAISS 索引が必要 |
| **QA** | QA ペアの出力トークン（CE loss） | FAISS 不要、JSONL から直接学習 |

## セットアップ

```bash
uv sync
```

## 使い方

### 推論

```bash
uv run python mlp.py --mode infer \
  --model_dir model/YYYYMMDDHHMM_qwen25-05b-instruct-qa \
  --prompt "基準地震動の策定方法について教えてください。"

# Base LM の出力をスキップして MLP Memory のみ出力
uv run python mlp.py --mode infer \
  --model_dir model/YYYYMMDDHHMM_qwen25-05b-instruct-qa \
  --prompt "基準地震動の策定方法について教えてください。" \
  --skip_base_lm
```

### QA 学習

```bash
# Step 1: QA JSONL をトークナイズして保存（モデルロード不要）
uv run python mlp.py --mode qa-build \
  --qa_path data/qa.jsonl \
  --qa_prefix qa_ds \
  --max_samples 5000   # 省略時は全件

# Step 2: MLP を QA で学習
uv run python mlp.py --mode qa-train \
  --qa_prefix qa_ds \
  --epochs 50

# 一括実行（qa-build → qa-train → infer）
uv run python mlp.py --mode qa-full \
  --qa_path data/qa.jsonl \
  --prompt "基準地震動の策定方法について教えてください。"
```

### 継続学習

```bash
# 保存済みモデルから続きを学習して新しい model/ に保存
uv run python mlp.py --mode qa-train \
  --qa_prefix qa_ds_full \
  --resume_from model/YYYYMMDDHHMM_qwen25-05b-instruct-qa \
  --epochs 10 \
  -m "continued training: +10 epochs"
```

アーキテクチャ（`num_layers` / `use_final_layer` / `target_layer_index`）が一致しない場合はエラーを表示して終了します。

### kNN 学習

```bash
# Step 1: データストア構築
uv run python mlp.py --mode build --corpus corpus.txt

# Step 2: kNN ターゲット計算
uv run python mlp.py --mode knn

# Step 3: MLP 訓練
uv run python mlp.py --mode train

# 一括実行（build → knn → train → infer）
uv run python mlp.py --mode full --corpus corpus.txt \
  --prompt "基準地震動の策定方法について教えてください。"
```

## 引数

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--mode` | `infer` | `build`/`knn`/`train`/`infer`/`full` または `qa-build`/`qa-train`/`qa-full` |
| `--model_name` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace モデル名 |
| `--model_dir` | `None` | 保存済み MLPMemory ディレクトリ（`infer` 時に必須） |
| `--corpus` | `None` | テキストコーパス（`build`/`full` 時に必須） |
| `--save_prefix` | `datastore` | kNN datastore のプレフィックス |
| `--qa_path` | `None` | QA JSONL ファイル（`qa-build`/`qa-full` 時に必須） |
| `--qa_prefix` | `qa_ds` | QA datastore のプレフィックス |
| `--max_samples` | `None` | QA サンプル数の上限（省略時は全件） |
| `--epochs` | `20` | 訓練エポック数 |
| `--batch_size` | `64` | 訓練バッチサイズ |
| `--num_layers` | `22` | MLP の残差ブロック数 |
| `--lambda_interp` | `0.45` | 推論時の MLP 補間係数 |
| `--max_new_tokens` | `1024` | 推論時の最大生成トークン数 |
| `--skip_base_lm` | `False` | 推論時に Base LM の出力をスキップ |
| `--resume_from` | `None` | 継続学習するモデルディレクトリ |
| `--use_final_layer` | `False` | 70%層 + 最終隠れ層を MLP 入力に加算 |
| `--prompt` | `Transformerの仕組みを...` | 推論プロンプト |
| `--K` | `64` | kNN の近傍数 |
| `--tau` | `10.0` | kNN 距離スケール |
| `--alpha` | `0.4` | KL 損失の重み（kNN 方式のみ） |
| `--max_length` | `2048` | コーパスチャンク長（文字数） |

## アーキテクチャ

**MLPMemory**: M 個の Pre-LN 残差ブロック
```
h = hidden_state  (target layer, float32)
for _ in range(M):
    h = h + FFN(LayerNorm(h))   # FFN: Linear(d,4d) → GELU → Linear(4d,d)
logits = h @ embed_weight.T
```

- target layer: `int(num_hidden_layers * 0.7)` = layer 16（24層モデル）
- `--use_final_layer` 指定時: `h = h_layer16 + h_final`
- embed_weight は凍結

**QA 学習の入出力**
```
入力: hidden_state[P-1 : P+T-1]  (出力トークン位置の前の hidden state)
教師: output_ids[0 : T]           (出力トークン列)
損失: CE(logits, output_ids)
```
P = prompt 長, T = output 長

## 生成ファイル

| ファイル | 内容 |
|---------|------|
| `{prefix}_keys.npy` | 隠れ状態ベクトル（kNN 方式） |
| `{prefix}_vals.npy` | 次トークン ID（kNN 方式） |
| `{prefix}_targets.npy` | kNN トークン分布（kNN 方式） |
| `{qa_prefix}_qa_plens.npy` | プロンプト長（QA 方式） |
| `{qa_prefix}_qa_ids.npy` | トークン ID 配列（QA 方式） |
| `model/YYYYMMDDHHMM_*/` | 訓練済み MLPMemory |

## 動作確認済み環境

- CUDA 12.2 / NVIDIA RTX 3090
- faiss-gpu-cu12
- Python 3.11
