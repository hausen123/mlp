# MLP Memory

kNN-LM の知識を MLP に蒸留し、推論時に LM の出力と補間するシステム。

## 概要

コーパスからデータストアを構築し、kNN 検索で得た近傍トークン分布を教師信号として MLP を訓練する。推論時は LM の出力確率と MLP の出力確率を線形補間して次トークンを選択する。

```
p_final = λ * p_MLP + (1 - λ) * p_LM
```

## セットアップ

```bash
uv sync
```

## パイプライン

### 一括実行

```bash
uv run python mlp.py --corpus corpus.txt
```

### ステップ別実行（mlp.py）

```bash
# Step 1: データストア構築（hidden states + トークン ID を保存）
uv run python mlp.py --corpus corpus.txt --mode build

# Step 2: kNN ターゲット計算（近傍トークン分布を保存）
uv run python mlp.py --corpus corpus.txt --mode knn

# Step 3: MLP 訓練
uv run python mlp.py --corpus corpus.txt --mode train

# Step 4: 推論（Base LM と MLP Memory を比較）
uv run python mlp.py --corpus corpus.txt --mode infer --prompt "任意のプロンプト"
```

### ステップ別実行（test.py）

```bash
uv run python test.py --step 1 --prompt "任意のプロンプト"  # Base LM 推論
uv run python test.py --step 2  # データストア構築
uv run python test.py --step 3  # kNN ターゲット計算
uv run python test.py --step 4  # MLP 訓練
uv run python test.py --step 5 --prompt "任意のプロンプト"  # MLP Memory 推論
```

## 引数

### mlp.py

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--corpus` | (必須) | テキストコーパスのパス |
| `--model_name` | `Qwen/Qwen2.5-0.5B` | HuggingFace モデル名 |
| `--save_prefix` | `datastore` | データストアファイルの prefix |
| `--max_length` | `256` | チャンク長（トークン数） |
| `--batch_size` | `64` | 訓練バッチサイズ |
| `--epochs` | `3` | 訓練エポック数 |
| `--K` | `64` | kNN の近傍数 |
| `--tau` | `10.0` | kNN 距離のスケールパラメータ |
| `--alpha` | `0.4` | KL 損失の重み（`α * KL + (1-α) * CE`） |
| `--lambda_interp` | `0.45` | 推論時の MLP 補間係数 |
| `--mode` | `full` | `build` / `knn` / `train` / `infer` / `full` |
| `--prompt` | `Transformerの仕組みを...` | 推論プロンプト |

### test.py

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--step` | `0`（全実行） | 実行するステップ（1〜5） |
| `--prompt` | `原子力発電所の耐震設計において` | 推論プロンプト（step 1, 5 で使用） |

## 生成ファイル

| ファイル | 内容 |
|---------|------|
| `{prefix}_keys.npy` | 隠れ状態ベクトル（float32） |
| `{prefix}_vals.npy` | 次トークン ID（int64） |
| `{prefix}_targets.npy` | kNN トークン分布（dict の配列） |
| `mlp_memory.pt` | 訓練済み MLP の重み |

## アーキテクチャ

**MLPMemory**
```
LayerNorm → Linear(d, 2d) → GELU → Linear(2d, d) → matmul(embed_weight.T)
```

- 入力: target layer の hidden state（デフォルト: 全層数の 70% 番目）
- 出力: 語彙サイズのロジット
- embed_weight は凍結

**損失関数**
```
L = α * KL(p_kNN || p_MLP) + (1 - α) * CE(p_MLP, true_token)
```

## 動作確認済み環境

- CUDA 12.2 / NVIDIA RTX 3090
- faiss-gpu-cu12 1.13.2
- Python 3.11
