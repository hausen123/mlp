# MLP Memory

kNN-LM の知識を MLP に蒸留、または QA ペアを教師信号として MLP を訓練し、推論時に LM の出力と補間するシステム。

## 概要

```
p_final = λ * p_MLP + (1 - λ) * p_LM
```

複数の学習方式を実装し比較できる。

| 方式 | 教師信号 | 特徴 |
|------|---------|------|
| **kNN** | コーパスから構築した kNN トークン分布 | FAISS 索引が必要 |
| **QA** | QA ペアの出力トークン（CE loss） | FAISS 不要、JSONL から直接学習 |
| **QA-kNN** | QA コーパスから構築した kNN 分布 | QA + FAISS |
| **RAG** | kNN ベースのベクトル検索 | クエリに応じた文書検索 |

## セットアップ

```bash
uv sync
```

外部サービス（QA データ生成に必要）:
- Ollama (`qwen3:14b`) — `http://localhost:11434`
- E5 embedding API — `http://kawarasaki02.info/embedding/e5`

`.env` に設定:
```
OLLAMA_URL=http://localhost:11434
QWEN_MODEL=qwen3:14b
API_BASE_URL=http://kawarasaki02.info
```

## QA データ生成

PDF からテキストを抽出して QA JSONL を生成するまでの一連のフロー。

### 1. PDF → テキスト変換

```bash
python test/pdf_to_text.py <URL> [--out data/output.txt]
```

### 2. テキスト → QA JSONL（RAG ワークフロー）

```bash
# --mode facts: チャンク → fact 抽出 → 質問生成 → RAG 回答
python augument.py data/text.txt --mode facts --max-chunks 30 -o data/qa.jsonl

# --mode active: チャンク → ストラテジー → active reading QA
python augument.py data/text.txt --mode active --max-chunks 30 -o data/qa.jsonl
```

| 引数 | 説明 |
|------|------|
| `filepath` | 入力テキストファイル |
| `--mode` | `facts`（RAG ワークフロー）または `active`（active reading） |
| `--max-chunks` | 処理チャンク数の上限（ランダムサンプリング） |
| `--output`, `-o` | 出力 JSONL パス（省略時は自動生成） |
| `--max-tokens`, `-t` | LLM レスポンスの最大トークン数（デフォルト: 2048） |

## MLP 学習・推論

### 推論

```bash
uv run python mlp.py --mode infer \
  --model_dir model/YYYYMMDDHHMM_qwen25-05b-instruct-qa \
  --prompt "基準地震動の策定方法について教えてください。"

# Base LM の出力をスキップして MLP Memory のみ出力
uv run python mlp.py --mode infer \
  --model_dir model/YYYYMMDDHHMM_qwen25-05b-instruct-qa \
  --skip_base_lm
```

### QA 学習

```bash
# Step 1: QA JSONL をトークナイズして保存（モデルロード不要）
uv run python mlp.py --mode qa-build \
  --qa_path data/qa.jsonl \
  --qa_prefix tmp/qa_ds

# Step 2: MLP を QA で学習
uv run python mlp.py --mode qa-train \
  --qa_prefix tmp/qa_ds \
  --epochs 20 \
  -m "コメント（必須）"

# 一括実行（qa-build → qa-train → infer）
uv run python mlp.py --mode qa-full \
  --qa_path data/qa.jsonl \
  --prompt "基準地震動の策定方法について教えてください。" \
  -m "コメント（必須）"
```

### 継続学習

```bash
uv run python mlp.py --mode qa-train \
  --qa_prefix tmp/qa_ds_full \
  --resume_from model/YYYYMMDDHHMM_qwen25-05b-instruct-qa \
  --epochs 10 \
  -m "continued training: +10 epochs"
```

アーキテクチャ（`num_layers` / `use_final_layer` / `target_layer_index`）が一致しない場合はエラーを表示して終了。

### QA-kNN 学習

```bash
# 一括実行（qa-knn-build → knn → train → infer）
uv run python mlp.py --mode qa-knn-full \
  --qa_path data/qa.jsonl \
  --qa_prefix tmp/qa_knn_ds \
  -m "コメント（必須）"
```

### kNN 学習（コーパス方式）

```bash
uv run python mlp.py --mode build --corpus data/corpus.txt
uv run python mlp.py --mode knn
uv run python mlp.py --mode train -m "コメント（必須）"

# 一括実行
uv run python mlp.py --mode full --corpus data/corpus.txt \
  --prompt "基準地震動の策定方法について教えてください。" \
  -m "コメント（必須）"
```

### RAG（ベクトル検索）

```bash
# インデックス構築
uv run python mlp.py --mode rag-build --qa_path data/qa.jsonl

# 検索推論
uv run python mlp.py --mode rag-infer \
  --prompt "基準地震動の策定方法について教えてください。"
```

### Lambda サーベイ

```bash
# λ=0.0〜1.0 で出力を比較
python test/lambda_survey.py \
  --model_dir model/YYYYMMDDHHMM_qwen25-05b-instruct-qa \
  --prompt "基準地震動の策定方法について教えてください。"

# 任意の λ 値を指定
python test/lambda_survey.py \
  --model_dir model/YYYYMMDDHHMM_... \
  --lambdas 0.5 0.6 0.7 0.8
```

## 引数一覧（mlp.py）

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--mode` | `infer` | 動作モード（下記参照） |
| `--model_name` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace モデル名 |
| `--model_dir` | `None` | 保存済み MLPMemory ディレクトリ（`infer` 時に必須） |
| `--corpus` | `None` | テキストコーパス（`build`/`full` 時に必須） |
| `--save_prefix` | `tmp/datastore` | kNN datastore のプレフィックス |
| `--qa_path` | `None` | QA JSONL ファイル |
| `--qa_prefix` | `tmp/qa_ds` | QA datastore のプレフィックス |
| `--max_samples` | `None` | QA サンプル数の上限（省略時は全件） |
| `--epochs` | `20` | 訓練エポック数 |
| `--batch_size` | `64` | 訓練バッチサイズ |
| `--num_layers` | `22` | MLP の残差ブロック数 |
| `--lambda_interp` | `0.45` | 推論時の MLP 補間係数 |
| `--max_new_tokens` | `1024` | 推論時の最大生成トークン数 |
| `--skip_base_lm` | `False` | 推論時に Base LM の出力をスキップ |
| `--resume_from` | `None` | 継続学習するモデルディレクトリ |
| `--use_final_layer` | `False` | 70%層 + 最終隠れ層を MLP 入力に加算 |
| `--checkpoint_every` | `3` | N エポックごとにチェックポイント保存 |
| `-m`, `--comment` | `None` | 学習コメント（学習モードで必須） |
| `--K` | `64` | kNN の近傍数 |
| `--tau` | `1.0` | kNN 距離スケール |
| `--alpha` | `0.4` | KL 損失の重み（kNN 方式のみ） |
| `--max_length` | `2048` | コーパスチャンク長（文字数） |
| `--rag_k` | `3` | RAG 検索の近傍数 |

**モード一覧**

| モード | 説明 |
|--------|------|
| `build` | コーパスからデータストア構築 |
| `knn` | kNN ターゲット分布を計算 |
| `train` | kNN 方式で MLP 訓練 |
| `infer` | 推論 |
| `full` | build → knn → train → infer |
| `qa-build` | QA JSONL をトークナイズ |
| `qa-train` | QA 方式で MLP 訓練 |
| `qa-full` | qa-build → qa-train → infer |
| `qa-knn-build` | QA コーパスからデータストア構築 |
| `qa-knn-full` | qa-knn-build → knn → train → infer |
| `rag-build` | RAG 用ベクトルインデックス構築 |
| `rag-infer` | RAG ベクトル検索で推論 |

## アーキテクチャ

**MLPMemory**: M 個の Pre-LN 残差ブロック
```
h = hidden_state  (target layer, float32)
for _ in range(M):
    h = h + FFN(LayerNorm(h))   # FFN: Linear(d,2d) → GELU → Linear(2d,d)
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

## プロジェクト構造

```
mlp.py              # MLP Memory コア実装
augument.py         # QA データ生成（RAG ワークフロー）
src/
  config/settings.py    # 環境変数設定（.env から読込）
  core/llm_client.py    # Ollama API クライアント
  core/vector_db.py     # FAISS + SQLite3 ベクトル DB
  utils/processors.py   # テキスト処理ユーティリティ
test/
  lambda_survey.py      # Lambda サーベイスクリプト
  pdf_to_text.py        # PDF → テキスト変換
  test.py               # ステップ実行テスト
data/               # 入力テキスト・JSONL
tmp/                # 中間ファイル（gitignore）
model/              # 訓練済みモデル（gitignore）
```

## 生成ファイル

| ファイル | 内容 |
|---------|------|
| `tmp/{prefix}_keys.npy` | 隠れ状態ベクトル（kNN 方式） |
| `tmp/{prefix}_vals.npy` | 次トークン ID（kNN 方式） |
| `tmp/{prefix}_targets.npy` | kNN トークン分布（kNN 方式） |
| `tmp/{prefix}_qa_plens.npy` | プロンプト長（QA 方式） |
| `tmp/{prefix}_qa_ids.npy` | トークン ID 配列（QA 方式） |
| `data/text_vectors.db` | RAG 用 SQLite3 DB |
| `data/text_vectors.faiss` | RAG 用 FAISS インデックス |
| `model/YYYYMMDDHHMM_*/` | 訓練済み MLPMemory |

## 動作確認済み環境

- CUDA 12.2 / NVIDIA RTX 3090
- faiss-gpu-cu12
- Python 3.11
