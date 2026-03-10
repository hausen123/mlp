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

なお、比較用のベースラインとしてRAG回答作成機能も実装済み。

## セットアップ

```bash
pipenv install
```

外部サービス（QA データ生成に必要）:
- E5 embedding API — `http://kawarasaki02.info/embedding/e5`（RAG インデックス構築）
- Ollama（ローカル）— QA 生成に使用する LLM
- Gemini API — `--mode augment` のテキスト拡張に使用

`.env` に設定:
```
OLLAMA_URL=http://localhost:11434
QWEN_MODEL=qwen3:14b
E5_EMBEDDING_URL=http://kawarasaki02.info/embedding/e5
GEMINI_API_KEY=<your_api_key>
GEMINI_MODEL=gemini-2.0-flash
```
## QA データ生成

PDF からテキストを抽出して QA JSONL を生成するまでの一連のフロー。

### 1. PDF → テキスト変換

```bash
python test/pdf_to_text.py <URL> [--out data/output.txt]
```

### 2. テキスト拡張（オプション）

QA 生成前にテキストのバリエーションを増やす。Gemini を使用。

```bash
pipenv run python augment.py data/text.txt --mode augment -o data/text_augmented.txt
```

| 引数 | 説明 |
|------|------|
| `filepath` | 入力テキストファイル |
| `--output`, `-o` | 出力テキストファイルパス（省略時は `data/YYYYMMDDHHMM_<name>_augmented.txt`） |
| `--max-chunks` | 処理チャンク数の上限 |
| `--max-tokens`, `-t` | Gemini レスポンスの最大トークン数（デフォルト: 2048） |

各チャンクに対して `permute_sentence`（文順序入れ替え × 3）× `multiply_sentence`（言い換え × 5）を適用し、最大 4 × 6 = 24 バリアントを生成。

### 3. テキスト → QA JSONL（RAG ワークフロー）

```bash
# --mode facts（推奨）: チャンク → fact 抽出 → 質問生成 → RAG 回答
pipenv run python augment.py data/text.txt --mode facts --max-chunks 100

# --mode active: チャンク → 学習戦略 → active reading QA（MLP 学習には不適）
pipenv run python augment.py data/text.txt --mode active --max-chunks 30
```

| 引数 | 説明 |
|------|------|
| `filepath` | 入力テキストファイル |
| `--mode` | `facts`（RAG ワークフロー・推奨）、`active`（active reading）、`augment`（テキスト拡張） |
| `--max-chunks` | 処理チャンク数の上限（デフォルト: 先頭から順番） |
| `--random` | チャンクをランダムサンプリング（省略時は先頭から） |
| `--output`, `-o` | 出力ファイルパス（省略時は自動生成） |
| `--max-tokens`, `-t` | LLM レスポンスの最大トークン数（デフォルト: 2048） |

> **注意**: `--mode active` は学習戦略プロンプトが instruction に混入するため、MLP Memory 学習データには使用しないこと。

## MLP 学習・推論

### 推論

```bash
pipenv run python mlp.py --mode infer \
  --model_dir model/YYYYMMDDHHMM_qwen25-05b-instruct-qa \
  --prompt "基準地震動の策定方法について教えてください。"

# Base LM の出力をスキップして MLP Memory のみ出力
pipenv run python mlp.py --mode infer \
  --model_dir model/YYYYMMDDHHMM_qwen25-05b-instruct-qa \
  --skip_base_lm
```

### QA 学習

```bash
# QA JSONL から直接学習
pipenv run python mlp.py --mode qa-train \
  --qa_path data/qa.jsonl \
  --epochs 20 \
  -m "コメント（必須）"
```

### 継続学習

```bash
pipenv run python mlp.py --mode qa-train \
  --qa_path data/qa.jsonl \
  --resume_from model/YYYYMMDDHHMM_qwen25-05b-instruct-qa \
  --epochs 10 \
  -m "resume from model/YYYYMMDDHHMM_qwen25-05b-instruct-qa. training: +10 epochs"
```

アーキテクチャ（`num_layers` / `use_final_layer` / `target_layer_index`）が一致しない場合はエラーを表示して終了。

### QA-kNN 学習

```bash
# 一括実行（qa-knn-build → knn → train → infer）
pipenv run python mlp.py --mode qa-knn-full \
  --qa_path data/qa.jsonl \
  -m "コメント（必須）"
```

### kNN 学習（コーパス方式）

```bash
pipenv run python mlp.py --mode build --corpus data/corpus.txt
pipenv run python mlp.py --mode knn
pipenv run python mlp.py --mode train -m "コメント（必須）"

# 一括実行
pipenv run python mlp.py --mode full --corpus data/corpus.txt \
  --prompt "基準地震動の策定方法について教えてください。" \
  -m "コメント（必須）"
```

### RAG（ベクトル検索）

```bash
# インデックス構築
pipenv run python mlp.py --mode rag-build --qa_path data/qa.jsonl

# 検索推論
pipenv run python mlp.py --mode rag-infer \
  --prompt "基準地震動の策定方法について教えてください。"
```

### Lambda サーベイ

```bash
# デフォルト λ=[0.6, 0.8, 1.0] で出力を比較
pipenv run python test/lambda_survey.py \
  --model_dir model/YYYYMMDDHHMM_qwen25-05b-instruct-qa \
  --rag_prefix tmp/rag_000155788 \
  --prompt "基準地震動の策定方法について教えてください。"

# 任意の λ 値を指定
pipenv run python test/lambda_survey.py \
  --model_dir model/YYYYMMDDHHMM_... \
  --lambdas 0.6 0.8 1.0
```

## 引数一覧（mlp.py）

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--mode` | `infer` | 動作モード（下記参照） |
| `--model_name` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace モデル名 |
| `--model_dir` | `None` | 保存済み MLPMemory ディレクトリ（`infer` 時に必須） |
| `--corpus` | `None` | テキストコーパス（`build`/`full` 時に必須） |
| `--save_prefix` | `tmp/datastore` | kNN datastore のプレフィックス |
| `--qa_path` | `None` | QA JSONL ファイル（`qa-train`/`qa-full`/`qa-knn-full` 時に必須） |
| `--max_samples` | `None` | QA サンプル数の上限（省略時は全件） |
| `--rag_prefix` | `tmp/rag` | RAG インデックスのプレフィックス |
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
| `qa-train` | QA JSONL から直接 MLP 訓練（`--qa_path` 必須） |
| `qa-full` | qa-train → infer（`--qa_path` 必須） |
| `qa-knn-full` | QA-kNN 学習 → infer（`--qa_path` 必須） |
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
augment.py         # QA データ生成（RAG ワークフロー）・テキスト拡張（augment モード）
src/
  config/settings.py    # 環境変数設定（.env から読込）
  core/llm_client.py    # Ollama API クライアント（QA 生成用）
  llm/gemini.py         # Gemini API クライアント（augment モード用）
  llm/kawarasaki.py     # kawarasaki02 API クライアント（未使用）
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
| `tmp/.qa_cache_*/` | QA トークナイズキャッシュ（自動生成・内部利用） |
| `data/text_vectors.db` | RAG 用 SQLite3 DB |
| `data/text_vectors.faiss` | RAG 用 FAISS インデックス |
| `model/YYYYMMDDHHMM_*/` | 訓練済み MLPMemory |

## 動作確認済み環境

- CUDA 12.2 / NVIDIA RTX 3090
- faiss-gpu-cu12
- Python 3.11
