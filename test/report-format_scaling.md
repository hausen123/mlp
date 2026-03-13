# [タイトル（例: スケーリングサーベイ（Xd FFN, M=Y）— 独立性クエリ品質 vs データ量）]

## 目的
[スケーリング実験の目的を説明。何のアーキテクチャで何を比較するか。]

## 変更点
- FFN: Linear(d, Xd)（mlp.py MLPBlock）
- M=Y（num_layers=Y）

## ベースモデル
Qwen/Qwen2.5-0.5B-Instruct（hidden_size=896, num_hidden_layers=24）

## アーキテクチャ
- target_layer: 16（= int(24 * 0.7)）
- use_final_layer: true（h = layer16 + final layer）
- num_layers (M): Y（MLPBlock 残差スタック数）
- FFN hidden_dim: 896 × X = ZZZZ（Xd）
- 総パラメータ: 約XX.XM

## データ準備
- 元JSONL: `data/XXXXX.jsonl`（N件）
- `data/wo_independence.jsonl`: 元JSOLから「原子力規制委員会」AND「独立性」を含む行を除外（N件）
- `data/independence.jsonl`: 1,377件（固定）

## 実験ステップ

| Step | w/o行数 | 総件数 | モデル | 結果 |
|------|---------|-------|-------|------|
| 1 | 0 | 1,377 | YYYYMMDDHHMM | △/✓ |
| 2 | 4,096 | 5,473 | YYYYMMDDHHMM | △/✓ |
| 3 | 8,192 | 9,569 | YYYYMMDDHHMM | △/✓ |
| 4 | 16,384 | 17,761 | YYYYMMDDHHMM | △/✓ |
| 5 | 32,768 | 34,145 | YYYYMMDDHHMM | △/✓ |
| 6 | 65,536 | 66,913 | YYYYMMDDHHMM | △/✓ |
| 7 | 131,072 | 132,449 | YYYYMMDDHHMM | △/✓ |
| 8 | — | 273,516（元JSONL全件） | YYYYMMDDHHMM | △/✓ |

## 学習設定
- mode: qa-train
- batch_size: 64, epochs: 10
- max_tokens_per_step: 2048
- GPU: RTX 3090 24GB, CUDA 12.2

## 学習コマンド（テンプレート）
```bash
pipenv run python mlp.py \
  --mode qa-train \
  --qa_path data/mixed_stepN.jsonl \
  --epochs 10 \
  --num_layers Y \
  --use_final_layer \
  -m "Xd M=Y scaling step N: independence(1377) + wo(XXXX)"
```

## 推論コマンド（λ=0.8, temperature=0.2）
```bash
pipenv run python mlp.py \
  --mode infer \
  --model_dir model/YYYYMMDDHHMM_qwen25-05b-instruct-qa \
  --lambda_interp 0.8 \
  --temperature 0.2 \
  --skip_base_lm \
  --prompt "XXXXX"
```

## 結果

### Step 1: independence only（1,377件）
- モデル: `model/YYYYMMDDHHMM_qwen25-05b-instruct-qa`
- 学習: loss E10=X.XXXX、X.Xmin
- 品質: △/✓（キーワード確認・問題点）

**λ=0.8 出力:**
[省略なしの生成テキスト]

### Step 2: independence + wo 4,096行（計5,473件）
- モデル: `model/YYYYMMDDHHMM_qwen25-05b-instruct-qa`
- 学習: loss E10=X.XXXX、X.Xmin
- 品質: △/✓（キーワード確認・問題点）

**λ=0.8 出力:**
[省略なしの生成テキスト]

### Step 3: independence + wo 8,192行（計9,569件）
- モデル: `model/YYYYMMDDHHMM_qwen25-05b-instruct-qa`
- 学習: loss E10=X.XXXX、X.Xmin
- 品質: △/✓（キーワード確認・問題点）

**λ=0.8 出力:**
[省略なしの生成テキスト]

### Step 4: independence + wo 16,384行（計17,761件）
- モデル: `model/YYYYMMDDHHMM_qwen25-05b-instruct-qa`
- 学習: loss E10=X.XXXX、X.Xmin
- 品質: △/✓（キーワード確認・問題点）

**λ=0.8 出力:**
[省略なしの生成テキスト]

### Step 5: independence + wo 32,768行（計34,145件）
- モデル: `model/YYYYMMDDHHMM_qwen25-05b-instruct-qa`
- 学習: loss E10=X.XXXX、X.Xmin
- 品質: △/✓（キーワード確認・問題点）

**λ=0.8 出力:**
[省略なしの生成テキスト]

### Step 6: independence + wo 65,536行（計66,913件）
- モデル: `model/YYYYMMDDHHMM_qwen25-05b-instruct-qa`
- 学習: loss E10=X.XXXX、X.Xmin
- 品質: △/✓（キーワード確認・問題点）

**λ=0.8 出力:**
[省略なしの生成テキスト]

### Step 7: independence + wo 131,072行（計132,449件）
- モデル: `model/YYYYMMDDHHMM_qwen25-05b-instruct-qa`
- 学習: loss E10=X.XXXX、X.Xmin
- 品質: △/✓（キーワード確認・問題点）

**λ=0.8 出力:**
[省略なしの生成テキスト]

### Step 8: 元JSONL全件（273,516件）
- モデル: `model/YYYYMMDDHHMM_qwen25-05b-instruct-qa`
- 学習: loss E10=X.XXXX、X.Xmin
- データ: [混合方法の説明、例: wo_independence + independence の混合 / 元JSONL全件]
- 品質: △/✓（キーワード確認・問題点）

**λ=0.8 出力（クエリ1）:**
[省略なしの生成テキスト]

**λ=0.8 出力（クエリ2）:**
[省略なしの生成テキスト]

## 考察
- 品質劣化閾値: Step X（N件）から劣化 / 全ステップで品質維持
- M=22比較: [M=22との閾値比較]
- 結論: [全体的な考察]
