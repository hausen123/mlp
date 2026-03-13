# [タイトル]

## 目的
[モデル作成目的の説明]

### モデルパス
model/YYYYMMDDHHMM_qwen25-05b-instruct-qa

### ベースモデル
Qwen/Qwen2.5-0.5B-Instruct（hidden_size=896, num_hidden_layers=24）

### アーキテクチャ
- target_layer: 16（= int(24 * 0.7)）
- use_final_layer: [true/false]（h = layer16 + final layer）
- num_layers (M): 8（MLPBlock 残差スタック数）
- FFN hidden_dim: 896 × 4 = 3584（4d）
- 総パラメータ: 約25.7M

### データ
- 学習JSONL: data/XXXXX.jsonl（N件、説明）
- 作成方法: [元JSOLからの抽出・混合方法を記述]

### 学習設定
- mode: [qa-train / qa-full / full]
- batch_size: 64, epochs: 10
- max_tokens_per_step: 2048
- 損失: E1=X.XXX → E10=X.XXX
- 学習時間: XX.Xmin
- GPU: RTX 3090 24GB, CUDA 12.2

### 再現コマンド
```bash
pipenv run python mlp.py \
  --mode qa-train \
  --qa_path data/XXXXX.jsonl \
  --epochs 10 \
  --num_layers 8 \
  --use_final_layer \
  -m "説明"
```

### 推論コマンド
```bash
pipenv run python mlp.py \
  --mode infer \
  --model_dir model/YYYYMMDDHHMM_qwen25-05b-instruct-qa \
  --lambda_interp 0.8 \
  --temperature 0.2 \
  --skip_base_lm \
  --prompt "XXXXX"
```

### プロンプト
「XXXXX」

### 出力
[省略なしの生成テキスト]

### 結果コメント
[品質評価・キーワード確認・問題点など]

### 調査ファイル
test/lambda_survey_YYYYMMDDHHMM.txt
