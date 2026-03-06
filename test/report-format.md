## [説明] λ=X.X 最良メモ

### モデルパス
model/YYYYMMDDHHММ_qwen25-05b-instruct

### データパス
data/XXXXX.jsonl（データセット説明）
→ datastoreプレフィックス: XXXXX_ds

### 学習設定
- mode: [qa-knn-full / full / qa-train]
- base_model: Qwen/Qwen2.5-0.5B-Instruct
- target_layer: 16 / use_final_layer: [true/false]
- hidden_dim: 896, num_layers: 22
- K: 64, tau: 1.0, alpha: 0.4
- batch_size: 64, epochs: N
- 損失: E1=X.XXX → EN=X.XXX

### プロンプト
「XXXXX」

### λ=X.X 出力
[省略なしの生成テキスト]

### 調査ファイル
test/lambda_survey_YYYYMMDDHHММ.txt
