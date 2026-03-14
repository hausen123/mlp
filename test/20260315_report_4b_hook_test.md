# 4B MLP hook実装テスト（動作確認）

## 目的
`feature/hook-hidden-states` ブランチの `_extract_hidden_hook` 実装を Qwen3-4B-Instruct-2507 で動作確認する。
1epochの軽量学習でモデルを作成し、推論パイプラインの正常動作を確認する。

### モデルパス
model/202603142135_qwen3-4b-instruct-2507-qa

### ベースモデル
Qwen/Qwen3-4B-Instruct-2507（hidden_size=2560, num_hidden_layers=36）

### アーキテクチャ
- target_layer: 25（= int(36 * 0.7)）
- use_final_layer: true（h = layer25 + final layer）
- num_layers (M): 11（MLPBlock 残差スタック数）
- FFN hidden_dim: 2560 × 4 = 10240（4d）
- 総パラメータ: 576.9M（2.31GB fp32）

### データ
- 学習JSONL: data/old/mixed_step2.jsonl（5473件、ritch+independence混合）
- 作成方法: catastrophic forgetting実験用の混合リプレイデータ（0.5B学習で使用済み）

### 学習設定
- mode: qa-train
- batch_size: 16, epochs: 1
- max_tokens_per_step: 2048
- 損失: E1=2.267（1epochのみ、学習途中）
- 学習時間: 5.8min
- GPU: RTX 3090 24GB, CUDA 12.2
- 実装: forward hook（output_hidden_states=True 廃止、target+final層のみ保持）
- VRAM peak: 22.45GB（bs=16）

### 再現コマンド
```bash
pipenv run python mlp.py \
  --mode qa-train \
  --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --qa_path data/old/mixed_step2.jsonl \
  --batch_size 16 \
  --epochs 1 \
  --num_layers 11 \
  --use_final_layer \
  -m "hook test: Qwen3-4B-Instruct-2507 mixed_step2 bs=16 1ep"
```

### 推論コマンド
```bash
pipenv run python mlp.py \
  --mode infer \
  --model_dir model/202603142135_qwen3-4b-instruct-2507-qa \
  --lambda_interp 0.8 \
  --prompt "原子力規制委員会の独立性について説明してください。"
```

※ `--model_name` 不要（model_dir/config.json の base_model_name を自動参照）

### プロンプト
「原子力規制委員会の独立性について説明してください。」

### 出力

**λ=0.0（pure Base LM）**
```
原子力規制委員会（AFC）は、日本のエネルギー政策において重要な役割を果たしています。
（組織的独立・財政的独立・人事の独立・規制権限の独立性を網羅。福島事故後の改革を正確に記述）
```

**λ=0.6（MLP Memory）**
```
災害対策や基本的な法的枠組みについては、原子力災害対策基本法が適用されています。
→ 「政府の個別的な指揮監督権を排除」「専門的知見に基づいて判断」等のキーワードは出現するが
  途中から「法法法法…」のloop collapseに移行
```

**λ=0.8（MLP Memory）**
```
災害対策委員会は…
→「政府の個別的な指揮監督権を排除」「身分保障」「任命権者の一存で罷免できない」等
  正確なキーワードが含まれるが、早期に「適用法適用法…」のloop collapseに移行
```

**λ=1.0（pure MLP）**
```
→ λ=0.8と同様のパターン。loop collapseが全λで共通
```

### 結果コメント

**動作確認（目的達成）:**
- `base_model_name` 自動参照修正: `--mode infer --model_dir` 指定時に config.json から読み込むよう修正、正常動作確認
- Target layer 25/36, M=11 正常ロード・推論確認
- VRAM peak 22.45GB（24GB以内に収まる）

**品質評価:**
- Base LM（Qwen3-4B単独）: 全λで高品質。0.5Bと比べ構造化・正確性が大幅に向上
- MLP Memory: 全λでloop collapse。1epoch・loss=2.267では学習不足（想定内）
- キーワードは混合データから引き出されるが（「指揮監督権を排除」「罷免不可」）、安定生成には至らない

**次のステップ:**
- 4Bモデルの本格学習（epochs増加、bs増加）には8-bit Adam等のVRAM削減施策が推奨
- 現状bs=16/1epochはテスト用途のみ

### 調査ファイル
test/lambda_survey_202603142135.txt
