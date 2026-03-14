# 推論速度・メモリ比較レポート（2026-03-14）

## 目的
ベースモデルサイズ（0.5B vs 4B）と推論方式（plain generate / token-by-token / hidden states取り出し）が
推論速度・VRAMに与える影響を定量的に比較する。

## 環境
- GPU: RTX 3090 24GB, CUDA 12.2
- プロンプト: 「原子力発電所の耐震設計において、基準地震動の策定方法について説明してください。」
- max_new_tokens: 200

## テストスクリプト
- `test/infer_speed_test.py` — plain generate のみ（ベースライン）
- `test/infer_hidden_test.py` — 3方式比較（plain / token-by-token / hidden states）

## モデル

| モデル | パラメータ | hidden_size | layers | target_layer_index | VRAM（weights） |
|--------|-----------|-------------|--------|--------------------|----------------|
| Qwen/Qwen2.5-0.5B-Instruct | 0.5B | 896 | 24 | 16（= int(24×0.7)） | 0.99 GB |
| Qwen/Qwen3-4B-Instruct-2507 | 4B | 2560 | 36 | 25（= int(36×0.7)） | 8.05 GB |

## 結果

### 推論速度（tok/s）

| 方式 | Qwen2.5-0.5B | Qwen3-4B |
|------|-------------|---------|
| plain generate | 36.0 | 19.5 |
| token-by-token | 36.5 | 19.5 |
| hidden states=True | 35.8 | 19.2 |

### VRAMピーク（GB）・delta

| 方式 | 0.5B base | 0.5B peak | 0.5B delta | 4B base | 4B peak | 4B delta |
|------|-----------|-----------|------------|---------|---------|---------|
| plain generate | 1.00 GB | 1.00 GB | +5 MB | 8.05 GB | 8.09 GB | +36 MB |
| token-by-token | 1.00 GB | 1.02 GB | +18 MB | 8.05 GB | 8.09 GB | +35 MB |
| hidden states=True | 1.00 GB | 1.02 GB | +21 MB | 8.05 GB | 8.09 GB | +36 MB |

## 考察

### 推論速度
- Qwen3-4B は Qwen2.5-0.5B の約 **1.85倍遅い**（モデルが8倍大きい割には差が小さい）
- `token-by-token`（Pythonループ）と `plain generate`（HuggingFace最適化実装）で速度差なし
  → KVキャッシュが効いており、各ステップの計算量は同等
- `output_hidden_states=True` の速度オーバーヘッドはほぼゼロ（< 2%）

### メモリ
- `output_hidden_states=True` のメモリ増加は微小（0.5B: +3MB、4B: +1MB）
  → 各トークン生成ごとに隠れ状態を取り出してすぐ参照・破棄するためピーク増加が抑えられる
- plain generate は KVキャッシュのみで動くため delta が最小（+5MB）
- token-by-token は logits tensor を毎ステップ生成するため delta がやや大きい（+18MB）
- **MLP Memory推論のVRAMボトルネックはhidden state取り出しではなく、モデルウェイト自体**

### Qwen3-4B でのMLP学習可否
- weights 8.05 GB + backward/optimizer states ≒ 24GB超 → RTX 3090でOOM
- 推論のみなら 8.09 GB で動作可能（MLP追加でも +数十MBで収まる見込み）
- 学習には80GB GPU（A100/H100）か量子化（QLoRA等）が必要

## 結論
推論方式（plain / token-by-token / hidden states）による速度・メモリ差は無視できるレベル。
MLP Memory推論を4Bモデルに適用しても **~19 tok/s、~8.1 GB** で動作可能。
