# VRAM使用量レポートと改善案（2026-03-15）

## 環境

- GPU: RTX 3090 24GB, CUDA 12.2
- モデル: Qwen2.5-0.5B-Instruct (0.5B) / Qwen3-4B-Instruct-2507 (4B)
- 学習設定: batch_size=64 (0.5B) / 16 (4B), seq_len=512, max_tokens_per_step=2048
- hidden state取得: forward hook（target層+最終層のみ、`_extract_hidden_hook`）

---

## 1. 実測VRAM推移

| ステージ | 0.5B alloc | 0.5B peak | 4B alloc | 4B peak |
|---------|-----------|----------|---------|---------|
| ① base model ロード | 0.99 GB | 0.99 GB | 8.04 GB | 8.04 GB |
| ② MLP + optimizer 初期化 | 1.74 GB | 1.74 GB | 11.91 GB | 11.91 GB |
| ③ base model forward (hook) | ~1.97 GB | **3.52 GB** | ~12.0 GB | **13.81 GB** |
| ④ keys抽出後（MLP forward前） | 1.93 GB | — | 11.96 GB | — |
| ⑤ MLP forward chunk0 | — | 3.76 GB | — | **15.51 GB** |
| ⑥ backward | — | **7.37 GB** | — | **19.23 GB** |
| ⑦ optimizer.step() 後 | 3.80 GB | **7.59 GB** | 20.14 GB | **22.45 GB** |

> ⑦ peak = optimizer.step()初回でAdam statesが確保されるため alloc を超える

---

## 2. peak 22.45GB の理論内訳（4B, ⑦ optimizer.step()時）

```
base model weights (bf16)              8.04 GB
MLP weights      (fp32, 576.9M)        2.31 GB
embed_weight buffer (fp32)             1.55 GB   ← vocab×hidden_dim 定数テンソル
Adam exp_avg     (fp32)                2.31 GB   ← MLP weightsと同サイズ
Adam exp_avg_sq  (fp32)                2.31 GB   ← 同上
gradients        (fp32)                2.31 GB   ← 同上
─────────────────────────────────────────────────
persistent 小計                       18.83 GB

logits tensor [2048, 151936] (fp32)    1.24 GB   ← backward中に生存
backward activations (MLP 11ブロック)  ~2.38 GB
─────────────────────────────────────────────────
peak 合計                             ~22.45 GB
```

### MLP関連だけで MLP weights の4倍を消費

| テンソル | サイズ | 倍率 |
|---------|------|-----|
| MLP weights | 2.31 GB | ×1 |
| gradients | 2.31 GB | ×1 |
| Adam exp_avg | 2.31 GB | ×1 |
| Adam exp_avg_sq | 2.31 GB | ×1 |
| **MLP関連合計** | **9.24 GB** | **×4** |

### MLP パラメータ数（実測）

| モデル | hidden_dim | M (blocks) | FFN構造 | 合計 |
|--------|-----------|-----------|--------|------|
| 0.5B | 896 | 8 | Linear(896→3584→896) | 51.4M = 0.21GB fp32 |
| 4B | 2560 | 11 | Linear(2560→10240→2560) | 576.9M = 2.31GB fp32 |

> FFN展開比は **4×**（2×と誤推定していたが実測で訂正）

### embed_weight buffer の内訳

| モデル | テンソル形状 | サイズ(fp32) | 備考 |
|--------|------------|------------|------|
| 0.5B | [151936, 896] | 0.54 GB | base modelと同居、detached copy |
| 4B | [151936, 2560] | 1.55 GB | 同上 |

---

## 3. 改善案

### 案① 8-bit Adam（最大効果・コスト最小）

```python
import bitsandbytes as bnb
optimizer = bnb.optim.AdamW8bit(mlp.parameters(), lr=4e-4)
```

Adam statesをint8量子化（exp_avg + exp_avg_sq → 各1/4）

| | 現状 | 改善後 | 削減 |
|--|------|-------|------|
| Adam states (0.5B) | 0.41 GB | 0.10 GB | -0.31 GB |
| Adam states (4B) | 4.62 GB | **1.15 GB** | **-3.47 GB** |

精度への影響：ほぼなし（実績多数）

---

### 案② embed_weight を bf16 で保持

```python
self.register_buffer("embed_weight", embed_weight.bfloat16())
# forward内:
logits = torch.matmul(h.bfloat16(), self.embed_weight.T).float()
```

| | 現状 | 改善後 | 削減 |
|--|------|-------|------|
| embed_weight (0.5B) | 0.54 GB | 0.27 GB | -0.27 GB |
| embed_weight (4B) | 1.55 GB | **0.77 GB** | **-0.78 GB** |

---

### 案③ AMP（Automatic Mixed Precision）

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    logits = mlp(k)
```

forward/backward をbf16で計算、activationsとlogitsが半分に

| | 現状 | 改善後 | 削減 |
|--|------|-------|------|
| logits [2048,151936] | 1.24 GB | 0.62 GB | -0.62 GB |
| backward activations (4B) | ~2.38 GB | ~1.19 GB | ~-1.19 GB |
| **4B合計** | | | **~-1.81 GB** |

---

## 4. 改善後のpeak VRAM推定（4B）

| 施策 | peak削減 |
|------|---------|
| 現状 | **22.45 GB** |
| + 案①（8-bit Adam） | **-3.47 GB** → 18.98 GB |
| + 案②（embed bf16） | **-0.78 GB** → 18.20 GB |
| + 案③（AMP bf16） | **-1.81 GB** → 16.39 GB |
| **全案適用** | **~16.4 GB** |

RTX 3090 (24GB) に対して現状1.55GBの余裕しかないが、
案①だけで5GB超の余裕が確保でき、batch_sizeを16→32〜64に増やすことも可能になる。

---

## 5. 推論時VRAM（参考）

| モデル | peak |
|--------|------|
| 0.5B（base + MLP） | ~1.1 GB |
| 4B（base + MLP） | ~8.2 GB |

推論時はAdam states・gradients・activationsが不要のため学習時と大きく異なる。
4Bの推論であれば余裕をもってRTX 3090で動作する。
