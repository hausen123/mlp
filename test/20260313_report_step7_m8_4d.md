# 4d FFN M=8 スケーリングサーベイ Step7 モデル評価

## 目的
4d FFN・M=8構成でスケーリングサーベイStep7（wo_independence 131,072行混合）まで学習し、
独立性クエリおよび立地審査指針クエリで知識保持を確認する。
M=22ではStep4（17,761件）で崩壊したのに対し、M=8の耐性を検証する。

### モデルパス
model/202603130753_qwen25-05b-instruct-qa

### ベースモデル
Qwen/Qwen2.5-0.5B-Instruct（hidden_size=896, num_hidden_layers=24）

### アーキテクチャ
- target_layer: 16（= int(24 * 0.7)）
- use_final_layer: true（h = layer16 + final layer）
- num_layers (M): 8（MLPBlock 残差スタック数）
- FFN hidden_dim: 896 × 4 = 3584（4d）
- 総パラメータ: 約25.7M

### データ
- 学習JSONL: data/mixed_step7.jsonl（132,449件）
- 作成方法: data/independence.jsonl（1,377件）+ data/wo_independence.jsonl 先頭131,072行のconcat
  - wo_independence.jsonl: 元273,516件から「原子力規制委員会」AND「独立性」含む行を除外（272,139件）
  - independence比率: 約1.0%

### 学習設定
- mode: qa-train
- batch_size: 64, epochs: 10
- max_tokens_per_step: 2048
- 損失: E10=0.1388
- 学習時間: 192.8min
- GPU: RTX 3090 24GB, CUDA 12.2

### 再現コマンド
```bash
pipenv run python mlp.py \
  --mode qa-train \
  --qa_path data/mixed_step7.jsonl \
  --epochs 10 \
  --num_layers 8 \
  --use_final_layer \
  -m "4d M=8 scaling step7: independence(1377) + wo(131072)"
```

### 推論コマンド
```bash
pipenv run python mlp.py \
  --mode infer \
  --model_dir model/202603130753_qwen25-05b-instruct-qa \
  --lambda_interp 0.8 \
  --temperature 0.2 \
  --skip_base_lm \
  --prompt "XXXXX"
```

---

### プロンプト1
「原子力規制委員会の独立性について説明してください。」

### 出力
原子力規制委員会の独立性は、国家行政組織法第3条第2項に基づき設置された3条委員会として、独立してその職権を行使することが求められています。具体的には、委員長及び委員は、両議院の同意を得て内閣総理大臣が任命するとされ、任期中は任命権者の一存で罷免できないなど、身分保障の観点からも独立性が高められているためです。

また、原子力規制委員会は、専門的な技術的裁量を持ち、独立かつ中立の立場から原子力規制を行うことが可能であることを示しています。このように、原子力規制委員会は、政治的・経済的影響や他の政府機関からの不当な圧力に左右されることなく、科学的・技術的な判断に基づいて中立的に職務を遂行することを保証していることから、独立性が高められていると言えます。

### 結果コメント
✓ 良質。3条委員会・両議院の同意・罷免不可・身分保障・政治的影響排除が含まれる。
independence比率1%にもかかわらず知識を保持。M=22（Step4で崩壊）と対照的。

---

### プロンプト2
「立地審査指針の原則的立地条件とは？」

### 出力
立地審査指針の原則的立地条件とは、過去に大きな事故や災害の誘因となるような事象がなかったこと、原子力発電所がその安全防護施設との関連性を考慮した結果、公衆の安全を確保するために必要な適切な場所を選ぶことができるというものです。これは、（旧）仮想事故に基づく考え方であり、原子力発電所の安全性を確保するための重要な要件です。

### 結果コメント
✓ 学習データに準拠した回答。仮想事故・公衆安全確保・適切立地の概念が含まれる。
independence.jsonlとは異なるドメイン（wo_independence側）の知識も保持されている。

---

### 調査ファイル
test/20260312_scaling_survey_4d_m8.md
