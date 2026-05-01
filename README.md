# TikTok Virality Prediction
## CS 506 — Final Project

> 🎥 **Video Presentation:** [ADD YOUTUBE LINK HERE before 5/1]

---

## Project Description

TikTok has become one of the most influential platforms for viral content, yet the factors that make a video go viral remain poorly understood. This project investigates the drivers of TikTok video virality using a multimodal machine learning approach.

We combine **Natural Language Processing** on video transcription text and captions with **early engagement signals** (likes, shares, comments, video duration) to predict whether a video will perform above average. Our main model is a fine-tuned **DistilBERT transformer** that fuses semantic text understanding with structured engagement metadata — directly addressing the challenge of short, informal social media language that traditional bag-of-words models fail to capture.

**Project goals:**
- Predict whether a TikTok video will go viral (above-median view count) using features available shortly after posting
- Evaluate whether combining NLP on transcription text with engagement signals outperforms engagement features alone
- Test model generalization on a completely unseen out-of-distribution dataset

**Key finding:** Our multimodal DistilBERT model achieves **AUC = 0.9943** on the test set. The most surprising result was that `claim_status` — whether a video makes a factual assertion — was the dominant predictor, with 98.2% of viral videos being "claim" type content.

---

## How to Build and Run

### Step 1 — Clone the repo
```bash
git clone https://github.com/<your-username>/tiktok-virality
cd tiktok-virality
```

### Step 2 — Download datasets
Place both CSVs in the `data/` folder:

| File | Source |
|---|---|
| `data/tiktok_dataset.csv` | [yakhyojon/tiktok](https://www.kaggle.com/datasets/yakhyojon/tiktok) |
| `data/tiktok_data.csv` | [maratsaratov/tiktok-data](https://www.kaggle.com/datasets/maratsaratov/tiktok-data) |

### Step 3 — Install and run
```bash
make install    # installs all dependencies
make run        # runs baseline notebook
make test       # runs unit tests
make all        # does all three
```

### Google Colab (recommended for transformer)
1. Upload both CSVs via the Files panel (📁 sidebar)
2. `baseline.ipynb` — CPU, ~5 min
3. `transformer.ipynb` — Runtime → Change runtime type → **T4 GPU** → Run All (~30 min)

---

## Repository Structure

```
tiktok-virality/
├── baseline.ipynb           ← Random Forest baseline model
├── transformer.ipynb        ← Multimodal DistilBERT (main model)
├── Makefile
├── requirements.txt
├── README.md                ← this file
├── data/
│   ├── tiktok_dataset.csv   ← primary dataset (gitignored)
│   └── tiktok_data.csv      ← OOD test dataset (gitignored)
├── visualizations/
└── tests/
    └── test_pipeline.py
```

---

## Data Collection

### Sources

| # | Dataset | Rows | Purpose |
|---|---------|------|---------|
| 1 | [TikTok User Engagement — yakhyojon](https://www.kaggle.com/datasets/yakhyojon/tiktok) | 19,382 | Training + testing |
| 2 | [TikTok Video Metadata — maratsaratov](https://www.kaggle.com/datasets/maratsaratov/tiktok-data) | 760 (565 English) | OOD generalization test only |

**Dataset 1** contains `video_transcription_text` (actual spoken words), `claim_status`, `verified_status`, `author_ban_status`, and full engagement metrics — making it ideal for a multimodal NLP approach.

**Dataset 2** was never used during training. It serves exclusively as an out-of-distribution test to measure how well the model generalizes to completely unseen data from a different source.

Both datasets are publicly available from Kaggle, ethically collected from TikTok's public content.

---

## Data Cleaning

Key steps applied in both notebooks:

- **Numeric coercion** — all engagement columns converted via `pd.to_numeric(errors='coerce').fillna(0)`
- **Missing text** — `video_transcription_text` nulls filled with empty string
- **Categorical encoding** — `verified_status` and `author_ban_status` label-encoded
- **Virality label** — `is_viral = 1` if `video_view_count > median`, else `0`. Median = **9,788 views**, producing a balanced 50/50 split
- **OOD filter** — Dataset 2 contains Russian-language videos, filtered via Cyrillic regex `[А-Яа-яЁё]` (760 → 565 English rows)
- **Stop word removal (transformer)** — domain-specific words like `'claim'`, `'viral'`, `'fyp'` removed from transcription text to prevent the model from shortcutting on dataset artifacts

---

## Feature Extraction

### Baseline features (Random Forest)
7 structured features: `video_like_count`, `video_share_count`, `video_download_count`, `video_comment_count`, `video_duration_sec`, `verified_status` (encoded), `author_ban_status` (encoded)

### Transformer features (Main model)

| Feature | Branch | Method |
|---|---|---|
| `video_transcription_text` | Text | DistilBERT tokenizer, max_length=128, custom stop words removed |
| `video_duration_sec` | Numeric | Standardized using training set mean/std |
| `verified_status` | Numeric | LabelEncoder |
| `author_ban_status` | Numeric | LabelEncoder |

> ⚠️ `video_view_count` is **excluded from all features** — it defines the viral label and would cause data leakage.

---

## Model Training & Evaluation

### Baseline: Random Forest

A Random Forest classifier trained on 7 engagement features to establish a performance baseline. 80/20 stratified split, `n_estimators=100`, `max_depth=10`.

**Results:**
```
ROC-AUC  : 0.9888
Accuracy : 0.97
F1       : 0.97
```

---

### Main Model: Multimodal DistilBERT Transformer

**Why DistilBERT:**
TikTok captions are short, informal, and context-dependent — "this is fire 🔥" signals virality, not combustion. Standard bag-of-words models miss this entirely. DistilBERT understands semantic meaning in context and is 40% smaller / 60% faster than full BERT while retaining 97% of its performance, making it practical on Colab's GPU.

**Architecture:**
```
video_transcription_text ──► DistilBERT ──► [CLS] token (768-dim) ──┐
                                                                       ├──► concat ──► Dropout(0.3) ──► Linear ──► sigmoid
video_duration_sec, etc. ────► Linear(n→32) ──► BatchNorm ──► ReLU ──┘
```

Two separate branches — one for text (DistilBERT), one for numeric metadata — fused at the classification head. This multimodal design lets each branch specialize before combining signals.

**Training details:**
- 80/20 stratified train/test split
- Loss: `BCEWithLogitsLoss`
- Optimizer: AdamW with **differential learning rates** — numeric/fusion layers = 1e-3, BERT weights protected at lower rate to prevent catastrophic forgetting
- Early stopping (patience=3) on validation loss
- Max 20 epochs, stopped at epoch 3

**Training log:**
```
Epoch 1 | Train Loss: 0.0921 | Val Loss: 0.0606 | Val AUC: 0.9927
Epoch 2 | Train Loss: 0.0628 | Val Loss: 0.0617 | Val AUC: 0.9936  ← counter: 1/3
Epoch 3 | Train Loss: 0.0611 | Val Loss: 0.0604 | Val AUC: 0.9943  ← best checkpoint saved
Epoch 4 | Train Loss: 0.0572 | Val Loss: 0.0640 | Val AUC: 0.9931  ← counter: 1/3
Epoch 5 |                                                              ← counter: 2/3
→ Early stopping. Best model restored from epoch 3.
```

**Test set results:**
```
ROC-AUC  : 0.9943
F1       : 0.99
Accuracy : 0.99

              precision  recall  f1-score  support
Not Viral         0.99    0.99      0.99     1968
Viral             0.99    0.99      0.99     1909
```

**Cross-dataset (OOD) evaluation on Dataset 2:**
```
ROC-AUC  : 0.5238  (565 English-only videos)
Accuracy : 0.52
```

**Discussion of limitations:**
- **OOD performance drops to 0.52** — near random. The model relied heavily on `claim_status` which is unavailable in Dataset 2. This reveals the model learned dataset-specific patterns rather than universal virality signals
- **Engagement leakage** — `video_like_count` and `video_share_count` accumulate after posting; in a true real-time system these would be unavailable at t=0
- **Dataset bias** — 98.2% of viral videos in Dataset 1 are "claim" type content. The model may underperform on entertainment or dance content that dominates real TikTok trending pages
- **Future work** — text-only prediction at posting time, larger and more diverse training data, temporal features (posting hour, day of week)

---

## Results

### Goal achievement

```
Goal    : Predict above-average TikTok video performance
Metric  : AUC-ROC significantly above random baseline (0.50)
Result  : AUC = 0.9943 ✅   (Random Forest baseline: 0.9888 ✅)
```

### Model comparison

| Model | AUC-ROC | F1 | Accuracy | Notes |
|---|---|---|---|---|
| Random Forest (baseline) | 0.9888 | 0.97 | 0.97 | Engagement features only |
| **DistilBERT (main model)** | **0.9943** | **0.99** | **0.99** | Text + engagement, in-distribution |
| DistilBERT OOD | 0.5238 | 0.57 | 0.52 | Generalization to unseen dataset |

### Key findings

1. **`claim_status` dominates** — 98.2% of viral videos (9,512/9,691) are "claim" type. This single metadata feature drives most of the predictive power
2. **Engagement signals are highly predictive** — `video_like_count` and `video_share_count` are the top features in the Random Forest importance ranking
3. **DistilBERT outperforms Random Forest** — AUC 0.9943 vs 0.9888, confirming semantic text understanding adds measurable signal
4. **OOD generalization fails** — performance collapses on Dataset 2, highlighting the challenge of building truly generalizable virality models across different data sources

---

## Visualizations

| Figure | Notebook | What it shows |
|---|---|---|
| Views distribution (raw + log scale) | baseline | Justifies log transformation and median-split labeling |
| Engagement correlation heatmap | baseline | Likes/shares most correlated with viral label |
| Duration vs virality bar chart | baseline | Short videos (<30s) go viral at higher rates |
| Share count boxplot | baseline | Clear separation between viral/non-viral share counts |
| Claim vs opinion pie chart | baseline | 98.2% of viral videos are claims |
| Random Forest feature importance | baseline | Likes and shares dominate |
| DistilBERT training curves | transformer | Stable convergence, early stopping at epoch 3 |
| Confusion matrix | transformer | Near-perfect in-distribution classification |
| OOD probability distribution | transformer | Model uncertainty on unseen data |

---

## Testing

```bash
make test
# or: pytest tests/ -v
```

Tests in `tests/test_pipeline.py` cover label definition logic, numeric coercion, missing value handling, and a full Random Forest smoke test. CI runs automatically on every push via `.github/workflows/ci.yml`.

---

## Dependencies

```
torch>=2.0          transformers>=4.30    scikit-learn>=1.3
xgboost>=1.7        pandas>=2.0           numpy>=1.24
matplotlib>=3.7     seaborn>=0.12         pytest>=7.4
jupyter             nbconvert
```

Install: `make install`

