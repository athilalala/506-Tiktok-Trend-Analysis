# TikTok Virality Prediction
## CS 506 — Final Project

>  **Video Presentation:** [ADD YOUTUBE LINK HERE before 5/1]

---

## Project Description

This project investigates the factors that drive TikTok video virality using a multimodal machine learning approach. We combine Natural Language Processing on video transcription text with early engagement signals (likes, shares, comments, video duration) to predict whether a TikTok video will perform above average within the first 24–72 hours of posting.
Our main model is a fine-tuned DistilBERT transformer that fuses semantic text understanding with structured engagement metadata — directly addressing the challenge of short, informal social media language that traditional bag-of-words models fail to capture. We also train a Random Forest baseline using engagement features alone to measure the contribution of text-based NLP.
**Project goals:**

Predict whether a TikTok video will go viral (above-median view count) using features available shortly after posting
Evaluate whether combining NLP on transcription text with engagement signals outperforms engagement features alone
Test model generalization on a completely unseen out-of-distribution dataset

**Key finding:** 
Our multimodal DistilBERT model achieves AUC = 0.9943 on the test set. The most surprising result was that claim_status — whether a video makes a factual assertion rather than an opinion — was the dominant predictor, with 98.2% of viral videos being "claim" type content.
---

## How to Build and Run

> **This is the most important section — follow these steps to fully reproduce our results.**

### Prerequisites
- Python 3.10+
- GPU recommended for transformer (free T4 on Google Colab)
- Kaggle account to download datasets

### Step 1 — Clone the repo
```bash
git clone https://github.com/<your-username>/tiktok-virality
cd tiktok-virality
```

### Step 2 — Download datasets
Download both CSVs from Kaggle and place in the `data/` folder:

| File | Kaggle Link |
|---|---|
| `data/tiktok_dataset.csv` | [yakhyojon/tiktok](https://www.kaggle.com/datasets/yakhyojon/tiktok) |
| `data/tiktok_data.csv` | [maratsaratov/tiktok-data](https://www.kaggle.com/datasets/maratsaratov/tiktok-data) |

### Step 3 — Install dependencies and run
```bash
make install    # pip installs all dependencies from requirements.txt
make run        # executes baseline.ipynb and saves outputs
make test       # runs unit tests in tests/
make all        # runs all three steps above
```

### Step 4 — Run on Google Colab (required for transformer model)
The transformer model requires a GPU. Run it on Google Colab:

1. Go to [colab.research.google.com](https://colab.research.google.com) and open `transformer.ipynb`
2. Select **Runtime → Change runtime type → T4 GPU**
3. Upload `tiktok_dataset.csv` and `tiktok_data.csv` using the 📁 Files panel
4. Click **Runtime → Run All** — completes in ~30 minutes

The baseline model (`baseline.ipynb`) runs on CPU and completes in ~5 minutes via `make run`.

### Testing & CI
```bash
make test
# or directly: pytest tests/ -v
```
Unit tests in `tests/test_pipeline.py` cover label logic, data cleaning, and model smoke tests.
The GitHub Actions workflow (`.github/workflows/ci.yml`) runs all tests automatically on every push.

---

## Repository Structure

```
tiktok-virality/
├── baseline.ipynb           ← Random Forest baseline model
├── transformer.ipynb        ← Multimodal DistilBERT (main model)
├── Makefile                 ← install / run / test
├── requirements.txt         ← all dependencies
├── README.md                ← this file
├── data/
│   ├── tiktok_dataset.csv   ← primary dataset (gitignored)
│   └── tiktok_data.csv      ← OOD test dataset (gitignored)
├── visualizations/          ← saved figures
└── tests/
    └── test_pipeline.py     ← unit tests
```

---

## Data Collection

| # | Dataset | Rows | Purpose |
|---|---------|------|---------|
| 1 | [TikTok User Engagement — yakhyojon](https://www.kaggle.com/datasets/yakhyojon/tiktok) | 19,382 | Training + testing |
| 2 | [TikTok Video Metadata — maratsaratov](https://www.kaggle.com/datasets/maratsaratov/tiktok-data) | 760 (565 English) | OOD generalization test only |

**Dataset 1** contains `video_transcription_text` (actual spoken words), `claim_status`, `verified_status`, `author_ban_status`, and full engagement metrics — making it ideal for a multimodal NLP approach.

**Dataset 2** was never seen during training. It serves exclusively as an out-of-distribution test to measure how well the model generalizes to completely unseen data from a different source.

Both datasets are publicly available on Kaggle, ethically collected from TikTok's public content.

---

## Data Processing

### Cleaning steps (both notebooks)
- **Numeric coercion** — all engagement columns converted via `pd.to_numeric(errors='coerce').fillna(0)`
- **Missing text** — `video_transcription_text` nulls filled with empty string
- **Categorical encoding** — `verified_status` and `author_ban_status` label-encoded
- **Virality label** — `is_viral = 1` if `video_view_count > median`, else `0`. Median = **9,788 views**, giving a balanced 50/50 split
- **OOD Cyrillic filter** — Dataset 2 filtered via regex `[А-Яа-яЁё]` to keep English-only rows (760 → 565)
- **Stop word removal (transformer only)** — domain words like `'claim'`, `'viral'`, `'fyp'` removed via compiled regex to prevent the model shortcutting on dataset-specific artifacts

> ⚠️ `video_view_count` is **excluded from all features** — it directly defines the viral label and including it would be data leakage.

### Feature extraction

**Baseline (Random Forest)** — 7 structured engagement features:
`video_like_count`, `video_share_count`, `video_download_count`, `video_comment_count`, `video_duration_sec`, `verified_status` (encoded), `author_ban_status` (encoded)

**Main model (DistilBERT)** — multimodal features:

| Feature | Branch | Method |
|---|---|---|
| `video_transcription_text` | Text | DistilBERT tokenizer, max_length=128, stop words removed |
| `video_duration_sec` | Numeric | Standardized using training set mean/std |
| `verified_status` | Numeric | LabelEncoder |
| `author_ban_status` | Numeric | LabelEncoder |

---

## Modeling

### Baseline: Random Forest

Trained on 7 engagement features as a performance baseline. 80/20 stratified split, `n_estimators=100`, `max_depth=10`.

```
ROC-AUC : 0.9888  |  F1 : 0.97  |  Accuracy : 0.97
```

---

### Main Model: Multimodal DistilBERT Transformer

**Why DistilBERT:**
TikTok captions are short, informal, and context-dependent — "this is fire 🔥" signals virality, not combustion. DistilBERT understands semantic meaning in context and is 40% smaller and 60% faster than full BERT while retaining 97% of its performance — practical for Colab's GPU environment.

**Architecture:**
```
video_transcription_text ──► DistilBERT ──► [CLS] token (768-dim) ──┐
                                                                       ├──► concat ──► Dropout(0.3) ──► Linear ──► sigmoid
video_duration_sec, etc. ────► Linear(n→32) ──► BatchNorm ──► ReLU ──┘
```

Two branches — text (DistilBERT) and numeric metadata — fused at the classification head. Each branch specializes before combining signals.

**Training:**
- 80/20 stratified train/test split
- Loss: `BCEWithLogitsLoss`
- AdamW with **differential learning rates** (numeric layers = 1e-3, BERT weights protected to prevent catastrophic forgetting)
- Early stopping patience=3 on validation loss — stopped at epoch 3

**Training log:**
```
Epoch 1 | Train Loss: 0.0921 | Val Loss: 0.0606 | Val AUC: 0.9927
Epoch 2 | Train Loss: 0.0628 | Val Loss: 0.0617 | Val AUC: 0.9936  ← counter: 1/3
Epoch 3 | Train Loss: 0.0611 | Val Loss: 0.0604 | Val AUC: 0.9943  ← best checkpoint saved
Epoch 4 | Train Loss: 0.0572 | Val Loss: 0.0640 | Val AUC: 0.9931  ← counter: 1/3
Epoch 5 |                                                              ← counter: 2/3
→ Early stopping triggered. Best model restored from epoch 3.
```

**In-distribution test results:**
```
ROC-AUC : 0.9943  |  F1 : 0.99  |  Accuracy : 0.99

              precision  recall  f1-score  support
Not Viral         0.99    0.99      0.99     1968
Viral             0.99    0.99      0.99     1909
```

**Cross-dataset (OOD) results — Dataset 2 (never seen during training):**
```
ROC-AUC : 0.5238  |  Accuracy : 0.52  (565 English-only videos)
```

**Limitations:**
- OOD AUC drops to 0.52 because `claim_status` — the dominant predictor — is unavailable in Dataset 2
- `video_like_count` and `video_share_count` accumulate after posting; unavailable at true t=0 prediction time
- Dataset 1 is 98.2% "claim" content — model may underperform on dance or entertainment videos
- Future work: text-only prediction at posting time, temporal features, more diverse training data

---

## Visualizations

Both notebooks generate inline visualizations. Key figures:

| Figure | What it shows | Claim supported |
|---|---|---|
| Views distribution (raw + log) | Extreme skew in view counts | Justifies log transformation and median split |
| Engagement correlation heatmap | Pearson correlation of all features with virality | Likes/shares most correlated with viral label |
| Duration vs virality bar chart | Viral rate by video length bucket | Short videos go viral at higher rates |
| Share count boxplot | Log share counts — viral vs not viral | Clear class separation confirms shares as strong signal |
| Claim vs opinion pie chart | Claim/opinion breakdown in viral videos | 98.2% of viral videos are claim-type content |
| Random Forest feature importance | Top features ranked by importance | Likes and shares dominate over text features |
| DistilBERT training curves | Loss and AUC per epoch | Stable convergence, early stopping at epoch 3 |
| Confusion matrix | Predicted vs actual on test set | Near-perfect in-distribution classification |
| OOD probability distribution | Predicted probability KDE on Dataset 2 | Shows model uncertainty on unseen data |

> Interactive versions of the views distribution, correlation heatmap, and model comparison can be generated using Plotly — uncomment the Plotly cells at the end of `baseline.ipynb`.

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
| **DistilBERT (main)** | **0.9943** | **0.99** | **0.99** | Text + engagement, in-distribution |
| DistilBERT OOD | 0.5238 | 0.57 | 0.52 | Cross-dataset generalization |

### Key findings
1. **`claim_status` dominates** — 98.2% of viral videos are "claim" type. This single feature drives most of the predictive power
2. **Engagement signals are highly predictive** — `video_like_count` and `video_share_count` are the top features in the Random Forest importance ranking
3. **DistilBERT outperforms Random Forest** — AUC 0.9943 vs 0.9888, confirming semantic text understanding adds measurable signal
4. **OOD generalization fails** — performance collapses on Dataset 2, highlighting the challenge of building truly generalizable virality models

---

