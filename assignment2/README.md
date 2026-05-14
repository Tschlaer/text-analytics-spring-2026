# Assignment 2 — Text Classification (IMDB Movie Reviews)

**Student:** Thomas Schlaerth
**Date:** *[Submission Date]*
**Course:** Text Analytics — Spring 2026

## Overview

Binary sentiment classification of IMDB movie reviews. Each review is labeled `positive` or `negative`; the goal is a supervised classifier built from scratch (TF-IDF features + classical ML), trained, tuned, evaluated, and tested on 20 custom-written examples.

## Dataset

- **Source:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- **Size:** 50,000 reviews (25,000 positive / 25,000 negative)
- **Task type:** Binary classification
- **Balance:** Perfectly balanced (1:1)
- **Loading:** Downloaded directly in the notebook via `kagglehub`

## Repository layout

```
assignment2/
├── README.md                     ← this file
├── notebooks/
│   └── classification.ipynb      ← main notebook
├── data/
│   └── README.md                 ← dataset is pulled at runtime via kagglehub
├── docs/
│   ├── AI_usage_log.md           ← prompts used + learning progression
│   ├── reflection.md             ← 5 reflection questions
│   └── technical_memo.md         ← 1-page memo (convert to PDF for submission)
├── figures/                      ← auto-generated plots and CSVs
└── models/                       ← saved best model + vectorizer
```

## How to reproduce

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib jupyter kagglehub openpyxl
jupyter notebook notebooks/classification.ipynb     # run all cells
```

The notebook was developed in Google Colab and uses `kagglehub` to fetch the dataset at runtime, so no manual download is required.

## Best model results

**Tuned Logistic Regression** with **TF-IDF (unigrams + bigrams, 5,000 features)** and `class_weight='balanced'`. The regularization strength `C` was tuned via 3-fold cross-validated grid search over `{0.1, 0.5, 1.0, 2.0, 4.0}` using F1 as the scoring metric.

| Metric | Value |
|---|---|
| Best model | Logistic Regression (tuned) + TF-IDF (1,2)-grams |
| Test F1 | **0.8886** |
| Test accuracy | *[fill from classification report]* |
| Test precision | *[fill from classification report]* |
| Test recall | *[fill from classification report]* |
| Training time | 0.370 s |
| Vocabulary size | 5,000 features |

Performance threshold (F1 ≥ 0.70) is comfortably met.

## Important class

The **positive** class matters more for this dataset under a movie-recommender deployment scenario. The downstream product would be surfacing well-reviewed movies to viewers, so a false positive (recommending a movie that is actually disliked) costs user trust more than a false negative (missing one good movie among many). The objective is to maximize true positives while minimizing false positives — i.e., precision on the positive class.

Under that lens, the Tuned Logistic Regression model wins: it has the highest F1 score and one of the lowest false-positive rates among the three candidates.

## Model comparison (5 criteria)

| Criterion | LogReg (tuned) | MultinomialNB | LinearSVM |
|---|---|---|---|
| **1. F1 score** | **0.8886** | 0.8594 | 0.8859 |
| **2. Training speed** | 0.370 s | 0.037 s (fastest) | 0.989 s |
| **3. Performance on important class (positive)** | Best | Worst | Better |
| **4. Interpretability** | High (per-word coefficients) | High (per-word log-probabilities) | High (per-word coefficients) |
| **5. Robustness to class imbalance** | `class_weight='balanced'` | No weighting (relies on prior) | `class_weight='balanced'` |

**Winner:** Tuned Logistic Regression — highest F1 and accuracy with one of the lowest false-positive rates. Naive Bayes is dramatically faster to train but pays for it with a ~3-point F1 deficit. Linear SVM is essentially tied with LogReg on accuracy but trains nearly 3× slower.

## Custom inference summary

The model was tested on 20 newly written examples spanning three difficulty buckets:

| Bucket | Examples | Model correct |
|---|---|---|
| Easy (clear sentiment) | 10 | *[fill in from notebook output]* |
| Tricky (sarcasm, mixed, negation, comparison) | 5 | *[fill in]* |
| Out-of-domain (game, music, book, TV, restaurant) | 5 | *[fill in]* |
| **Total** | **20** | **18 / 20** |

**Key findings** (full discussion in `docs/reflection.md`):
- The model handled out-of-domain reviews well — positive/negative sentiment vocabulary transfers across entertainment domains.
- The misses were on **mixed-sentiment** "tricky" examples where the reviewer praised some elements while criticizing others.
- This matches the error pattern observed in the 20-row manual review on real IMDB test data, where long synopsis-style reviews with little explicit opinion (and reviews containing real-world comparisons) were also harder.

## Final recommendation

**Deploy Tuned Logistic Regression with TF-IDF (1,2)-grams** for movie-review sentiment classification. It delivers the best F1 (0.8886) and the best performance on the positive class, with sub-second training. Mixed-sentiment reviews and long plot-summary reviews are the known failure mode; for production, monitor the positive-class precision and consider a third "mixed" class if such reviews are common in incoming traffic.

## AI Usage (Tier 2)

This assignment was completed under Tier 2 restrictions:
- AI (Claude) was used for **implementation** (notebook scaffolding, vectorizer/model code, debugging) and for **generating the 20 custom inference examples** (Step 7), both of which are explicitly permitted.
- **Error analysis, model selection reasoning, the important-class identification, and the custom-inference reflection are my own work**, written without AI generation.
- See `docs/AI_usage_log.md` for prompts and learning progression.
