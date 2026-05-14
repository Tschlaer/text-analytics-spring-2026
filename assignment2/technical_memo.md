# Technical Memo — IMDB Sentiment Classification

**TO:** Technical Manager
**FROM:** Thomas Schlaerth
**RE:** Movie-Review Sentiment Classifier — Deployment Recommendation
**DATE:** *[Submission Date]*

---

## Summary

I developed a binary text classifier that labels movie reviews as positive or negative using the **IMDB 50K dataset**. After testing **Multinomial Naive Bayes, Logistic Regression, and Linear SVM** with three vectorization schemes (TF-IDF unigrams, TF-IDF unigrams + bigrams, and Count Vectorizer with bigrams), I recommend **tuned Logistic Regression with TF-IDF unigrams + bigrams (5,000 features)** for production deployment.

## Why this model

The intended deployment scenario is a **movie recommender** that uses predicted-positive reviews to surface films viewers are likely to enjoy. In that scenario, the metric that matters most is **precision on the positive class** — a false positive (recommending a film that is actually disliked) erodes user trust in the recommender far more than a false negative (missing one good film among thousands). The goal is therefore to maximize true positives while minimizing false positives.

The recommended model achieves:

- **Test F1: 0.8886**
- Best-in-class true-positive rate among the three candidates
- One of the lowest false-positive rates of the three candidates
- Training time of **0.37 seconds** on commodity hardware

This outperforms Multinomial Naive Bayes by roughly **3 F1 points** while training in under half a second. Linear SVM matched the recommended model on accuracy but trained nearly three times slower (0.99 s vs. 0.37 s) and offers no probability scores out of the box — a real disadvantage if downstream services need calibrated confidence for ranking or thresholding.

The bigram features matter: switching from unigrams to unigrams + bigrams captured negation phrases ("not good", "never again") that flip sentiment polarity and that a unigram-only model treats as unrelated to the underlying sentiment.

## Key limitation

The model struggles with two specific patterns observed in both the held-out test set and the 20 custom inference examples:

1. **Mixed-sentiment reviews** — reviews that praise some elements while criticizing others (e.g., "the acting was wonderful, but the story made no sense"). The model collapses these to a single dominant signal and can pick the wrong side.
2. **Long synopsis-style reviews with little explicit opinion** — when a reviewer spends most of the text describing the plot rather than evaluating it, the model has little signal to work with and can be misled by incidental vocabulary or by real-world comparisons that bring negative-sounding words into an otherwise positive review.

In a small custom test of 20 examples, the model agreed with my judgment on 18; both misses were mixed-sentiment "tricky" cases.

## Recommendation

Deploy tuned Logistic Regression + TF-IDF (1,2)-grams for movie-review sentiment classification. Monitor positive-class precision in production traffic and re-evaluate quarterly. If mixed-sentiment reviews become a meaningful share of incoming traffic, consider introducing a third "mixed" classification or escalating to a transformer-based model (e.g., DistilBERT fine-tuned on our domain) that can capture sentence-level scope of opinion.
