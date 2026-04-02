# Kaggle NLP Competition — Author Identification

**Top 5 of 390** in a Kaggle NLP competition for author identification using transformer-based models.

## Approach

- **Model:** DeBERTa-v3-large (microsoft/deberta-v3-large)
- **Strategy:** Stratified K-Fold cross-validation (6 folds)
- **Tokenization:** Max length 512, padded for GPU throughput
- **Training:** HuggingFace Transformers + Trainer API
- **Hardware:** Multi-GPU training with CUDA optimization

## Key Techniques

- Stratified K-Fold to prevent data leakage across author classes
- Ensemble predictions across folds for robust generalization
- Hyperparameter tuning for learning rate, batch size, and epochs
- Feature engineering on text length, vocabulary richness, and stylistic markers

## Tech Stack

- Python, PyTorch, HuggingFace Transformers
- Pandas, NumPy, Scikit-learn
- CUDA / GPU-accelerated training

## Results

Placed **Top 5 out of 390 participants** through strong feature engineering and cross-validation strategy.

## Dataset

- `train.csv` — Labeled text samples from 3 authors (EAP, HPL, MWS)
- `test.csv` — Unlabeled samples for prediction
- `auxiliary_labeled.csv` — Additional labeled data

