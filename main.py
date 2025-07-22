import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
    DataCollatorWithPadding, EarlyStoppingCallback
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier

# -------------------- SETUP -------------------- #
MODEL_LIST = [
    "roberta-large",
    "microsoft/deberta-v3-base",
    "google/electra-large-discriminator"
]
MODEL_ALIASES = [
    "roberta",
    "deberta",
    "electra"
]
NUM_FOLDS = 5
MAX_LEN = 512
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------- DATA LOAD -------------------- #
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
author2label = {a: i for i, a in enumerate(sorted(train['author'].unique()))}
label2author = {i: a for a, i in author2label.items()}
train['label'] = train['author'].map(author2label)

# -------------------- HANDCRAFTED FEATURE ENGINEERING -------------------- #
def make_handcrafted_features(df):
    return pd.DataFrame({
        'text_len': df['text'].str.len(),
        'num_words': df['text'].str.split().apply(len),
        'num_capitals': df['text'].apply(lambda x: sum(1 for c in x if c.isupper())),
        'num_exclaims': df['text'].str.count('!'),
        'num_questions': df['text'].str.count(r'\?'),
        'num_punct': df['text'].str.count(r'[,.!?:;]'),
        # Add more features as desired!
    })

train_feats = make_handcrafted_features(train)
test_feats = make_handcrafted_features(test)

# Add TF-IDF features
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100)
tfidf_train = tfidf.fit_transform(train['text'])
tfidf_test = tfidf.transform(test['text'])
from scipy.sparse import hstack
train_feats_full = hstack([train_feats, tfidf_train])
test_feats_full = hstack([test_feats, tfidf_test])

# -------------------- KFOLD INIT -------------------- #
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

# -------------------- TRANSFORMER ENSEMBLE -------------------- #
def get_trainer_and_dataset(model_name, tokenizer, train_df, val_df):
    def preprocess(tokenizer, df):
        return tokenizer(
            df["text"].tolist(),
            truncation=True,
            padding=False,
            max_length=MAX_LEN,
            return_tensors=None
        )
    class SpookyDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels=None):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            if self.labels is not None:
                item["labels"] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.encodings["input_ids"])
    # Tokenize
    train_encodings = preprocess(tokenizer, train_df)
    val_encodings = preprocess(tokenizer, val_df)
    train_dataset = SpookyDataset(train_encodings, train_df['label'].values)
    val_dataset = SpookyDataset(val_encodings, val_df['label'].values)
    return train_dataset, val_dataset

def transformer_fold_predict(model_name, train_df, val_df, test_df, fold, num_labels=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, val_dataset = get_trainer_and_dataset(model_name, tokenizer, train_df, val_df)
    test_encodings = tokenizer(
        test_df["text"].tolist(),
        truncation=True,
        padding=False,
        max_length=MAX_LEN,
        return_tensors=None
    )
    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            return item
        def __len__(self):
            return len(self.encodings["input_ids"])
    test_dataset = TestDataset(test_encodings)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    training_args = TrainingArguments(
        output_dir=f"./results_{model_name.replace('/', '-')}_fold{fold + 1}",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=f"./logs_{model_name.replace('/', '-')}_fold{fold + 1}",
        evaluation_strategy="epoch",      # FIX: use both set to 'epoch'
        save_strategy="epoch",            # FIX: use both set to 'epoch'
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        seed=SEED + fold,
        report_to="none"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
        return {"log_loss": log_loss(labels, probs)}
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.0)],
    )
    trainer.train()

    val_logits = trainer.predict(val_dataset).predictions
    val_probs = torch.nn.functional.softmax(torch.tensor(val_logits), dim=-1).numpy()
    test_logits = trainer.predict(test_dataset).predictions
    test_probs = torch.nn.functional.softmax(torch.tensor(test_logits), dim=-1).numpy()
    return val_probs, test_probs

# -------------------- MAIN LOOP: ENSEMBLE TRAINING -------------------- #
oof_preds = {alias: np.zeros((len(train), 3)) for alias in MODEL_ALIASES}
test_preds = {alias: np.zeros((NUM_FOLDS, len(test), 3)) for alias in MODEL_ALIASES}
oof_preds['lgbm'] = np.zeros((len(train), 3))
test_preds['lgbm'] = np.zeros((NUM_FOLDS, len(test), 3))

for fold, (train_idx, val_idx) in enumerate(skf.split(train, train['label'])):
    print(f"\n==== Fold {fold+1}/{NUM_FOLDS} ====")
    train_fold = train.iloc[train_idx].reset_index(drop=True)
    val_fold = train.iloc[val_idx].reset_index(drop=True)

    # Transformers
    for model_name, alias in zip(MODEL_LIST, MODEL_ALIASES):
        print(f"Training {alias.upper()}...")
        val_probs, test_probs = transformer_fold_predict(model_name, train_fold, val_fold, test, fold)
        oof_preds[alias][val_idx] = val_probs
        test_preds[alias][fold] = test_probs

    # LightGBM
    print("Training LightGBM...")
    lgbm = LGBMClassifier(
        objective='multiclass',
        n_estimators=400,
        learning_rate=0.08,
        random_state=SEED+fold,
        class_weight='balanced'
    )
    lgbm.fit(
        train_feats_full[train_idx], train['label'].values[train_idx],
        eval_set=[(train_feats_full[val_idx], train['label'].values[val_idx])],
        early_stopping_rounds=30,
        verbose=False
    )
    oof_preds['lgbm'][val_idx] = lgbm.predict_proba(train_feats_full[val_idx])
    test_preds['lgbm'][fold] = lgbm.predict_proba(test_feats_full)

# -------------------- ENSEMBLING: Probability Averaging -------------------- #
print("\nAveraging ensemble probabilities...")
weights = {alias: 1 for alias in MODEL_ALIASES}
weights['lgbm'] = 1

oof_final = sum(w * oof_preds[alias] for alias, w in weights.items()) / sum(weights.values())
oof_logloss = log_loss(train['label'].values, oof_final)
print(f"==== OOF LOGLOSS (Ensemble): {oof_logloss:.5f} ====")

test_final = sum(w * np.mean(test_preds[alias], axis=0) for alias, w in weights.items()) / sum(weights.values())
eps = 1e-15
test_final = test_final / test_final.sum(axis=1, keepdims=True)
test_final = np.clip(test_final, eps, 1 - eps)

# -------------------- SUBMISSION -------------------- #
sub = pd.DataFrame(test_final, columns=[label2author[i] for i in range(3)])
sub.insert(0, "id", test['id'])
sub.to_csv("submission_ensemble.csv", index=False, float_format="%.12f")
print("submission_ensemble.csv written.")

oof_df = pd.DataFrame(oof_final, columns=[label2author[i] for i in range(3)])
oof_df["id"] = train["id"]
oof_df["true_label"] = train["author"]
oof_df.to_csv("oof_predictions_ensemble.csv", index=False)
print("oof_predictions_ensemble.csv written.")
