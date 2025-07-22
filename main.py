import os
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
    DataCollatorWithPadding, EarlyStoppingCallback
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import torch

# ---- NEW: Set deterministic seeds globally ----
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 1. Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
author2label = {a: i for i, a in enumerate(sorted(train['author'].unique()))}
label2author = {i: a for a, i in author2label.items()}
train['label'] = train['author'].map(author2label)

MODEL_NAME = "roberta-large"
MAX_LEN = 384
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(tokenizer, df):
    return tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=False,       # for dynamic padding
        max_length=MAX_LEN,
        return_tensors=None  # tensors in collator
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

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    return {"log_loss": log_loss(labels, probs)}

data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

NUM_FOLDS = 5
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros((len(train), 3))
test_preds = np.zeros((len(test), 3))

for fold, (train_idx, val_idx) in enumerate(skf.split(train, train['label'])):
    print(f"\n==== Fold {fold+1}/{NUM_FOLDS} ====")
    # ---- NEW: Set per-fold seeds (superstition, but helps stability) ----
    np.random.seed(SEED + fold)
    torch.manual_seed(SEED + fold)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + fold)

    train_fold = train.iloc[train_idx].reset_index(drop=True)
    val_fold = train.iloc[val_idx].reset_index(drop=True)

    train_encodings = preprocess(tokenizer, train_fold)
    val_encodings = preprocess(tokenizer, val_fold)

    train_dataset = SpookyDataset(train_encodings, train_fold['label'].values)
    val_dataset = SpookyDataset(val_encodings, val_fold['label'].values)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )
    # ---- NEW: Enable gradient checkpointing ----
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=f"./results_fold{fold + 1}",
        num_train_epochs=6,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        # ---- TUNE weight decay ----
        weight_decay=0.05,   # try 0.05 (or 0.1 if needed)
        # ---- LABEL SMOOTHING ----
        label_smoothing_factor=0.1,
        # ---- LR SCHEDULER WARMUP ----
        warmup_ratio=0.1,
        logging_dir=f"./logs_fold{fold + 1}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        eval_accumulation_steps=2,   # ---- for OOM safety
        fp16=torch.cuda.is_available(),
        seed=SEED + fold,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2,
