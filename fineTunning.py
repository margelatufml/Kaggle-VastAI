import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import torch

# Sanity check: print out GPU info at start
print("CUDA available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
    print("Num GPUs:", torch.cuda.device_count())

MODEL_PATH = "./mlm_spooky"  # Where MLM pretraining saved
FOLDS = 5
BATCH_SIZE = 64   # <-- Try 32 or 64, you have 80GB VRAM! (set lower if OOM)
MAX_LEN = 384

# 1. Load labeled data (combine Kaggle + aux labeled)
train = pd.read_csv("train_combined.csv")
test = pd.read_csv("test.csv")
author2label = {a: i for i, a in enumerate(sorted(train["author"].unique()))}
label2author = {i: a for a, i in author2label.items()}
train["label"] = train["author"].map(author2label)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

oof_preds = np.zeros((len(train), 3))
test_preds = np.zeros((len(test), 3))
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

for fold, (tr_idx, val_idx) in enumerate(skf.split(train, train["label"])):
    print(f"\n=== Fold {fold+1}/{FOLDS} ===")
    train_data = Dataset.from_pandas(train.iloc[tr_idx][["text", "label"]].reset_index(drop=True))
    val_data = Dataset.from_pandas(train.iloc[val_idx][["text", "label"]].reset_index(drop=True))
    train_data = train_data.map(tokenize, batched=True)
    val_data = val_data.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)

    args = TrainingArguments(
        output_dir=f"./finetune_fold{fold+1}",
        num_train_epochs=3,       
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"./logs_fold{fold+1}",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=False,                  # << ALWAYS True for A100!
        bf16=True,                # << UNCOMMENT if you want bf16 (optional, if model supports)
        seed=42+fold,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    val_logits = trainer.predict(val_data).predictions
    val_probs = torch.nn.functional.softmax(torch.tensor(val_logits), dim=-1).numpy()
    oof_preds[val_idx] = val_probs

    # Predict test set on each fold and average
    if fold == 0:
        test_data = Dataset.from_pandas(test[["text"]])
        test_data = test_data.map(tokenize, batched=True)
        test_logits_all = np.zeros((FOLDS, len(test), 3))
    test_logits = trainer.predict(test_data).predictions
    test_probs = torch.nn.functional.softmax(torch.tensor(test_logits), dim=-1).numpy()
    test_logits_all[fold] = test_probs

# Average test preds
test_preds = test_logits_all.mean(axis=0)
print(f"\nOOF log-loss: {log_loss(train['label'], oof_preds):.5f}")

sub = pd.DataFrame(test_preds, columns=[label2author[i] for i in range(3)])
sub.insert(0, "id", test["id"])
sub.to_csv("submission.csv", index=False, float_format="%.12f")
print("submission.csv written.")
