import os
import pandas as pd, numpy as np, torch
from datasets import load_dataset, Dataset
from transformers import (AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling,
                          AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# === SETTINGS ===
MODEL_NAME = "roberta-large"
UNLABELED_FILE = "unlabeled_pretraining_corpus.txt"
TRAIN_CSVS = ["train.csv", "auxiliary_labeled.csv", "train_backtranslated.csv", "train_eda.csv"]
TEST_CSV = "test.csv"
MLM_DIR = "./mlm_spooky"
AUG_CSV = "train_augmented.csv"
BATCH_SIZE = 64  # Fit for A100
MAX_LEN = 128
FOLDS = 5
EPOCHS = 4
SEED = 42
os.makedirs("plots", exist_ok=True)

# === STEP 1: DOMAIN-ADAPTIVE MLM PRETRAINING ===
print("\n==== [1] DOMAIN-ADAPTIVE MLM PRETRAINING ====")
if not os.path.exists(MLM_DIR):
    print("-> Starting MLM pretraining on spooky unlabeled corpus...")
    dataset = load_dataset('text', data_files=UNLABELED_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    def tokenize(batch): return tokenizer(batch['text'], truncation=True, max_length=128)
    tokenized = dataset.map(tokenize, batched=True, num_proc=4, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    args = TrainingArguments(
        output_dir=MLM_DIR,
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=BATCH_SIZE,
        save_steps=5000,
        learning_rate=5e-5,
        prediction_loss_only=True,
        bf16=True,   # A100
        fp16=False,
        report_to="none"
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized['train'], data_collator=collator)
    trainer.train()
    trainer.save_model(MLM_DIR)
    print("MLM pretraining done!\n")
else:
    print("-> MLM checkpoint already exists, skipping...")

# === STEP 2: DATA AUGMENTATION & COMBINATION ===
print("\n==== [2] DATA AUGMENTATION & COMBINE ====")
# NOTE: If you already have the CSVs, this just concatenates.
dfs = [pd.read_csv(f) for f in TRAIN_CSVS if os.path.exists(f)]
combined = pd.concat(dfs, ignore_index=True)
combined.to_csv(AUG_CSV, index=False)
print(f"Combined data written to {AUG_CSV}: {combined.shape}")

# Plot class distribution
plt.figure(figsize=(6,3))
sns.countplot(y=combined["author"])
plt.title("Author Distribution in Augmented Train")
plt.tight_layout()
plt.savefig("plots/author_distribution.png")
plt.close()

# Text length distribution
combined['text_len'] = combined['text'].str.len()
plt.figure(figsize=(6,4))
sns.histplot(combined['text_len'], bins=40, kde=True)
plt.title("Text Length Distribution (Augmented Data)")
plt.tight_layout()
plt.savefig("plots/text_length_distribution.png")
plt.close()

# === STEP 3: CROSS-VALIDATED FINETUNING (with PLOTS) ===
print("\n==== [3] CROSS-VALIDATED FINETUNING (ROBERTA) ====")
train = pd.read_csv(AUG_CSV)
test = pd.read_csv(TEST_CSV)
author2label = {a: i for i, a in enumerate(sorted(train["author"].unique()))}
label2author = {i: a for a, i in author2label.items()}
train["label"] = train["author"].map(author2label)
tokenizer = AutoTokenizer.from_pretrained(MLM_DIR, use_fast=True)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

oof_preds = np.zeros((len(train), 3))
test_preds = np.zeros((len(test), 3))
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
fold_logloss = []
all_cms = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(train, train["label"])):
    print(f"\n=== Fold {fold+1}/{FOLDS} ===")
    train_data = Dataset.from_pandas(train.iloc[tr_idx][["text", "label"]].reset_index(drop=True)).map(tokenize, batched=True)
    val_data = Dataset.from_pandas(train.iloc[val_idx][["text", "label"]].reset_index(drop=True)).map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(MLM_DIR, num_labels=3)
    args = TrainingArguments(
        output_dir=f"./finetune_fold{fold+1}",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"./logs_fold{fold+1}",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,  # For A100
        fp16=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        seed=SEED+fold,
        report_to="none"
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_data, eval_dataset=val_data,
        compute_metrics=None, callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer.train()
    val_logits = trainer.predict(val_data).predictions
    val_probs = torch.nn.functional.softmax(torch.tensor(val_logits), dim=-1).numpy()
    oof_preds[val_idx] = val_probs
    y_val = train.iloc[val_idx]["label"].values
    ll = log_loss(y_val, val_probs)
    fold_logloss.append(ll)
    print(f"Fold {fold+1} log-loss: {ll:.5f}")

    # Confusion matrix
    y_pred = val_probs.argmax(axis=1)
    cm = confusion_matrix(y_val, y_pred)
    all_cms.append(cm)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(author2label.keys()), yticklabels=list(author2label.keys()))
    plt.title(f"Confusion Matrix Fold {fold+1}")
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(f"plots/cm_fold{fold+1}.png")
    plt.close()

    if fold == 0:
        test_data = Dataset.from_pandas(test[["text"]]).map(tokenize, batched=True)
        test_logits_all = np.zeros((FOLDS, len(test), 3))
    test_logits = trainer.predict(test_data).predictions
    test_probs = torch.nn.functional.softmax(torch.tensor(test_logits), dim=-1).numpy()
    test_logits_all[fold] = test_probs

# Aggregate results
test_preds = test_logits_all.mean(axis=0)
oof_ll = log_loss(train['label'], oof_preds)
print(f"\n==== Mean CV log-loss: {np.mean(fold_logloss):.5f}, OOF log-loss: {oof_ll:.5f} ====")

# Save summary plots
plt.figure(figsize=(7,4))
plt.plot(np.arange(1,FOLDS+1), fold_logloss, marker="o")
plt.title("CV Log-Loss per Fold")
plt.xlabel("Fold")
plt.ylabel("Log-Loss")
plt.tight_layout()
plt.savefig("plots/cv_logloss_per_fold.png")
plt.close()

# Optionally, feature importance/embeddings can be visualized with additional code

# Output test preds
sub = pd.DataFrame(test_preds, columns=[label2author[i] for i in range(3)])
sub.insert(0, "id", test["id"])
sub.to_csv("submission_roberta.csv", index=False, float_format="%.12f")
print("submission_roberta.csv written.")

# Summary classification report (on all OOF)
y_true = train['label'].values
y_pred = oof_preds.argmax(axis=1)
report = classification_report(y_true, y_pred, target_names=[label2author[i] for i in range(3)])
print("\n==== Classification Report (OOF, all folds) ====")
print(report)
with open("plots/classification_report.txt", "w") as f:
    f.write(report)
