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
from collections import defaultdict

# -------------------- SETUP -------------------- #
MODEL_NAME = "roberta-large"
MAX_LEN = 512
NUM_FOLDS = 5
AUG_FRAC = 0.25  # Fraction of training samples to augment (change to taste)
AUG_METHODS = ['back_translation', 'eda']  # Choose: 'back_translation', 'eda', or both
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# -------------------- DATA LOAD -------------------- #
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
author2label = {a: i for i, a in enumerate(sorted(train['author'].unique()))}
label2author = {i: a for a, i in author2label.items()}
train['label'] = train['author'].map(author2label)

# -------------------- ARGOS TRANSLATE SETUP -------------------- #
try:
    import argostranslate.package, argostranslate.translate
    # Download and install models if not present
    def setup_argos():
        pkgs = [
            "https://github.com/argosopentech/argos-translate/releases/download/v2.5.0/en_fr.argosmodel",
            "https://github.com/argosopentech/argos-translate/releases/download/v2.5.0/fr_en.argosmodel",
        ]
        for pkg_url in pkgs:
            fname = pkg_url.split('/')[-1]
            if not os.path.exists(os.path.expanduser(f"~/.local/share/argos-translate/packages/{fname}")):
                print(f"Downloading and installing Argos Translate model: {fname}")
                argostranslate.package.install_from_path(
                    argostranslate.package.download_package(pkg_url)
                )
    setup_argos()
    _ARGOS_READY = True
except Exception as e:
    print("Argos Translate not available or failed to install:", e)
    _ARGOS_READY = False

def back_translate_argos(text, src_lang='en', pivot_lang='fr'):
    if not _ARGOS_READY:
        return text
    try:
        translations = argostranslate.translate.get_installed_languages()
        src = [l for l in translations if l.code == src_lang][0]
        pivot = [l for l in translations if l.code == pivot_lang][0]
        text_pivot = src.get_translation(pivot).translate(text)
        text_back = pivot.get_translation(src).translate(text_pivot)
        return text_back
    except Exception as e:
        print(f"Argos back-translation failed: {e}")
        return text

# -------------------- SIMPLE EDA: SYNONYM REPLACE -------------------- #
from nltk.corpus import wordnet
import nltk
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def synonym_replacement(text, n=1):
    words = text.split()
    new_words = words.copy()
    for _ in range(n):
        idxs = [i for i, w in enumerate(new_words) if len(wordnet.synsets(w)) > 0]
        if not idxs: break
        idx = random.choice(idxs)
        syns = wordnet.synsets(new_words[idx])
        lemmas = set([l.name().replace('_', ' ') for syn in syns for l in syn.lemmas()])
        lemmas.discard(new_words[idx])
        if lemmas:
            new_words[idx] = random.choice(list(lemmas))
    return ' '.join(new_words)

def eda_augment(text):
    # You can expand with insertion, swap, delete, etc.
    return synonym_replacement(text, n=1)

# -------------------- AUGMENT TRAIN DATA -------------------- #
def augment_train_df(df, frac=AUG_FRAC, methods=AUG_METHODS):
    n = int(len(df) * frac)
    rows = df.sample(n=n, random_state=42).reset_index(drop=True)
    aug_texts, aug_labels = [], []
    for i, row in tqdm(rows.iterrows(), total=len(rows), desc="Augmenting"):
        text, label = row['text'], row['label']
        if 'back_translation' in methods:
            text_bt = back_translate_argos(text)
            aug_texts.append(text_bt)
            aug_labels.append(label)
        if 'eda' in methods:
            text_eda = eda_augment(text)
            aug_texts.append(text_eda)
            aug_labels.append(label)
    df_aug = pd.DataFrame({'text': aug_texts, 'label': aug_labels})
    df_combined = pd.concat([df, df_aug], axis=0, ignore_index=True)
    print(f"Original: {len(df)}, After Augmentation: {len(df_combined)}")
    return df_combined

# Augment training data
train_aug = augment_train_df(train[['text', 'label']], frac=AUG_FRAC, methods=AUG_METHODS)

# -------------------- TOKENIZATION -------------------- #
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    return {"log_loss": log_loss(labels, probs)}

# Data Collator
data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

# -------------------- KFOLD ROBERTA -------------------- #
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train), 3))
test_preds = np.zeros((len(test), 3))
train_aug = train_aug.reset_index(drop=True)
for fold, (train_idx, val_idx) in enumerate(skf.split(train_aug, train_aug['label'])):
    print(f"\n==== Fold {fold+1}/{NUM_FOLDS} ====")
    train_fold = train_aug.iloc[train_idx].reset_index(drop=True)
    val_fold = train_aug.iloc[val_idx].reset_index(drop=True)

    train_encodings = preprocess(tokenizer, train_fold)
    val_encodings = preprocess(tokenizer, val_fold)

    train_dataset = SpookyDataset(train_encodings, train_fold['label'].values)
    val_dataset = SpookyDataset(val_encodings, val_fold['label'].values)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )

    training_args = TrainingArguments(
        output_dir=f"./results_fold{fold + 1}",
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=f"./logs_fold{fold + 1}",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        seed=42 + fold,
        report_to="none"
    )

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

    # Evaluate on original fold (not augmented fold)
    val_orig = train.iloc[val_idx].reset_index(drop=True)
    val_orig_enc = preprocess(tokenizer, val_orig)
    val_orig_dataset = SpookyDataset(val_orig_enc, val_orig['label'].values)
    val_logits = trainer.predict(val_orig_dataset).predictions
    val_probs = torch.nn.functional.softmax(torch.tensor(val_logits), dim=-1).numpy()
    oof_preds[val_idx] = val_probs

    if fold == 0:
        test_encodings = preprocess(tokenizer, test)
        test_dataset = SpookyDataset(test_encodings)
        test_logits_all = np.zeros((NUM_FOLDS, len(test), 3))
    test_logits = trainer.predict(test_dataset).predictions
    test_probs = torch.nn.functional.softmax(torch.tensor(test_logits), dim=-1).numpy()
    test_logits_all[fold] = test_probs

# Average test predictions over folds
test_preds = np.mean(test_logits_all, axis=0)
eps = 1e-15
test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)
test_preds = np.clip(test_preds, eps, 1 - eps)
oof_logloss = log_loss(train['label'].values, oof_preds)
print(f"\n==== OOF LOGLOSS (CV estimate): {oof_logloss:.5f} ====")

# Submission
sub = pd.DataFrame(test_preds, columns=[label2author[i] for i in range(3)])
sub.insert(0, "id", test['id'])
sub.to_csv("submission.csv", index=False, float_format="%.12f")
print("submission.csv written.")

oof_df = pd.DataFrame(oof_preds, columns=[label2author[i] for i in range(3)])
oof_df["id"] = train["id"]
oof_df["true_label"] = train["author"]
oof_df.to_csv("oof_predictions.csv", index=False)
