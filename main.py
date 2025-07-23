import os
import warnings
warnings.filterwarnings("ignore")  # Suppress transformers and sklearn warnings

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
    DataCollatorWithPadding, EarlyStoppingCallback
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import pos_tag, ne_chunk, tree2conlltags
from nltk.tokenize import TreebankWordTokenizer
import string
import torch
from tqdm import tqdm

# -- For deterministic runs:
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

tokenizer_tb = TreebankWordTokenizer()

# === Feature Functions ===

sia = SentimentIntensityAnalyzer()
def sentiment_nltk(text):
    res = sia.polarity_scores(str(text))
    return res['compound']

def chars_between_commas(text):
    chunks = str(text).split(",")
    if len(chunks) < 2:
        return len(str(text))
    return np.mean([len(chunk) for chunk in chunks])

def count_unknown_symbols(text):
    symbols_known = string.ascii_letters + string.digits + string.punctuation
    return sum([not x in symbols_known for x in str(text)])

def get_persons(text):
    def bind_names(tagged_words):
        names, name = [], []
        for i, w in enumerate(tagged_words):
            if i == 0:
                continue
            if "PERSON" in w[2]:
                name.append(w[0])
            else:
                if len(name) != 0:
                    names.append(" ".join(name))
                name = []
        if len(name) != 0:
            names.append(" ".join(name))
        return names
    try:
        tokens = tokenizer_tb.tokenize(str(text))
        res_ne_tree = ne_chunk(pos_tag(tokens))
        res_ne = tree2conlltags(res_ne_tree)
        res_ne_list = [list(x) for x in res_ne]
        return bind_names(res_ne_list)
    except:
        return []

def count_person_names(text):
    try:
        return len(get_persons(text))
    except:
        return 0

def get_words(text):
    words = tokenizer_tb.tokenize(str(text))
    return [word for word in words if word not in string.punctuation]

def count_punctuation(text):
    return sum([x in string.punctuation for x in str(text)])

def count_capitalized_words(text):
    return sum([word.istitle() for word in get_words(text)])

def count_uppercase_words(text):
    return sum([word.isupper() for word in get_words(text)])

def first_word_len(text):
    words = get_words(text)
    if len(words) == 0:
        return 0
    return len(words[0])

def last_word_len(text):
    words = get_words(text)
    if len(words) == 0:
        return 0
    return len(words[-1])

def text_len(text):
    return len(str(text))

def word_count(text):
    return len(get_words(text))

def extract_features(df):
    print("[INFO] Extracting features from text...")
    features = pd.DataFrame(index=df.index)
    features['sentiment'] = df['text'].apply(sentiment_nltk)
    features['chars_between_commas'] = df['text'].apply(chars_between_commas)
    features['unknown_symbol_count'] = df['text'].apply(count_unknown_symbols)
    features['person_name_count'] = df['text'].apply(count_person_names)
    features['punctuation_count'] = df['text'].apply(count_punctuation)
    features['capitalized_count'] = df['text'].apply(count_capitalized_words)
    features['uppercase_count'] = df['text'].apply(count_uppercase_words)
    features['first_word_len'] = df['text'].apply(first_word_len)
    features['last_word_len'] = df['text'].apply(last_word_len)
    features['text_len'] = df['text'].apply(text_len)
    features['word_count'] = df['text'].apply(word_count)
    return features

print("Reading CSVs...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
author2label = {a: i for i, a in enumerate(sorted(train['author'].unique()))}
label2author = {i: a for a, i in author2label.items()}
train['label'] = train['author'].map(author2label)

print("Extracting features...")
X_train_feat = extract_features(train)
X_test_feat = extract_features(test)

# ==== RoBERTa Section ====
MODEL_NAME = "roberta-large"
MAX_LEN = 384
# For A100: use bf16, for most others: fp16, for old cards: fp32
USE_BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
USE_FP16 = torch.cuda.is_available() and not USE_BF16
BATCH_SIZE = 32 if USE_BF16 else 8  # Go higher for A100, lower for 3090/other

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
    import numpy as np
    logits, labels = eval_pred
    logits_tensor = torch.tensor(logits)
    # Clamp logits for numerical stability
    logits_tensor = torch.clamp(logits_tensor, -30, 30)
    probs = torch.nn.functional.softmax(logits_tensor, dim=-1).cpu().numpy()
    if not np.isfinite(probs).all():
        print("[WARNING] NaN or Inf in predictions. Skipping log_loss.")
        return {"log_loss": 99.99}
    return {"log_loss": log_loss(labels, probs)}

data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
NUM_FOLDS = 5
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
fold_splits = list(skf.split(train, train['label']))

oof_preds_roberta = np.zeros((len(train), len(author2label)))
test_preds_roberta_folds = np.zeros((NUM_FOLDS, len(test), len(author2label)))

print("Tokenizing test set for RoBERTa...")
test_encodings = preprocess(tokenizer, test)
test_dataset = SpookyDataset(test_encodings)

print(f"Training RoBERTa-large on {'A100 bf16' if USE_BF16 else 'GPU (fp16)' if USE_FP16 else 'CPU (fp32)'}...")

for fold, (train_idx, val_idx) in enumerate(fold_splits):
    print(f"\n==== RoBERTa Fold {fold+1}/{NUM_FOLDS} ====")
    train_fold = train.iloc[train_idx].reset_index(drop=True)
    val_fold = train.iloc[val_idx].reset_index(drop=True)
    train_enc = preprocess(tokenizer, train_fold)
    val_enc = preprocess(tokenizer, val_fold)
    train_dataset = SpookyDataset(train_enc, train_fold['label'].values)
    val_dataset = SpookyDataset(val_enc, val_fold['label'].values)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(author2label),
        torch_dtype=torch.bfloat16 if USE_BF16 else (torch.float16 if USE_FP16 else torch.float32),
    )
    training_args = TrainingArguments(
        output_dir=f"./results_fold{fold+1}",
        num_train_epochs=5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=f"./logs_fold{fold+1}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=USE_FP16,
        bf16=USE_BF16,
        dataloader_num_workers=4,
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
    val_logits = trainer.predict(val_dataset).predictions
    # Clamp again just in case
    val_probs = torch.nn.functional.softmax(torch.clamp(torch.tensor(val_logits), -30, 30), dim=-1).cpu().numpy()
    oof_preds_roberta[val_idx] = val_probs
    test_logits = trainer.predict(test_dataset).predictions
    test_probs = torch.nn.functional.softmax(torch.clamp(torch.tensor(test_logits), -30, 30), dim=-1).cpu().numpy()
    test_preds_roberta_folds[fold] = test_probs
    del model, trainer, train_dataset, val_dataset
    torch.cuda.empty_cache()

test_preds_roberta = np.mean(test_preds_roberta_folds, axis=0)
eps = 1e-15
test_preds_roberta = test_preds_roberta / test_preds_roberta.sum(axis=1, keepdims=True)
test_preds_roberta = np.clip(test_preds_roberta, eps, 1 - eps)
roberta_oof_logloss = log_loss(train['label'].values, oof_preds_roberta)
print(f"\nRoBERTa OOF Log Loss (CV estimate): {roberta_oof_logloss:.5f}")

# ========== Feature Model (LogisticRegression or LightGBM) ==========
print("Training feature-based Logistic Regression model with 5-fold CV...")
oof_preds_feat = np.zeros((len(train), len(author2label)))
test_preds_feat_folds = np.zeros((NUM_FOLDS, len(test), len(author2label)))
for fold, (train_idx, val_idx) in enumerate(fold_splits):
    print(f"Feature Model Fold {fold+1}/{NUM_FOLDS}")
    X_tr, y_tr = X_train_feat.iloc[train_idx], train['label'].iloc[train_idx]
    X_val, y_val = X_train_feat.iloc[val_idx], train['label'].iloc[val_idx]
    clf = LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial', solver='lbfgs')
    clf.fit(X_tr, y_tr)
    oof_preds_feat[val_idx] = clf.predict_proba(X_val)
    test_preds_feat_folds[fold] = clf.predict_proba(X_test_feat)
    del clf

test_preds_feat = np.mean(test_preds_feat_folds, axis=0)
test_preds_feat = test_preds_feat / test_preds_feat.sum(axis=1, keepdims=True)
test_preds_feat = np.clip(test_preds_feat, eps, 1 - eps)
feat_oof_logloss = log_loss(train['label'].values, oof_preds_feat)
print(f"Feature Model OOF Log Loss: {feat_oof_logloss:.5f}")

# ========== Ensemble ==========
print("Searching for optimal ensemble weights...")
best_w = 0.5
best_loss = float('inf')
for w in np.linspace(0, 1, 21):
    blended_oof = w * oof_preds_roberta + (1 - w) * oof_preds_feat
    loss = log_loss(train['label'].values, blended_oof)
    if loss < best_loss:
        best_loss = loss
        best_w = w
print(f"Best ensemble weight for RoBERTa = {best_w:.2f}, yielding OOF Log Loss = {best_loss:.5f}")

final_test_preds = best_w * test_preds_roberta + (1 - best_w) * test_preds_feat
final_test_preds = final_test_preds / final_test_preds.sum(axis=1, keepdims=True)
final_test_preds = np.clip(final_test_preds, eps, 1 - eps)

ensemble_oof = best_w * oof_preds_roberta + (1 - best_w) * oof_preds_feat
ensemble_oof_logloss = log_loss(train['label'].values, ensemble_oof)
print(f"\nEnsemble OOF Log Loss: {ensemble_oof_logloss:.5f}")

sub = pd.DataFrame(final_test_preds, columns=[label2author[i] for i in range(len(author2label))])
sub.insert(0, "id", test['id'])
sub.to_csv("submission.csv", index=False, float_format="%.12f")
print("submission.csv has been written with ensemble predictions.")

oof_df = pd.DataFrame(ensemble_oof, columns=[label2author[i] for i in range(len(author2label))])
oof_df["id"] = train["id"]
oof_df["true_author"] = train["author"]
oof_df.to_csv("oof_predictions_combined.csv", index=False, float_format="%.12f")
print("oof_predictions_combined.csv has been written with OOF probabilities and true labels.")
