import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# -------------------- DATA LOAD -------------------- #
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
author2label = {a: i for i, a in enumerate(sorted(train['author'].unique()))}
label2author = {i: a for a, i in author2label.items()}
train['label'] = train['author'].map(author2label)

# -------------------- TF-IDF FEATURES -------------------- #
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=10000)
tfidf_train = tfidf.fit_transform(train['text'])
tfidf_test = tfidf.transform(test['text'])

# -------------------- KFOLD LOGREG -------------------- #
NUM_FOLDS = 5
SEED = 42
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

oof_preds = np.zeros((len(train), 3))
test_preds = np.zeros((len(test), 3))

for fold, (train_idx, val_idx) in enumerate(skf.split(train, train['label'])):
    print(f"\n==== Fold {fold + 1}/{NUM_FOLDS} ====")
    X_train, X_val = tfidf_train[train_idx], tfidf_train[val_idx]
    y_train, y_val = train['label'].values[train_idx], train['label'].values[val_idx]

    clf = LogisticRegression(C=2.0, max_iter=200, multi_class='multinomial', solver='lbfgs', random_state=SEED + fold)
    clf.fit(X_train, y_train)

    oof_preds[val_idx] = clf.predict_proba(X_val)
    test_preds += clf.predict_proba(tfidf_test) / NUM_FOLDS

oof_logloss = log_loss(train['label'].values, oof_preds)
print(f"\n==== OOF LOGLOSS (LogReg+TFIDF): {oof_logloss:.5f} ====")

# -------------------- SUBMISSION -------------------- #
sub = pd.DataFrame(test_preds, columns=[label2author[i] for i in range(3)])
sub.insert(0, "id", test['id'])
sub.to_csv("submission_logreg_tfidf.csv", index=False, float_format="%.12f")
print("submission_logreg_tfidf.csv written.")

oof_df = pd.DataFrame(oof_preds, columns=[label2author[i] for i in range(3)])
oof_df["id"] = train["id"]
oof_df["true_label"] = train["author"]
oof_df.to_csv("oof_predictions_logreg.csv", index=False)
print("oof_predictions_logreg.csv written.")
