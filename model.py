import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss

SEED = 42
FOLDS = 5
np.random.seed(SEED)

# Load merged data
train = pd.read_csv("train_combined.csv")
test = pd.read_csv("test.csv")
author2label = {a: i for i, a in enumerate(sorted(train["author"].unique()))}
label2author = {v: k for k, v in author2label.items()}
train["label"] = train["author"].map(author2label)
y = train["label"].values

# Simple TFIDF-LR baseline (char & word)
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 6), max_features=40000, sublinear_tf=True)
X = vec.fit_transform(train["text"])
X_test = vec.transform(test["text"])

# CV train
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros((len(train), 3))
test_pred = np.zeros((len(test), 3))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    model = LogisticRegression(solver='lbfgs', C=4, multi_class='multinomial', max_iter=400, random_state=SEED, n_jobs=-1)
    model.fit(X[tr_idx], y[tr_idx])
    oof[val_idx] = model.predict_proba(X[val_idx])
    test_pred += model.predict_proba(X_test) / FOLDS
    print(f"Fold {fold+1} log-loss: {log_loss(y[val_idx], oof[val_idx]):.5f}")

print("\n==== CV log-loss: {:.5f} ====".format(log_loss(y, oof)))

# Submission
sub = pd.DataFrame(test_pred, columns=[label2author[i] for i in range(3)])
sub.insert(0, "id", test["id"])
sub.to_csv("submission.csv", index=False, float_format="%.12f")
print("submission.csv written.")
