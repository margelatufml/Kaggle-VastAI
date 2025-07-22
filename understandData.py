import pandas as pd
import numpy as np
from collections import Counter
import string

# 1. LOAD DATA
df = pd.read_csv('train.csv')
print("\n=== First few rows ===")
print(df.head())
print("\n=== Info ===")
print(df.info())
print("\n=== Describe ===")
print(df.describe(include='all'))

# 2. AUTHOR DISTRIBUTION
print('\n=== Author distribution ===')
print(df['author'].value_counts(normalize=False))
print('\n=== Author distribution (%) ===')
print(df['author'].value_counts(normalize=True) * 100)

# 3. TEXT LENGTH ANALYSIS
df['char_count'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
df['sentence_count'] = df['text'].apply(lambda x: x.count('.') + x.count('!') + x.count('?'))
print('\n=== Text length stats by author ===')
print(df.groupby('author')[['char_count', 'word_count', 'sentence_count']].describe().T)

# 4. AVERAGE SENTENCE LENGTH
print('\n=== Average sentence length (words) by author ===')
print(df.groupby('author')['word_count'].mean())

# 5. LEXICAL DIVERSITY & UNIQUE WORDS
df['unique_word_count'] = df['text'].apply(lambda x: len(set(x.lower().translate(str.maketrans('', '', string.punctuation)).split())))
df['lexical_diversity'] = df['unique_word_count'] / df['word_count']
print('\n=== Lexical diversity by author ===')
print(df.groupby('author')['lexical_diversity'].mean())

# 6. MOST COMMON WORDS
def get_top_words(texts, n=20):
    words = ' '.join(texts).lower().translate(str.maketrans('', '', string.punctuation)).split()
    return Counter(words).most_common(n)

for author in df['author'].unique():
    print(f"\n=== Top 20 words for {author} ===")
    print(get_top_words(df[df['author']==author]['text'], 20))

# 7. UNIQUE VOCABULARY, OVERLAP, AND JACCARD SIMILARITY
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
X = vect.fit_transform(df['text'])
vocab = set(vect.get_feature_names_out())
print(f"\n=== Global vocabulary size: {len(vocab)} ===")

author_vocab = {}
for author in df['author'].unique():
    vect = CountVectorizer()
    vect.fit(df[df['author']==author]['text'])
    author_vocab[author] = set(vect.get_feature_names_out())

authors = df['author'].unique()
for i, a1 in enumerate(authors):
    for a2 in authors[i+1:]:
        overlap = len(author_vocab[a1] & author_vocab[a2])
        jaccard = overlap / len(author_vocab[a1] | author_vocab[a2])
        print(f"\n=== Vocab overlap {a1}-{a2}: {overlap} ===")
        print(f"=== Jaccard similarity {a1}-{a2}: {jaccard:.3f} ===")

# 8. BIGRAMS/TRIGRAMS
from sklearn.feature_extraction.text import CountVectorizer
def top_ngrams(texts, n=2, top_k=10):
    vec = CountVectorizer(ngram_range=(n,n), stop_words='english')
    bag = vec.fit_transform(texts)
    sum_words = bag.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top_k]

for author in df['author'].unique():
    print(f"\n=== Top 10 bigrams for {author} ===")
    print(top_ngrams(df[df['author']==author]['text'], n=2))
    print(f"\n=== Top 10 trigrams for {author} ===")
    print(top_ngrams(df[df['author']==author]['text'], n=3))

# 9. PUNCTUATION & SPECIAL CHARACTER USAGE
def count_punct(text):
    return sum(1 for c in text if c in string.punctuation)
df['punct_count'] = df['text'].apply(count_punct)
puncts = list(".,;:!?'-\"()[]{}")
for p in puncts:
    df[f'count_{p}'] = df['text'].apply(lambda x: x.count(p))

print("\n=== Average punctuation count per sentence by author ===")
print(df.groupby('author')['punct_count'].mean())
for p in puncts:
    print(f"\n=== Avg count of '{p}' per text by author ===")
    print(df.groupby('author')[f'count_{p}'].mean())

# 10. AUTHOR-SPECIFIC UNIQUE WORDS
author_unique_words = {}
for author in df['author'].unique():
    others = set().union(*[author_vocab[a] for a in df['author'].unique() if a != author])
    unique = author_vocab[author] - others
    author_unique_words[author] = unique
    print(f"\n=== Unique words for {author} (sample): {list(unique)[:20]} (total: {len(unique)}) ===")

# 11. SENTENCE STARTERS/ENDERS
for author in df['author'].unique():
    starters = df[df['author']==author]['text'].apply(lambda x: x.split()[0].lower() if len(x.split()) > 0 else "")
    top_starters = Counter(starters).most_common(10)
    print(f"\n=== Most common sentence starters for {author}: {top_starters} ===")
    enders = df[df['author']==author]['text'].apply(lambda x: x.split()[-1].strip(string.punctuation).lower() if len(x.split()) > 0 else "")
    top_enders = Counter(enders).most_common(10)
    print(f"=== Most common sentence enders for {author}: {top_enders} ===")

# 12. STOPWORDS RATIO
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
df['stopword_count'] = df['text'].apply(lambda x: sum(1 for w in x.split() if w.lower() in stops))
df['stopword_ratio'] = df['stopword_count'] / df['word_count']
print('\n=== Stopword ratio by author ===')
print(df.groupby('author')['stopword_ratio'].mean())

# 13. SAVE AS CSV
df.to_csv("train_eda_terminal.csv", index=False)

print("\n--- TERMINAL EDA complete! ---")
