import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

# Téléchargement des ressources
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

vectorizer = TfidfVectorizer(stop_words="english")

# --- Prétraitement article ---
def preprocess_article(article, max_sentences=50):
    sents = sent_tokenize(article)
    sents = [s.strip() for s in sents if len(s.strip()) > 0]
    if len(sents) > max_sentences:
        sents = sents[:max_sentences]

    cleaned_sents = []
    for sent in sents:
        words = word_tokenize(sent)
        words = [w for w in words if w.lower() not in stop_words and w not in string.punctuation]
        cleaned_sents.append(" ".join(words))
    return cleaned_sents

# --- TF-IDF sélection ---
def tfidf_select(sents, k=3):
    if len(sents) == 0:
        return []
    X = vectorizer.fit_transform(sents)
    scores = X.sum(axis=1).A1
    idx = scores.argsort()[-k:][::-1]
    idx_sorted = sorted(idx)
    selected = [sents[i] for i in idx_sorted]
    return selected

# --- Conversion phrase -> indices ---
def encode_sentences(sents, word2idx, max_len=20, max_sentences=50):
    seqs = []
    for sent in sents:
        tokens = word_tokenize(sent.lower())
        idxs = [word2idx.get(w, word2idx["<UNK>"]) for w in tokens]
        if len(idxs) < max_len:
            idxs += [word2idx["<PAD>"]] * (max_len - len(idxs))
        else:
            idxs = idxs[:max_len]
        seqs.append(idxs)

    while len(seqs) < max_sentences:
        seqs.append([word2idx["<PAD>"]] * max_len)

    return torch.tensor([seqs], dtype=torch.long)
