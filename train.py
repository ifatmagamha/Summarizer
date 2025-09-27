import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from model import EncoderRNN, SentenceSelector

import nltk
nltk.download('punkt_tab')

# -------------------
# 1. Préparation
# -------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

raw = load_dataset("cnn_dailymail", "3.0.0")
train_articles = [raw["train"][i]["article"] for i in range(200)]  # petit subset

# -------------------
# 2. Construction vocabulaire
# -------------------
def build_vocab(articles, min_freq=5):
    counter = Counter()
    for article in articles:
        for sent in sent_tokenize(article)[:50]:
            tokens = word_tokenize(sent.lower())
            counter.update(tokens)
    vocab = {w for w, f in counter.items() if f >= min_freq}
    word2idx = {w: i+2 for i, w in enumerate(sorted(vocab))}
    word2idx["<PAD>"] = 0
    word2idx["<UNK>"] = 1
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word

word2idx, idx2word = build_vocab(train_articles)
vocab_size = len(word2idx)

# -------------------
# 3. Dataset
# -------------------
def encode_sentence(sentence, word2idx, max_len=20):
    tokens = word_tokenize(sentence.lower())
    seq = [word2idx.get(w, word2idx["<UNK>"]) for w in tokens]
    return seq[:max_len] + [word2idx["<PAD>"]] * max(0, max_len-len(seq))

class SummaryDataset(Dataset):
    def __init__(self, articles, word2idx, max_len=20, max_sentences=50):
        self.data = []
        pad_sent = [word2idx["<PAD>"]] * max_len
        for article in articles:
            sents = sent_tokenize(article)[:max_sentences]
            encoded = [encode_sentence(s, word2idx, max_len) for s in sents]
            while len(encoded) < max_sentences:
                encoded.append(pad_sent)
            self.data.append(torch.tensor(encoded, dtype=torch.long))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

dataset = SummaryDataset(train_articles, word2idx)
loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

# -------------------
# 4. Modèle + Optimiseur
# -------------------
embed_size, hidden_size = 128, 256
encoder = EncoderRNN(vocab_size, embed_size, hidden_size).to(DEVICE)
decoder = SentenceSelector(hidden_size).to(DEVICE)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

# -------------------
# 5. Entraînement
# -------------------
for epoch in range(3):
    total_loss = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        encoder_outputs = encoder(batch)
        scores = decoder(encoder_outputs)

        labels = torch.zeros(scores.shape, device=DEVICE)  # labels fictifs (à améliorer avec TF-IDF)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

# -------------------
# 6. Sauvegarde
# -------------------
os.makedirs("saved_models", exist_ok=True)
torch.save(encoder.state_dict(), "saved_models/encoder.pth")
torch.save(decoder.state_dict(), "saved_models/decoder.pth")
with open("saved_models/vocab.json", "w") as f:
    json.dump({"word2idx": word2idx, "idx2word": idx2word}, f)

print(" Modèle et vocab sauvegardés dans saved_models/")
    