import streamlit as st
import torch
import json
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize, word_tokenize
import string

from model import EncoderRNN, SentenceSelector

# -------------------
# 1. Chargement du vocabulaire et du modèle
# -------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_vocab():
    with open("saved_models/vocab.json", "r") as f:
        vocab_data = json.load(f)
    word2idx = {k: int(v) if isinstance(v, int) else v for k, v in vocab_data["word2idx"].items()}
    idx2word = {int(k): v for k, v in vocab_data["idx2word"].items()}

    vocab_size = len(word2idx)
    embed_size, hidden_size = 128, 256

    encoder = EncoderRNN(vocab_size, embed_size, hidden_size, pad_idx=word2idx["<PAD>"]).to(DEVICE)
    decoder = SentenceSelector(hidden_size).to(DEVICE)

    encoder.load_state_dict(torch.load("saved_models/encoder.pth", map_location=DEVICE))
    decoder.load_state_dict(torch.load("saved_models/decoder.pth", map_location=DEVICE))
    encoder.eval()
    decoder.eval()

    return encoder, decoder, word2idx, idx2word

encoder, decoder, word2idx, idx2word = load_model_and_vocab()

# -------------------
# 2. Fonctions utilitaires
# -------------------
def preprocess_article(article, max_sentences=50):
    sents = sent_tokenize(article)[:max_sentences]
    cleaned = []
    for sent in sents:
        words = [w for w in word_tokenize(sent) if w.lower() not in string.punctuation]
        cleaned.append(" ".join(words))
    return cleaned

def tfidf_select(sents, k=3):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    if len(sents) == 0:
        return [], []
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sents)
    scores = X.sum(axis=1).A1
    idx = np.argsort(scores)[-k:][::-1]
    idx_sorted = sorted(idx)
    selected = [sents[i] for i in idx_sorted]
    return selected, idx_sorted

def encode_sentence(sentence, word2idx, max_len=20):
    tokens = word_tokenize(sentence.lower())
    seq = [word2idx.get(w, word2idx["<UNK>"]) for w in tokens]
    if len(seq) < max_len:
        seq += [word2idx["<PAD>"]] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    return seq

# -------------------
# 3. Interface Streamlit
# -------------------
st.title("📝 Résumeur automatique extractif")

input_type = st.radio("Choisir l'entrée :", ("Écrire un texte", "Importer un PDF"))
article_text = ""

if input_type == "Écrire un texte":
    article_text = st.text_area("✍️ Écris ton article ici :", height=200)
elif input_type == "Importer un PDF":
    uploaded_file = st.file_uploader("📄 Choisir un fichier PDF", type=["pdf"])
    if uploaded_file is not None:
        pdf = PdfReader(uploaded_file)
        for page in pdf.pages:
            article_text += page.extract_text()

if st.button("🚀 Générer résumé") and article_text.strip() != "":
    sents = preprocess_article(article_text, max_sentences=50)

    # Résumé TF-IDF
    selected_tfidf, _ = tfidf_select(sents, k=3)

    # Résumé RNN + Attention
    seqs = [encode_sentence(sent, word2idx) for sent in sents]
    batch = torch.tensor([seqs], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        encoder_outputs = encoder(batch)
        scores = decoder(encoder_outputs)

    num_real_sents = len(sents)
    topk_indices = torch.topk(scores, 3, dim=1).indices[0].cpu().numpy()
    topk_indices_valid = [i for i in topk_indices if i < num_real_sents]
    selected_rnn = [sents[i] for i in topk_indices_valid]

    st.subheader("📌 Résumé TF-IDF")
    st.write(" ".join(selected_tfidf))

    st.subheader("🤖 Résumé RNN + Attention")
    st.write(" ".join(selected_rnn))
