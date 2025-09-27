import torch
import torch.nn as nn

# --- Encodeur RNN ---
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, x):
        batch_size, max_sentences, max_len = x.size()
        x = x.view(batch_size * max_sentences, max_len)
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded)
        hidden = outputs[:, -1, :]  
        hidden = hidden.view(batch_size, max_sentences, -1)
        return hidden  # (batch, max_sentences, hidden*2)

# --- Sélecteur de phrases ---
class SentenceSelector(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, encoder_outputs):
        scores = self.linear(encoder_outputs).squeeze(-1)  
        return scores
