import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, embed_size, feature_dim=1280):
        super().__init__()
        self.linear = nn.Linear(feature_dim, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, features):
        return self.dropout(self.relu(self.linear(features)))


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.5 if num_layers > 1 else 0.0)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        # Drop <end> token: predict [img, <start>, w1, ..., wN], target [<start>, w1, ..., wN, <end>]
        embeddings = self.dropout(self.embed(captions[:, :-1]))
        # Prepend image feature as first timestep
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        lstm_out, _ = self.lstm(embeddings)
        return self.linear(lstm_out)


class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, feature_dim=1280):
        super().__init__()
        self.encoder = Encoder(embed_size, feature_dim)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, features, captions):
        features = self.encoder(features)
        return self.decoder(features, captions)
