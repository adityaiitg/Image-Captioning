import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, embed_size, feature_dim=4096):
        super(Encoder, self).__init__()
        # VGG19 features are 4096 dimensional, map to embed_size
        self.linear = nn.Linear(feature_dim, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, features):
        features = self.linear(features)
        features = self.relu(features)
        return self.dropout(features)

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, features, captions):
        # features shape: (batch_size, embed_size)
        # captions shape: (batch_size, seq_len)
        
        # We drop the <end> token to match sequence lengths for training
        # If captions are e.g., <start> a dog <end>, we pass <start> a dog
        embeddings = self.dropout(self.embed(captions[:, :-1]))
        
        # features shape becomes (batch_size, 1, embed_size)
        # embeddings shape is (batch_size, seq_len-1, embed_size)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(embeddings)
        
        # outputs shape: (batch_size, seq_len, vocab_size)
        outputs = self.linear(lstm_out)
        return outputs

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, feature_dim=4096):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = Encoder(embed_size, feature_dim)
        self.decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers)
        
    def forward(self, features, captions):
        features = self.encoder(features)
        outputs = self.decoder(features, captions)
        return outputs
