import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import json


class Vocabulary:
    def __init__(self, freq_threshold=1):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        return text.split()

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] = frequencies.get(word, 0) + 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def encode(self, text):
        tokenized_text = self.tokenize(text)
        return [self.stoi.get(word, self.stoi["<unk>"]) for word in tokenized_text]

    def decode(self, indices):
        return [self.itos.get(idx, "<unk>") for idx in indices]

    def save(self, filepath):
        with open(filepath, "w") as f:
            json.dump({"itos": self.itos, "stoi": self.stoi}, f)

    def load(self, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
            self.itos = {int(k): v for k, v in data["itos"].items()}
            self.stoi = data["stoi"]


class CaptionDataset(Dataset):
    def __init__(self, df, vocab):
        self.df = df
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        caption = row["caption"]
        features = torch.tensor(row["encoding_with_efficientnet"], dtype=torch.float32)

        if not caption.startswith("<start>"):
            caption = f"<start> {caption} <end>"

        encoded_caption = self.vocab.encode(caption)
        return features, torch.tensor(encoded_caption)


class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        features = torch.stack([item[0] for item in batch])
        captions = [item[1] for item in batch]
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        return features, captions
