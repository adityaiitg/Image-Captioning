import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.dataset import Vocabulary, CaptionDataset, Collate
from src.model import ImageCaptioningModel
from src.inference import greedy_predict

def main():
    print("1. Loading Data...")
    try:
        train_df = pd.read_parquet('dataset/processed_dataset/train.parquet')
        dev_df = pd.read_parquet('dataset/processed_dataset/dev.parquet')
        test_df = pd.read_parquet('dataset/processed_dataset/test.parquet')
        print(f"Train: {train_df.shape}")
        print(f"Dev: {dev_df.shape}")
        print(f"Test: {test_df.shape}")
    except Exception as e:
        print(f"Failed to load parquet files: {e}")
        return

    print("\n2. Building Vocabulary...")
    vocab = Vocabulary(freq_threshold=2)
    captions_list = train_df['caption'].tolist()
    vocab.build_vocabulary([c if c.startswith('<start>') else f"<start> {c} <end>" for c in captions_list])
    print(f"Total vocabulary size: {len(vocab)}")

    print("\n3. Creating Datasets and DataLoaders...")
    train_dataset = CaptionDataset(train_df, vocab)
    dev_dataset = CaptionDataset(dev_df, vocab)
    test_dataset = CaptionDataset(test_df, vocab)

    pad_idx = vocab.stoi["<pad>"]
    collate_fn = Collate(pad_idx=pad_idx)

    BATCH_SIZE = 128
    NUM_WORKERS = 0

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, collate_fn=collate_fn)

    print("\n4. Initializing Model...")
    device = torch.device("mps")
    print(f"Using device: {device} (explicit user override)")

    EMBED_SIZE = 512
    HIDDEN_SIZE = 512
    VOCAB_SIZE = len(vocab)
    NUM_LAYERS = 2
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 15

    model = ImageCaptioningModel(embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print("\n5. Training Loop with Validation...")
    writer = SummaryWriter(log_dir="runs/image_captioning")
    global_step = 0
    best_val_loss = float('inf')
    
    os.makedirs('weights', exist_ok=True)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for features, captions in progress_bar:
            features = features.to(device)
            captions = captions.to(device)
            
            outputs = model(features, captions)
            loss = criterion(outputs.view(-1, VOCAB_SIZE), captions.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to stabilize LSTM training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
            writer.add_scalar("Loss/train", loss.item(), global_step)
            global_step += 1
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_progress_bar = tqdm(dev_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for features, captions in val_progress_bar:
                features = features.to(device)
                captions = captions.to(device)
                
                outputs = model(features, captions)
                loss = criterion(outputs.view(-1, VOCAB_SIZE), captions.view(-1))
                val_loss += loss.item()
                val_progress_bar.set_postfix(loss=loss.item())
                
        avg_val_loss = val_loss / len(dev_loader)
        writer.add_scalar("Loss/validation", avg_val_loss, epoch)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"--> Validation loss improved! Saving model to weights/caption_model.pth")
            torch.save(model.state_dict(), 'weights/caption_model.pth')
            
    writer.close()

    print("\n7. Testing Inference...")
    index = np.random.randint(0, len(test_df))
    row = test_df.iloc[index]
    features = torch.tensor(row['encoding_with_efficientnet'], dtype=torch.float32).unsqueeze(0)
    actual_caption = row['caption']
    
    predicted_caption = greedy_predict(model, features, vocab, device=device)
    
    print(f"Image File: {row['file']}")
    print(f"Actual: {actual_caption}")
    print(f"Predicted: {predicted_caption}")
    print("\nSuccess! Training and inference completed seamlessly.")

if __name__ == "__main__":
    main()
