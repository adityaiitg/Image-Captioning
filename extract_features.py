import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from src.preprocessing import get_efficientnet_model, preprocess_image

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the EfficientNet model
    model = get_efficientnet_model(device)

    # We will process train, dev, and test
    splits = ['train', 'dev', 'test']
    
    for split in splits:
        parquet_path = f"dataset/processed_dataset/{split}.parquet"
        print(f"\nProcessing {split} split from {parquet_path}")
        
        try:
            df = pd.read_parquet(parquet_path)
            # Some images might be duplicate in the df (1 image -> 5 captions), 
            # but we only need to extract features for unique images.
            unique_images = df['file'].unique()
            
            # Dictionary to store features: {filename: feature_array}
            feature_dict = {}
            
            progress_bar = tqdm(unique_images, desc=f"Extracting features for {split}")
            for img_path_kaggle in progress_bar:
                # The parquet has kaggle paths. Adjust to local relative paths.
                img_name = img_path_kaggle.split('/')[-1]
                local_img_path = os.path.join("dataset", "Images", img_name)
                
                try:
                    img_tensor = preprocess_image(local_img_path).to(device)
                    with torch.no_grad():
                        # Extract feature and convert to numpy array of shape (1280,)
                        feature = model(img_tensor).cpu().numpy().flatten()
                        feature_dict[img_path_kaggle] = feature
                except Exception as e:
                    print(f"Error processing {local_img_path}: {e}")
                    # If image is missing, we could input a zero vector, but we assume raw dataset is present
                    feature_dict[img_path_kaggle] = np.zeros(1280, dtype=np.float32)

            # Map the new features back to the dataframe
            # The column is a list/array for each row
            print("Mapping features to dataframe...")
            df['encoding_with_efficientnet'] = df['file'].map(feature_dict)
            
            # Overwrite the parquet (or save as new, but overwriting is fine since it keeps VGG too if we don't drop)
            # It's safer to save as a new file or add the column to existing
            df.to_parquet(parquet_path)
            print(f"Saved {split} split with EfficientNetV2-S features to {parquet_path}")
            
        except Exception as e:
            print(f"Failed to process {split} split: {e}")

if __name__ == "__main__":
    main()
