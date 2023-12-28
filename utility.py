import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split

def create_train_dev_test(df, test_size=0.2, dev_size=0.25, random_state=None, stratify=None):
    """
    Create train, dev, and test sets from a DataFrame.

    Parameters:
    - df: DataFrame
    - target_col: Name of the target variable column
    - test_size: The proportion of the dataset to include in the test split
    - dev_size: The proportion of the dataset to include in the dev split
    - random_state: Seed for the random number generator
    - stratify: If not None, stratify the data based on this variable

    Returns:
    - train_df, dev_df, test_df
    """
    # Split the DataFrame into train_dev and test
    train_dev_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[stratify] if stratify else None
    )

    # Split the train_dev DataFrame into train and dev
    train_df, dev_df = train_test_split(
        train_dev_df, test_size=dev_size / (1 - test_size), random_state=random_state, stratify=train_dev_df[stratify] if stratify else None
    )

    return train_df, dev_df, test_df


def preprocess(image_path):
    # Load and resize image using PIL
    img = Image.open(image_path)
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img)
    
    # Expand dimensions to create a batch dimension
    img = img.unsqueeze(0)

    return img