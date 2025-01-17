# src/data_preprocessing.py
import os
import json
import pandas as pd
import re

def preprocess_text(text):
    # Tokenization, normalization, and handling Amharic-specific linguistic features
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

def preprocess_data():
    raw_data_dir = 'data/raw'
    processed_data_dir = 'data/processed'
    os.makedirs(processed_data_dir, exist_ok=True)
    
    for filename in os.listdir(raw_data_dir):
        if filename.endswith('.json'):
            with open(os.path.join(raw_data_dir, filename), 'r') as f:
                data = json.load(f)
            for message in data:
                message['message'] = preprocess_text(message['message'])
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(processed_data_dir, f'{filename}.csv'), index=False)

if __name__ == '__main__':
    preprocess_data()