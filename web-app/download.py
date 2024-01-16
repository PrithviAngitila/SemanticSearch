import pandas as pd
from datasets import load_dataset

dataset = load_dataset("lsb/simplewiki2023")
df = pd.DataFrame(dataset['train'])  # Replace 'train' with 'test' or 'validation' if needed

# Step 5: Save as CSV
df.to_csv('./data/dataset.csv', index=False)