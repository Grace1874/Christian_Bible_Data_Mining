import pandas as pd
from pathlib import Path

# Find and load the Bible dataset
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "Dataset" / "t_kjv.csv"

df = pd.read_csv(file_path)

print("FIRST FIVE ROWS")
print(df.head())

print("\nDATASET SIZE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDATASET INFORMATION")
df.info()