import pandas as pd
import re
from pathlib import Path

# Locate project folders
project_folder = Path(__file__).resolve().parent.parent

# Load original dataset
file_path = project_folder / "Dataset" / "t_kjv.csv"
df = pd.read_csv(file_path)

# Clean Bible verse text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["cleaned_text"] = df["t"].apply(clean_text)

# Save cleaned dataset
output_path = project_folder / "Results" / "cleaned_bible_verses.csv"
df.to_csv(output_path, index=False)

print("TEXT PREPROCESSING COMPLETED!")
print("\nOriginal verse:")
print(df["t"].iloc[0])

print("\nCleaned verse:")
print(df["cleaned_text"].iloc[0])

print("\nTotal verses processed:", len(df))
print("\nSaved to:", output_path)