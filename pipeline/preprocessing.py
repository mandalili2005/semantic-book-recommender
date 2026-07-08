import pandas as pd 
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

csv_path = ROOT / "data" / "raw" / "books_raw.csv"
processed_path = ROOT / "data" / "processed" / "books_cleaned.csv"

#import dataset 
if not csv_path.exists():
    df = pd.read_parquet(
        "hf://datasets/perceptron-743/good-reads-data/data/train-00000-of-00001.parquet"
    )
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} books.")
else:
    print("CSV already exists.")

df = pd.read_csv(csv_path)

#inspect 
print(df.head())
print(df.info())

print(df.columns)
print(df.shape)
print(df.isnull().sum())

if not processed_path.exists():
    #clean dataset

    print("Creating cleaned dataset...")

    df = df.drop(columns=[
        "edition",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "price"
    ])

    df["search_text"] = (
        df["title"].fillna("") + " " +
        df["author"].fillna("") + " " +
        df["genres"].fillna("") + " " +
        df["description"].fillna("")
    )

    processed_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(processed_path, index=False)

    print("Cleaned dataset saved!")

else:

    print("Processed dataset already exists.")

    df = pd.read_csv(processed_path)

#create new column for combined text for embeddings 
df["search_text"] = (
    df["title"].fillna("") + " " +
    df["author"].fillna("") + " " +
    df["genres"].fillna("") + " " +
    df["description"].fillna("")
)

print(df["search_text"].head())

df.to_csv("data/processed/books_cleaned.csv", index=False)
