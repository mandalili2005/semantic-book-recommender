from pathlib import Path 
import pandas as pd
import chromadb 
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent


#paths 
processed_path = ROOT / "data" / "processed" / "books_cleaned.csv"
embeddings_path = ROOT / "data" / "embeddings"


#load cleaned data
df = pd.read_csv(processed_path)


#load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

#create ChromaDB client
client = chromadb.PersistentClient(path=str(embeddings_path))

#create or get collection
collection = client.get_or_create_collection(name="books")

#prepare data
texts = df["search_text"].fillna("").tolist()
ids = df["bookId"].astype(str).tolist()

metadatas = df[["title", "author", "genres", "rating", "numRatings", "coverImg"]].fillna("").to_dict("records")

#generate embeddings
embeddings = model.encode(texts, show_progress_bar=True).tolist()

#add to ChromaDB
collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=texts,
    metadatas=metadatas
)

print(f"Stored {len(ids)} book embeddings in ChromaDB.")