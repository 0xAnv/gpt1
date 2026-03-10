import argparse
import sys
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

from gpt.tokenizer import BPETokenizer
from gpt.data import prepare_pretrain_data

def train_tokenizer(tokenizer_path: Path, max_books: int = None):
    print("Loading dataset for tokenizer training (streaming)...")
    dataset = load_dataset("lucadiliello/bookcorpusopen", split="train", streaming=True)
    
    if max_books is not None:
        dataset = dataset.take(max_books)
        
    total_to_train = max_books if max_books else 17868

    tokenizer = BPETokenizer()
    print("Preparing dataset for tokenizer training...")
    
    def doc_iterator():
        for example in tqdm(dataset, total=total_to_train, desc="Iterating through dataset for tokenizer", unit="book"):
            yield example["text"]
            
    tokenizer.train_from_iterator(doc_iterator(), vocab_size=40000, save_path=tokenizer_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Tokenizer and dataset.")
    parser.add_argument("--max_books", type=int, default=None, help="Max number of books (leave empty for all)")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Number of documents to tokenize in one batch (controls memory usage)")
    args = parser.parse_args()

    tokenizer_path = Path("data/pretrain/tokenizer.json")
    
    if not tokenizer_path.exists():
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        train_tokenizer(tokenizer_path, args.max_books)
    else:
        print(f"Tokenizer already exists at {tokenizer_path}")

    print(f"Preparing pretrain data with chunk_size={args.chunk_size}... This will write tokens to disk.")
    prepare_pretrain_data(
        tokenizer_path=tokenizer_path,
        output_dir="data/pretrain",
        dataset_name="lucadiliello/bookcorpusopen",
        dataset_split="train",
        max_books=args.max_books,
        chunk_size=args.chunk_size
    )
    print("Data preparation complete!")
