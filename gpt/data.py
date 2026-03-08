import json 
import logging 
import numpy as np 
from pathlib import Path 

import ftfy # fixes broken unicode texts
from datasets import load_dataset # bookcorpusopen dataset
from tqdm import tqdm

import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset

from gpt.tokenizer import BPETokenizer

logger = logging.getLogger(__name__) 

# pretrain data preparation function 
def prepare_pretrain_data(
    tokenizer_path:str | Path, 
    output_dir:str | Path = "data/pretrain", 
    dataset_name:str = "lucadiliello/bookcorpusopen", 
    dataset_split:str = "train", 
    max_books:int | None = None
) -> Path : 

    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(f"Tokeniser not found at: {tokenizer_path}")

    # Load the dataset 
    dataset = load_dataset(dataset_name, split=dataset_split)

    # Optional subset 
    if max_books is not None:
        dataset = dataset.select(range(min(max_books, len(dataset))))

    # Load our tokeniser 
    tokeniser = BPETokenizer(tokeniser_path=tokenizer_path)

    # Count total tokens 
    total_tokens = 0 
    for example in tqdm(dataset, desc="COunting tokens"):
        text = ftfy.fix_text(example['text'])  # clean the text 
        token_ids = tokeniser.encode(text=text, pad=False, truncate=False)
        total_tokens += len(token_ids)

    logger.info(f"Total tokens are: {total_tokens}")

    # create the output directoruy 
    output_dir = Path(output_dir) 
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_path = output_dir / "tokens.bin" 

    # create memory map (memmap) with known size
    token_array = np.memmap(bin_path, dtype=np.uint16, mode="w+", shape=(total_tokens, ))

    # Fill it up ;)
    offset = 0
    for example in tqdm(dataset, desc="Writing tokens"):
        text = ftfy.fix_text(example['text']) 
        token_ids = tokeniser.encode(text=text, pad=False, truncate=False) 
        length = len(token_ids) 
        token_array[offset : offset + length] = token_ids
        offset += length

    # Flush our shit to disk
    token_array.flush() 

    # saving companion json file for metadata 
    metadata = {
        "total_tokens": total_tokens, 
        "vocab_size": tokeniser.vocab_size, 
        "dataset_name": dataset_name, 
        "num_books": len(dataset),
        "dtype":"uint16"
    }
    
    # metadata tells about what the bin file contains
    metadata_path = output_dir/"metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return bin_path
    
# Pytorch dataset that reads the memmap and serves contiguous 512 token chunks
class PreTrainingDataset(Dataset):
    """
    Pytorch Dataset

    Args:
        data_path: str | Path = path to our memmap file 
        seq_len:int = Number of tokens per training example

    Returns:
        Takes an index and returns a dataset point
    """
    def __init__(self, data_path:str| Path, seq_len:int = 512 ) -> None:
        super().__init__()
        self.seq_len = seq_len 
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.num_chunks = len(self.data) // seq_len

    def __len__(self) -> int : 
        return self.num_chunks

    def __getitem__(self, idx:int) -> dict[str, torch.Tensor]:
        start = idx * self.seq_len 
        end = start + self.seq_len
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        return {"input_ids": chunk}

# Factory Function
def get_pretrain_dataloaders(
    data_path: str | Path, 
    seq_len: int = 512, 
    batch_size:int = 8,             # micro batch for grad acum
    val_split: float = 0.005,       # 0.5% for validation
    num_workers:int = 4, 
    seed:int = 774
) -> tuple[DataLoader, DataLoader]:
    """
    Creates train and validation DataLoaders from a pre-tokenized memmap file.

    Args:
        data_path: Path to the .bin memmap file.
        seq_len: Tokens per training example.
        batch_size: Micro batch size (gradient accumulation handled externally).
        val_split: Fraction of chunks reserved for validation.
        num_workers: DataLoader worker processes.
        seed: Random seed (for future use with any randomised operations).

    Returns:
        Tuple of (train_loader, val_loader).
    """

    # full dataset creation 
    dataset = PreTrainingDataset(data_path=data_path, seq_len=seq_len)
    total_chunks = len(dataset)

    # train and val splits 
    val_size = int(total_chunks * val_split)
    train_size = total_chunks - val_size

    # Splits 
    train_indices = range(0, train_size)
    val_indices = range(train_size, total_chunks) 

    # Datasets
    train_dataset = Subset(dataset=dataset, indices=train_indices)
    val_dataset = Subset(dataset=dataset, indices=val_indices)

    # Creating DataLoaders 
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=False
    )

    logger.info(
        f"DataLoaders ready - "
        f"train: {len(train_dataset)} chunks ({train_size} examples), "
        f"val: {len(val_dataset)} chunks ({val_size} examples), "
        f"micro_batch: {batch_size}"
    )

    return train_loader, val_loader

###################################################################
#                          FINETUNE DATA
###################################################################

from enum import Enum 
from dataclasses import dataclass 
from datasets import load_dataset


class TaskType(Enum):
    # The four task types from GPT1 paper(check sec 3.3)"""
    CLASSIFICATION = "classification"
    ENTAILMENT = "entailment"
    SIMILARITY = "similarity"
    MULTIPLE_CHOICE = "multiple_choice"

@dataclass(frozen=True)
class TaskConfig:
    # Dataclass describing a single downstream task
    hf_name:str                  # eg. "glue"
    hf_subset:str | None         # eg. "sst2", or None for single split
    task_type: TaskType          # type of task (ENUM)
    text_columns: list[str]      # columns holding input texts
    label_column: str            # column holding label
    num_labels: int              # number of output class(regression=1)
    is_regression:bool = False   # True for STS-B


# Task registry - Detailed information on multiple tasks 
TASK_REGISTRY: dict[str, TaskConfig] = {
    # ---------- Classification ----------
    "cola" : TaskConfig(
        hf_name="glue", hf_subset="cola", task_type=TaskType.CLASSIFICATION, 
        text_columns=['sentence'], label_column='label', num_labels=2
    ), 

    "sst2": TaskConfig(
        hf_name="glue", hf_subset="sst2", task_type=TaskType.CLASSIFICATION, 
        text_columns=['sentence'], label_column='label', num_labels=2
    ),

    # ---------- Entailment -----------
    "mnli" : TaskConfig(
        hf_name="glue", hf_subset="mnli", task_type=TaskType.ENTAILMENT, 
        text_columns=['premise', 'hypothesis'], label_column='label', num_labels=3
    ), 
    "qnli" : TaskConfig(
        hf_name="glue", hf_subset="qnli", task_type=TaskType.ENTAILMENT, 
        text_columns=['question', 'sentence'], label_column='label', num_labels=2
    ), 
    "rte": TaskConfig(
        hf_name="glue", hf_subset="rte", task_type=TaskType.ENTAILMENT, 
        text_columns=['sentence1', 'sentence2'], label_column='label', num_labels=2
    ), 
    "snli": TaskConfig(
        hf_name="stanford/snli", hf_subset=None, task_type=TaskType.ENTAILMENT, 
        text_columns=['premise', 'hypothesis'], label_column='label', num_labels=3
    ), 
    "scitail": TaskConfig(
        hf_name="allenai/scitail", hf_subset="tsv_format", task_type=TaskType.ENTAILMENT, 
        text_columns=['premise', 'hypothesis'], label_column='label', num_labels=2
    ), 

    # ---------- Similarity ----------
    "mrpc": TaskConfig(
        hf_name='glue', hf_subset='mrpc', task_type=TaskType.SIMILARITY, 
        text_columns=['sentence1', 'sentence2'], label_column='label', num_labels=2
    ), 
    "qqp": TaskConfig(
        hf_name="glue", hf_subset="qqp",task_type=TaskType.SIMILARITY, 
        text_columns=['question1', 'question2'], label_column='label', 
        num_labels=2
    ), 
    "stsb": TaskConfig(
        hf_name="glue", hf_subset="stsb", task_type=TaskType.SIMILARITY, 
        text_columns=['sentence1', 'sentence2'], label_column='label', num_labels=1, is_regression=True
    ), 
    # --------------- MUltiple Choice ------------------
    "race": TaskConfig(
        hf_name="ehovy/race", hf_subset="all", task_type=TaskType.MULTIPLE_CHOICE, 
        text_columns=['article', 'question', 'options'], # special handling needed 
        label_column='answer', num_labels=4
    )
}