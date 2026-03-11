import json 
import logging 
import numpy as np 
from pathlib import Path 

# data preprocessing and download
import ftfy # fixes broken unicode texts
from datasets import load_dataset # bookcorpusopen dataset
from tqdm import tqdm

# torch specifics
import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset

# finetuning data
from enum import Enum 
from dataclasses import dataclass 

# tokeniser
from gpt.tokenizer import BPETokenizer

logger = logging.getLogger(__name__) 


# pretrain data preparation function 
def prepare_pretrain_data(
    tokenizer_path:str | Path, 
    output_dir:str | Path = "data/pretrain", 
    dataset_name:str = "lucadiliello/bookcorpusopen", 
    dataset_split:str = "train", 
    max_books:int | None = None,
    chunk_size: int = 1000
) -> Path : 

    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(f"Tokeniser not found at: {tokenizer_path}")

    # Load the dataset with streaming to save memory
    dataset = load_dataset(dataset_name, split=dataset_split, streaming=True)

    # Optional subset 
    if max_books is not None:
        dataset = dataset.take(max_books)

    import os
    import multiprocessing
    

    # Load our tokeniser
    tokeniser = BPETokenizer(tokeniser_path=tokenizer_path)

    # Force Rust's Rayon library to use all available cores for encode_batch
    num_proc = str(max(1, multiprocessing.cpu_count()))
    os.environ["RAYON_RS_NUM_CPUS"] = num_proc
    os.environ["RAYON_NUM_THREADS"] = num_proc
    # create the output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    bin_path = output_dir_path / "tokens.bin"

    # We don't know the exact size yet, so we will append to binary file
    # Initialize file empty
    with open(bin_path, "wb") as f:
        pass

    total_tokens = 0
    num_docs = 0
    
    # Estimate total for progress bar (bookcorpusopen has ~17,868 books)
    total_to_process = max_books if max_books else 17868

    logger.info(f"Tokenizing dataset using native Rust multithreading ({num_proc} cores) in chunks of {chunk_size}...")

    with open(bin_path, "ab") as f:
        # We manually chunk the dataset iteration to minimize python string footprint in RAM
        chunk = []
        for example in tqdm(dataset, total=total_to_process, desc="Tokenizing & writing to disk", unit="book"):
            chunk.append(ftfy.fix_text(example['text']))
            num_docs += 1
            
            if len(chunk) == chunk_size:
                # encode_batch uses Rust rayon to tokenize the chunk on all CPU cores natively!
                batch_ids = tokeniser.encode(chunk, pad=False, truncate=False)
                
                # batch_ids is a list of lists of token ids. Write sequence by sequence to save RAM.
                for seq in batch_ids:
                    if seq:
                        data_bytes = np.array(seq, dtype=np.uint16).tobytes()
                        f.write(data_bytes)
                        total_tokens += len(seq)
                    
                # force garbage collection to prevent memory leaks from python lists
                import gc
                del batch_ids, chunk
                gc.collect()
                
                chunk = [] # release memory rapidly
                
        # process any remaining sentences in the final uneven chunk
        if chunk:
            batch_ids = tokeniser.encode(chunk, pad=False, truncate=False)
            for seq in batch_ids:
                if seq:
                    data_bytes = np.array(seq, dtype=np.uint16).tobytes()
                    f.write(data_bytes)
                    total_tokens += len(seq)

    # saving companion json file for metadata 
    metadata = {
        "total_tokens": total_tokens, 
        "vocab_size": tokeniser.vocab_size, 
        "dataset_name": dataset_name, 
        "num_books": num_docs,
        "dtype":"uint16"
    }
    
    # metadata tells about what the bin file contains
    metadata_path = output_dir_path/"metadata.json"
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

class TaskType(Enum):
    # The four task types from GPT1 paper(check sec 3.3)
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

##################################################################
#          FINETUNING FORMATING FUNCTIONS
##################################################################

def _format_classification(
    text:str, 
    tokenizer:BPETokenizer, 
    max_seq_len:int
) -> list[int] :
    """
    Classification: [start] text [extract]

    We encode the text WITHOUT special tokens (no padding, no truncation yet), 
    then manually wrap it with start and extract tokens, 
    then truncate/pad the final sequence.
    """
    start_id = tokenizer.tokeniser.token_to_id(BPETokenizer.START_TOKEN)
    cls_id = tokenizer.tokeniser.token_to_id(BPETokenizer.CLS_TOKEN)
    pad_id = tokenizer.tokeniser.token_to_id(BPETokenizer.PAD_TOKEN)

    # Encode raw text (no special tokens, no pad, no truncation)
    text_ids = tokenizer.encode(text, pad=False, truncate=False)

    # Build: [start] text [extract] 
    token_ids = [start_id] + text_ids + [cls_id]

    # Truncate to max_seq_len (cut text tokens, keep start and extract)
    if len(token_ids) > max_seq_len:
        # keep [start] + first (max_seq_len -2) text tokens + [extract]
        token_ids = [start_id] + text_ids[:max_seq_len-2] + [cls_id]
    
    # pad on the right 
    attention_len = len(token_ids)
    token_ids = token_ids + [pad_id] * (max_seq_len - attention_len)

    return token_ids


def _format_entailment(
    text1:str, 
    text2:str, 
    tokenizer:BPETokenizer, 
    max_seq_len:int
) -> list[int]:
    """ Entailment/NLI : [start] premise [delim] hypothesis [extract] """
    start_id = tokenizer.tokeniser.token_to_id(BPETokenizer.START_TOKEN)
    delim_id = tokenizer.tokeniser.token_to_id(BPETokenizer.END_TOKEN) #</s>
    cls_id = tokenizer.tokeniser.token_to_id(BPETokenizer.CLS_TOKEN)
    pad_id = tokenizer.tokeniser.token_to_id(BPETokenizer.PAD_TOKEN)

    text1_ids = tokenizer.encode(text=text1, pad=False, truncate=False)
    text2_ids = tokenizer.encode(text=text2, pad=False, truncate=False)

    # Build: [start] text1 [delimit] text2 [extract]
    token_ids = [start_id] + text1_ids + [delim_id] + text2_ids + [cls_id]

    # Truncate: trim text2 first, then text1 if still too long 
    if len(token_ids) > max_seq_len: 
        # budget = max_seq_len -3 tokens for [start], [delim], [extract]
        budget = max_seq_len - 3 
        # give each half but if one is shorter give rest to other 
        half = budget // 2
        if len(text1_ids) <= half:
            text2_ids = text2_ids[:budget - len(text1_ids)]
        elif len(text2_ids) <= half:
            text1_ids = text1_ids[:budget - len(text2_ids)]
        else:
            text1_ids = text1_ids[:half]
            text2_ids = text2_ids[:budget - half]
        token_ids = [start_id] + text1_ids + [delim_id] + text2_ids + [cls_id]
    
    attention_len = len(token_ids)
    token_ids = token_ids + [pad_id] * (max_seq_len - attention_len)

    return token_ids


def _format_similarity(
    text1:str, 
    text2:str, 
    tokenizer:BPETokenizer, 
    max_seq_len:int
) -> tuple[list[int], list[int]]:
    """
    Similarity TWO ORDERINGS 
        ordering1: [start] s1 [delimit] s2 [extract]
        ordering2: [start] s2 [delimit] s1 [extract]

    Returns a tuple of (ordering_1_ids, ordering_2_ids)
    """
    ordering1 = _format_entailment(text1=text1, text2=text2, tokenizer=tokenizer, max_seq_len=max_seq_len)
    ordering2 = _format_entailment(text1=text2, text2=text1, tokenizer=tokenizer, max_seq_len=max_seq_len)

    return ordering1, ordering2

def _format_multiple_choice(
    context:str, 
    options:list[str],
    tokenizer: BPETokenizer, 
    max_seq_len: int
) -> list[list[int]]: 
    """
    Multiple Choice: one sequence per option
      [start] context [delim] option_i [extract]

    Returns a list of token_id sequences, one per option.
    """
    return [
        _format_entailment(text1=context, text2=option, tokenizer=tokenizer, max_seq_len=max_seq_len)
        for option in options
    ]


############################################################
#                 FINETUNING DATASET
############################################################
class FinetuneDataset(Dataset):
    """
    Unified Pytorch dataset for all GPT 1 Finetuning tasks. 

    Handles the four task types from the paper (section 3.3):
        - Classification:    single text -> [start] text [extract]
        - Entailment:        text pair -> [start] t1 [delimit] t2 [extract]
        - Similarity:        text pair -> two orderings (elem-wise addition)
        - Multiple Choice:   context + N options -> N sequences
    """
    def __init__(
        self, 
        task_name:str, 
        tokenizer: BPETokenizer, 
        split:str = "train", 
        max_seq_len:int = 512
    ) -> None :
        super().__init__()

        if task_name not in TASK_REGISTRY: 
            raise ValueError(
                f"Unknown Task '{task_name}'. Available: {list(TASK_REGISTRY.keys())}"
            )

        self.config = TASK_REGISTRY[task_name]
        self.tokenizer = tokenizer 
        self.max_seq_len = max_seq_len
        self.task_name = task_name

        # Load the Hugging Face dataset 
        if self.config.hf_subset:
            self.dataset = load_dataset(self.config.hf_name, self.config.hf_subset, split=split)
        else: 
            self.dataset = load_dataset(self.config.hf_name, split=split)
        
        # Filter out examples with label == -1 (SNLI has some unlabelled examples)
        if task_name == "snli": 
            self.dataset = self.dataset.filter(lambda x : x['label'] != -1) 
        
        # For SciTail: convert string labels ('entails'/'neutral') to integers
        if task_name == "scitail":
            label_map = {"entails": 0, "neutral": 1}
            self.dataset = self.dataset.map(
                lambda x : {"label": label_map[x["label"]]}
            )

        # For RACE: convert letter answer ('A', 'B', 'C', 'D') to integer (0, 1, 2, 3)
        if task_name == "race": 
            answer_map = {
                "A" : 0 , 
                "B" : 1 , 
                "C" : 2 , 
                "D" : 3
            }
            self.dataset = self.dataset.map(
                lambda x : {'label' : answer_map[x['answer']]}
            )

        logger.info(
            f"FinetuneDataset '{task_name}' loaded - "
            f"split='{split}', examples={len(self.dataset)}, "
            f"type={self.config.task_type.value}"
        )

    def __len__(self) -> int : 
        return len(self.dataset)

    def __getitem__(self, idx:int) -> dict[str, torch.Tensor]:
        example = self.dataset[idx]

        match self.config.task_type:

            case TaskType.CLASSIFICATION:
                token_ids = _format_classification(
                    text=example[self.config.text_columns[0]], 
                    tokenizer=self.tokenizer, 
                    max_seq_len=self.max_seq_len
                )
                return {
                    "input_ids": torch.tensor(token_ids, dtype=torch.long), 
                    "label": torch.tensor(example[self.config.label_column], dtype=torch.long)
                }

            case TaskType.ENTAILMENT:
                token_ids = _format_entailment(
                        text1=example[self.config.text_columns[0]], 
                        text2=example[self.config.text_columns[1]], 
                        tokenizer=self.tokenizer, 
                        max_seq_len=self.max_seq_len
                    ) 
                return {
                    "input_ids": torch.tensor(token_ids, dtype=torch.long), 
                    "label": torch.tensor(example[self.config.label_column], dtype=torch.long)
                }

            case TaskType.SIMILARITY:
                ids_1, ids_2 = _format_similarity(
                        text1=example[self.config.text_columns[0]], 
                        text2=example[self.config.text_columns[1]], 
                        tokenizer=self.tokenizer, 
                        max_seq_len=self.max_seq_len
                    )
                # For regression (STS-B), label is a float
                label_dtype = torch.float if self.config.is_regression else torch.long
                return {
                    "input_ids_1": torch.tensor(ids_1, dtype=torch.long), 
                    "input_ids_2": torch.tensor(ids_2, dtype=torch.long), 
                    "label": torch.tensor(example[self.config.label_column], dtype=label_dtype)
                }

            case TaskType.MULTIPLE_CHOICE:
                # RACE: context = article + " " + question
                context = example['article'] + " " + example['question']
                options = example['options']
                all_ids = _format_multiple_choice(
                        context=context, 
                        options=options, 
                        tokenizer=self.tokenizer, 
                        max_seq_len=self.max_seq_len
                    )
                return {
                    "input_ids": torch.tensor(all_ids, dtype=torch.long),# shape: (num_options, max_seq_len)
                    "label": torch.tensor(example['label'], dtype=torch.long)
                }

# Factory function to get dataloaders 
def get_finetune_dataloaders(
    task_name: str, 
    tokenizer: BPETokenizer, 
    batch_size:int = 32, 
    max_seq_len:int = 512, 
    num_workers:int = 4
) -> dict[str, DataLoader]:

    """
    Creates DataLoaders for a fine-tuning task ( train + validation splits)

    Returns: 
        A Dict : {"train": train_loader, "validation": val_loader}
        For MNLI, also includes "validation_mismatched"
    """

    splits = ["train", "validation"]

    # MNLI has matched and mismatched validation sets 
    if task_name == "mnli":
        splits = ["train", "validation_matched", "validation_mismatched"]

    loaders:dict[str, DataLoader] = {}
    for split in splits: 
        dataset = FinetuneDataset(
            task_name=task_name, 
            tokenizer=tokenizer, 
            split=split, 
            max_seq_len=max_seq_len
        )
        loaders[split] = DataLoader(
            dataset=dataset, 
            batch_size= batch_size, 
            shuffle=(split=="train"), 
            num_workers=num_workers, 
            pin_memory=True, 
            drop_last=(split == "train")
        )

    return loaders

