import json 
import logging 
import numpy as np 
from pathlib import Path 

import ftfy # fixes broken unicode texts
from datasets import load_dataset # bookcorpusopen dataset

from gpt.tokenizer import BPETokenizer

# pretrain data preparation function 
def prepare_pretrain_data(
    tokenizer_path:str | Path, 
    output_dir:str | Path = "data/pretrain", 
    dataset_name:str = "lucadiliello/bookcorpusopen", 
    dataset_split:str = "train", 
    max_books:int | None = None
) -> Path : 
    pass

