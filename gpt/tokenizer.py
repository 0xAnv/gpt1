"""
Core BPE tokenizer module for GPT - 1.
Handles training a Byte-Pair Encoding model from scratch and wraps
the tokenizer for efficient inference and padded batched encoding.
"""

import logging 
from pathlib import Path 
from typing import Iterable 

# tokeniser specifics 
from tokenizers import Tokenizer 
from tokenizers.models import BPE 
from tokenizers.trainers import BpeTrainer 
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing

# logger 
logger = logging.getLogger(__name__)

# BPE tokeniser 
class BPETokenizer:
    """
    GPT-1 style Byte Pair encoding (BPE) tokenizer.
    We can load from a saved tokeniser or train a new one 
    """
    
    # constants 
    START_TOKEN : str = "<s>"
    END_TOKEN : str = "</s>" 
    PAD_TOKEN : str = "<pad>" 
    CLS_TOKEN : str = "<cls>"
    UNK_TOKEN : str = "<unk>"

    SPECIAL_TOKENS : list[str] = [
        START_TOKEN, END_TOKEN, PAD_TOKEN, CLS_TOKEN, UNK_TOKEN
    ] 

    def __init__(self, tokeniser_path: str | Path | None = None) -> None: 
        """
        Intialize the tokeniser 
        Args:
            tokeniser_path: If provided, loads a pre-trainer tokeniser from this path. 
            Otherwise, initializes an empty tokenizer for training.
        """
        if tokeniser_path:
            path_str = str(tokeniser_path) 
            if not Path(path_str).exists():
                raise FileNotFoundError(f"Tokenizer file not found: {path_str}")
            self.tokeniser = Tokenizer.from_file(path=path_str)
            logger.info(f"Loaded our tokenizer from {path_str} with vocab size {self.vocab_size}")
        else: 
            self.tokeniser = Tokenizer(BPE(unk_token=self.UNK_TOKEN))
            self._setup_pipeline() 
            logger.info("Initialized a fresh BPE Tokenizer. Ready for training.")

    @property
    def vocab_size(self) -> int : 
        """Returns the current size of tokenizer's vocab""" 
        return self.tokeniser.get_vocab_size() 
    
    def _setup_pipeline(self) -> None: 
        """ 
        Configures the tokenisation pipeline (Pre-tokenization, post-processingm decoding). 
        Uses Bytelevel, which operates directly on bytes, handling all Unicode characters (standard approach for GPT models).
        """
        # pre-tokenisation
        # spliting input into words (relies on space and punctuation at byte level)
        self.tokeniser.pre_tokenizer = ByteLevel(add_prefix_space=False)

        # Decoder : reverses the bytelevel tokenisation 
        self.tokeniser.decoder = ByteLevelDecoder() 
    
    def train_from_iterator(self, iterator: Iterable[str], vocab_size:int = 40_000, save_path: str | Path | None = None) -> None:
        """
        Trains the BPE Tokeniser from an iterator of strings
        Args:
            iterator: An iterable yielding strings ( corpus)
            vocab_size: The target vocabulary size
            save_path: Optional path to save tokeniser JSON file after training
        """
        logger.info(f"Training tokeniser with target vocab size: {vocab_size=}")

        trainer = BpeTrainer(
            vocab_size=vocab_size, 
            special_tokens=self.SPECIAL_TOKENS,
            show_progress=True 
        )

        self.tokeniser.train_from_iterator(iterator, trainer=trainer) 

        logger.info(f"Training Complete. Final Vocab Size: {self.vocab_size}")

        if save_path: 
            self.save(save_path) 
            # PEP 8 style discourage body on same line as if statements

    def save(self, save_path: str | Path) -> None:
        """saves the trained tokeniser to JSON file""" 
        path_str = str(save_path)
        self.tokeniser.save(path_str)
        logger.info(f"Tokeniser saved to {path_str=}")

    def encode(self, text: str | list[str], max_seq_len : int = 512, pad:bool = True, truncate: bool = True) -> list[int]|list[list[int]]:
        """
        Encodes a single string or batch of strings into token IDs,
        Args: 
            text: A single string or list of strings to encode 
            max_seq_len: maximum sequence length  (used for pad & truncation) 
            pad: whether to pad or not 
            truncatte: to truncate the sequence
        Returns: 
            A list of token IDs or list of lists of token ids (batch)
        """
        if truncate: # PEP 8
            self.tokeniser.enable_truncation(max_length=max_seq_len) 
        else: 
            self.tokeniser.no_truncation() 
        
        if pad:
            pad_id = self.tokeniser.token_to_id(self.PAD_TOKEN) 
            self.tokeniser.enable_padding(
                direction="right", 
                pad_id=pad_id, 
                pad_token=self.PAD_TOKEN, 
                length=max_seq_len
            )
        else : 
            self.tokeniser.no_padding()

        # batch encoding / single string encoding 
        if isinstance(text, list):
            encodings = self.tokeniser.encode_batch(text) 
            return [encoding.ids for encoding in encodings]
        
        # single continous sequence of str type 
        return self.tokeniser.encode(text).ids

    def decode(self, token_ids: list[int], skip_special_tokens:bool = True) -> str: 
        """
        Decodes a sequence of token IDs back into a string. 
        Args: 
            token_ids: A list of token IDs to decode. 
            skip_special_tokens: If True, removes padding, start/end tokens, etc.
        Returns:
            The decoded text string.
        """
        return self.tokeniser.decode(token_ids, skip_special_tokens=skip_special_tokens)

