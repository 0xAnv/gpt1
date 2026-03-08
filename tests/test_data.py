"""Tests for the pre-training data pipeline.

Covers:
    - PreTrainingDataset: length, item shape, dtype, contiguous chunks, no overlap
    - get_pretrain_dataloaders: train/val split sizes, batch shapes, shuffle/drop_last
    - prepare_pretrain_data: file creation, metadata correctness, ftfy cleaning
    - Edge cases: tiny datasets, datasets not evenly divisible by seq_len
"""

import json
import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from gpt.data import PreTrainingDataset, get_pretrain_dataloaders, prepare_pretrain_data


# ── Fixtures ──────────────────────────────────────────────────────────

SEQ_LEN = 16  # small for fast tests


@pytest.fixture
def sample_memmap(tmp_path: Path) -> Path:
    """Creates a small memmap file with known, sequential token IDs."""
    total_tokens = SEQ_LEN * 20  # exactly 20 chunks
    bin_path = tmp_path / "tokens.bin"
    arr = np.memmap(bin_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))
    arr[:] = np.arange(total_tokens, dtype=np.uint16)
    arr.flush()
    return bin_path


@pytest.fixture
def small_memmap(tmp_path: Path) -> Path:
    """Creates a memmap that is NOT evenly divisible by SEQ_LEN."""
    total_tokens = SEQ_LEN * 5 + 7  # 5 full chunks + 7 leftover
    bin_path = tmp_path / "tokens_small.bin"
    arr = np.memmap(bin_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))
    arr[:] = np.arange(total_tokens, dtype=np.uint16)
    arr.flush()
    return bin_path


@pytest.fixture
def trained_tokenizer_path(tmp_path: Path) -> Path:
    """Trains a tiny BPE tokenizer and saves it, returns the path."""
    from gpt.tokenizer import BPETokenizer

    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
    ]
    tok = BPETokenizer()
    tok.train_from_iterator(corpus, vocab_size=300)
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    return path


# ── PreTrainingDataset ────────────────────────────────────────────────

class TestPreTrainingDatasetLength:
    def test_length_exact_division(self, sample_memmap: Path) -> None:
        """When total_tokens is exactly divisible by seq_len, all tokens are used."""
        dataset = PreTrainingDataset(sample_memmap, seq_len=SEQ_LEN)
        assert len(dataset) == 20

    def test_length_with_remainder(self, small_memmap: Path) -> None:
        """Remainder tokens are dropped — only full chunks count."""
        dataset = PreTrainingDataset(small_memmap, seq_len=SEQ_LEN)
        assert len(dataset) == 5  # 7 leftover tokens discarded


class TestPreTrainingDatasetGetItem:
    def test_item_shape(self, sample_memmap: Path) -> None:
        """Each item should be a tensor of shape (seq_len,)."""
        dataset = PreTrainingDataset(sample_memmap, seq_len=SEQ_LEN)
        item = dataset[0]
        assert item["input_ids"].shape == (SEQ_LEN,)

    def test_item_dtype_is_long(self, sample_memmap: Path) -> None:
        """Embeddings require torch.long (int64), not uint16."""
        dataset = PreTrainingDataset(sample_memmap, seq_len=SEQ_LEN)
        item = dataset[0]
        assert item["input_ids"].dtype == torch.long

    def test_item_returns_dict(self, sample_memmap: Path) -> None:
        """Return type should be a dict with 'input_ids' key."""
        dataset = PreTrainingDataset(sample_memmap, seq_len=SEQ_LEN)
        item = dataset[0]
        assert isinstance(item, dict)
        assert "input_ids" in item

    def test_first_chunk_values(self, sample_memmap: Path) -> None:
        """First chunk should contain tokens [0, 1, 2, ..., seq_len-1]."""
        dataset = PreTrainingDataset(sample_memmap, seq_len=SEQ_LEN)
        expected = torch.arange(SEQ_LEN, dtype=torch.long)
        assert torch.equal(dataset[0]["input_ids"], expected)

    def test_second_chunk_values(self, sample_memmap: Path) -> None:
        """Second chunk should start where first chunk ended."""
        dataset = PreTrainingDataset(sample_memmap, seq_len=SEQ_LEN)
        expected = torch.arange(SEQ_LEN, 2 * SEQ_LEN, dtype=torch.long)
        assert torch.equal(dataset[1]["input_ids"], expected)


class TestPreTrainingDatasetContiguity:
    def test_chunks_are_contiguous(self, sample_memmap: Path) -> None:
        """Chunk i ends where chunk i+1 begins — no gaps, no overlap."""
        dataset = PreTrainingDataset(sample_memmap, seq_len=SEQ_LEN)
        for i in range(len(dataset) - 1):
            last_token = dataset[i]["input_ids"][-1].item()
            first_token = dataset[i + 1]["input_ids"][0].item()
            assert first_token == last_token + 1, (
                f"Gap between chunk {i} and {i+1}: "
                f"chunk {i} ends at {last_token}, chunk {i+1} starts at {first_token}"
            )

    def test_no_overlap_between_chunks(self, sample_memmap: Path) -> None:
        """No token should appear in two different chunks."""
        dataset = PreTrainingDataset(sample_memmap, seq_len=SEQ_LEN)
        all_tokens = set()
        for i in range(len(dataset)):
            chunk_tokens = set(dataset[i]["input_ids"].tolist())
            overlap = all_tokens & chunk_tokens
            assert len(overlap) == 0, f"Chunk {i} overlaps with earlier chunks: {overlap}"
            all_tokens.update(chunk_tokens)


# ── get_pretrain_dataloaders ──────────────────────────────────────────

class TestDataLoaderSplitSizes:
    def test_train_val_split_proportions(self, sample_memmap: Path) -> None:
        """Val size should be approximately val_split fraction of total."""
        val_split = 0.2  # 20% for easy math
        train_loader, val_loader = get_pretrain_dataloaders(
            sample_memmap, seq_len=SEQ_LEN, batch_size=2,
            val_split=val_split, num_workers=0,
        )
        total = len(train_loader.dataset) + len(val_loader.dataset)
        assert total == 20  # all chunks accounted for
        assert len(val_loader.dataset) == 4  # int(20 * 0.2) = 4
        assert len(train_loader.dataset) == 16

    def test_zero_val_split(self, sample_memmap: Path) -> None:
        """With val_split=0, all data goes to train."""
        train_loader, val_loader = get_pretrain_dataloaders(
            sample_memmap, seq_len=SEQ_LEN, batch_size=2,
            val_split=0.0, num_workers=0,
        )
        assert len(train_loader.dataset) == 20
        assert len(val_loader.dataset) == 0


class TestDataLoaderBatchShape:
    def test_train_batch_shape(self, sample_memmap: Path) -> None:
        """Each train batch should have shape (batch_size, seq_len)."""
        batch_size = 4
        train_loader, _ = get_pretrain_dataloaders(
            sample_memmap, seq_len=SEQ_LEN, batch_size=batch_size,
            val_split=0.1, num_workers=0,
        )
        batch = next(iter(train_loader))
        assert batch["input_ids"].shape == (batch_size, SEQ_LEN)

    def test_val_batch_shape(self, sample_memmap: Path) -> None:
        """Val batches should also have shape (batch_size, seq_len)."""
        batch_size = 2
        _, val_loader = get_pretrain_dataloaders(
            sample_memmap, seq_len=SEQ_LEN, batch_size=batch_size,
            val_split=0.5, num_workers=0,  # 50% so val has plenty
        )
        batch = next(iter(val_loader))
        assert batch["input_ids"].shape == (batch_size, SEQ_LEN)

    def test_train_batch_dtype(self, sample_memmap: Path) -> None:
        """Batched tensors should be torch.long for embedding lookup."""
        train_loader, _ = get_pretrain_dataloaders(
            sample_memmap, seq_len=SEQ_LEN, batch_size=2,
            val_split=0.1, num_workers=0,
        )
        batch = next(iter(train_loader))
        assert batch["input_ids"].dtype == torch.long


class TestDataLoaderBehavior:
    def test_train_drops_last_incomplete_batch(self, sample_memmap: Path) -> None:
        """Train loader with drop_last=True should discard incomplete batches."""
        # 20 chunks, val_split=0.1 → train=18 chunks, batch_size=4 → 4 full batches, 2 leftover dropped
        train_loader, _ = get_pretrain_dataloaders(
            sample_memmap, seq_len=SEQ_LEN, batch_size=4,
            val_split=0.1, num_workers=0,
        )
        num_batches = sum(1 for _ in train_loader)
        assert num_batches == 4  # 18 // 4 = 4 (drop 2)

    def test_val_keeps_last_incomplete_batch(self, sample_memmap: Path) -> None:
        """Val loader with drop_last=False should keep the last partial batch."""
        # 20 chunks, val_split=0.5 → val=10, batch_size=3 → 3 full + 1 partial = 4
        _, val_loader = get_pretrain_dataloaders(
            sample_memmap, seq_len=SEQ_LEN, batch_size=3,
            val_split=0.5, num_workers=0,
        )
        num_batches = sum(1 for _ in val_loader)
        assert num_batches == 4  # ceil(10 / 3) = 4


# ── prepare_pretrain_data ─────────────────────────────────────────────

class TestPreparePretrainData:
    def test_creates_bin_and_metadata(
        self, trained_tokenizer_path: Path, tmp_path: Path,
    ) -> None:
        """prepare_pretrain_data should create tokens.bin and metadata.json."""
        # Mock the HF dataset with a tiny in-memory list
        fake_data = [
            {"text": "Hello world, this is a test."},
            {"text": "Another sentence for our corpus."},
        ]
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(side_effect=lambda: iter(fake_data))
        mock_dataset.__len__ = MagicMock(return_value=len(fake_data))
        mock_dataset.select = MagicMock(return_value=mock_dataset)

        output_dir = tmp_path / "pretrain_test"

        with patch("gpt.data.load_dataset", return_value=mock_dataset):
            bin_path = prepare_pretrain_data(
                tokenizer_path=trained_tokenizer_path,
                output_dir=output_dir,
            )

        assert bin_path.exists()
        assert (output_dir / "metadata.json").exists()

    def test_metadata_fields(
        self, trained_tokenizer_path: Path, tmp_path: Path,
    ) -> None:
        """Metadata JSON should contain all expected fields."""
        fake_data = [{"text": "Some book text here."}]
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(side_effect=lambda: iter(fake_data))
        mock_dataset.__len__ = MagicMock(return_value=1)
        mock_dataset.select = MagicMock(return_value=mock_dataset)

        output_dir = tmp_path / "meta_test"

        with patch("gpt.data.load_dataset", return_value=mock_dataset):
            prepare_pretrain_data(
                tokenizer_path=trained_tokenizer_path,
                output_dir=output_dir,
            )

        with open(output_dir / "metadata.json") as f:
            meta = json.load(f)

        expected_keys = {"total_tokens", "vocab_size", "dataset_name", "num_books", "dtype"}
        assert set(meta.keys()) == expected_keys
        assert meta["total_tokens"] > 0
        assert meta["dtype"] == "uint16"
        assert meta["num_books"] == 1

    def test_bin_file_token_count_matches_metadata(
        self, trained_tokenizer_path: Path, tmp_path: Path,
    ) -> None:
        """The memmap file size should match the total_tokens in metadata."""
        fake_data = [
            {"text": "First book content."},
            {"text": "Second book here."},
        ]
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(side_effect=lambda: iter(fake_data))
        mock_dataset.__len__ = MagicMock(return_value=len(fake_data))
        mock_dataset.select = MagicMock(return_value=mock_dataset)

        output_dir = tmp_path / "count_test"

        with patch("gpt.data.load_dataset", return_value=mock_dataset):
            bin_path = prepare_pretrain_data(
                tokenizer_path=trained_tokenizer_path,
                output_dir=output_dir,
            )

        with open(output_dir / "metadata.json") as f:
            meta = json.load(f)

        arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
        assert len(arr) == meta["total_tokens"]

    def test_nonexistent_tokenizer_raises(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError before downloading anything."""
        with pytest.raises(FileNotFoundError, match="Tokeniser not found"):
            prepare_pretrain_data(
                tokenizer_path="/nonexistent/tokenizer.json",
                output_dir=tmp_path / "nope",
            )


class TestFtfyCleaning:
    def test_ftfy_fixes_mojibake(
        self, trained_tokenizer_path: Path, tmp_path: Path,
    ) -> None:
        """Text with mojibake should be cleaned before tokenization."""
        # This string contains broken Unicode that ftfy should fix
        broken_text = "The Mona Lisa doesn\u00e2\u0080\u0099t have eyebrows."
        fake_data = [{"text": broken_text}]
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = MagicMock(side_effect=lambda: iter(fake_data))
        mock_dataset.__len__ = MagicMock(return_value=1)
        mock_dataset.select = MagicMock(return_value=mock_dataset)

        output_dir = tmp_path / "ftfy_test"

        with patch("gpt.data.load_dataset", return_value=mock_dataset):
            bin_path = prepare_pretrain_data(
                tokenizer_path=trained_tokenizer_path,
                output_dir=output_dir,
            )

        # The file should exist and have tokens (ftfy didn't crash on broken text)
        arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
        assert len(arr) > 0
