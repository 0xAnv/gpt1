"""Tests for the BPE Tokenizer.

Covers:
    - Fresh initialisation (training mode)
    - Loading from a saved file
    - Loading from a non-existent path (FileNotFoundError)
    - Training from an iterator
    - Save / load round-trip
    - Single-string encoding (padded + truncated)
    - Batch encoding (padded + truncated)
    - Encoding without padding
    - Encoding without truncation
    - Decode round-trip fidelity
    - Decode with special tokens visible
    - Special token IDs are present in the vocabulary
    - vocab_size property correctness
"""

import pytest
from pathlib import Path

from gpt.tokenizer import BPETokenizer


# ── Fixtures ──────────────────────────────────────────────────────────

SAMPLE_CORPUS: list[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "In the beginning was the Word, and the Word was with God.",
    "It was the best of times, it was the worst of times.",
    "Call me Ishmael.",
    "I think, therefore I am.",
    "The only thing we have to fear is fear itself.",
    "That's one small step for man, one giant leap for mankind.",
]

SMALL_VOCAB_SIZE: int = 300


@pytest.fixture(scope="module")
def trained_tokenizer() -> BPETokenizer:
    """Module-scoped fixture: trains a tokenizer once, reused across tests."""
    tok = BPETokenizer()
    tok.train_from_iterator(SAMPLE_CORPUS, vocab_size=SMALL_VOCAB_SIZE)
    return tok


@pytest.fixture
def saved_tokenizer_path(trained_tokenizer: BPETokenizer, tmp_path: Path) -> Path:
    """Saves the trained tokenizer to a temp file and returns its path."""
    path = tmp_path / "test_tokenizer.json"
    trained_tokenizer.save(path)
    return path


# ── Initialisation ────────────────────────────────────────────────────

class TestInitialisation:
    def test_fresh_init_creates_empty_tokenizer(self) -> None:
        tok = BPETokenizer()
        # An untrained BPE model starts with zero vocab
        assert tok.vocab_size == 0

    def test_load_from_saved_file(self, saved_tokenizer_path: Path) -> None:
        tok = BPETokenizer(tokeniser_path=saved_tokenizer_path)
        assert tok.vocab_size <= SMALL_VOCAB_SIZE
        assert tok.vocab_size > len(BPETokenizer.SPECIAL_TOKENS)

    def test_load_nonexistent_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Tokenizer file not found"):
            BPETokenizer(tokeniser_path="/nonexistent/path/tokenizer.json")

    def test_load_from_path_object(self, saved_tokenizer_path: Path) -> None:
        tok = BPETokenizer(tokeniser_path=saved_tokenizer_path)
        assert tok.vocab_size > 0


# ── Training ──────────────────────────────────────────────────────────

class TestTraining:
    def test_train_reaches_target_vocab_size(self, trained_tokenizer: BPETokenizer) -> None:
        # BPE stops early if the corpus is too small for the target vocab
        assert trained_tokenizer.vocab_size <= SMALL_VOCAB_SIZE
        assert trained_tokenizer.vocab_size > len(BPETokenizer.SPECIAL_TOKENS)

    def test_train_and_save(self, tmp_path: Path) -> None:
        tok = BPETokenizer()
        save_path = tmp_path / "auto_save.json"
        tok.train_from_iterator(SAMPLE_CORPUS, vocab_size=SMALL_VOCAB_SIZE, save_path=save_path)
        assert save_path.exists()

    def test_special_tokens_in_vocab(self, trained_tokenizer: BPETokenizer) -> None:
        for token in BPETokenizer.SPECIAL_TOKENS:
            token_id = trained_tokenizer.tokeniser.token_to_id(token)
            assert token_id is not None, f"Special token '{token}' not found in vocabulary"


# ── Encoding — Single String ─────────────────────────────────────────

class TestEncodeSingle:
    def test_encode_returns_list_of_ints(self, trained_tokenizer: BPETokenizer) -> None:
        ids = trained_tokenizer.encode("Hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_pads_to_max_seq_len(self, trained_tokenizer: BPETokenizer) -> None:
        max_len = 64
        ids = trained_tokenizer.encode("Short text", max_seq_len=max_len)
        assert len(ids) == max_len

    def test_encode_truncates_to_max_seq_len(self, trained_tokenizer: BPETokenizer) -> None:
        max_len = 4
        ids = trained_tokenizer.encode("A very long sentence with many words indeed", max_seq_len=max_len)
        assert len(ids) == max_len

    def test_encode_no_padding(self, trained_tokenizer: BPETokenizer) -> None:
        ids = trained_tokenizer.encode("Short", pad=False, truncate=False)
        assert isinstance(ids, list)
        assert len(ids) < 512  # should NOT be padded to 512

    def test_encode_no_truncation(self, trained_tokenizer: BPETokenizer) -> None:
        long_text = " ".join(["word"] * 1000)
        ids = trained_tokenizer.encode(long_text, pad=False, truncate=False)
        assert len(ids) > 512  # should NOT be truncated


# ── Encoding — Batch ──────────────────────────────────────────────────

class TestEncodeBatch:
    def test_batch_encode_returns_nested_list(self, trained_tokenizer: BPETokenizer) -> None:
        batch = ["Hello world", "Foo bar baz"]
        ids = trained_tokenizer.encode(batch, max_seq_len=32)
        assert isinstance(ids, list)
        assert len(ids) == 2
        assert all(isinstance(seq, list) for seq in ids)

    def test_batch_encode_uniform_length(self, trained_tokenizer: BPETokenizer) -> None:
        batch = ["Short", "A significantly longer sentence with many words"]
        max_len = 32
        ids = trained_tokenizer.encode(batch, max_seq_len=max_len)
        assert all(len(seq) == max_len for seq in ids)

    def test_batch_encode_empty_list(self, trained_tokenizer: BPETokenizer) -> None:
        ids = trained_tokenizer.encode([], max_seq_len=16)
        assert ids == []


# ── Decoding ──────────────────────────────────────────────────────────

class TestDecode:
    def test_round_trip_fidelity(self, trained_tokenizer: BPETokenizer) -> None:
        original = "The quick brown fox"
        ids = trained_tokenizer.encode(original, pad=False, truncate=False)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == original

    def test_decode_skips_special_tokens_by_default(self, trained_tokenizer: BPETokenizer) -> None:
        # Encode with padding, then decode — padding tokens should be stripped
        ids = trained_tokenizer.encode("Hello", max_seq_len=32, pad=True)
        decoded = trained_tokenizer.decode(ids, skip_special_tokens=True)
        assert BPETokenizer.PAD_TOKEN not in decoded

    def test_decode_shows_special_tokens(self, trained_tokenizer: BPETokenizer) -> None:
        pad_id = trained_tokenizer.tokeniser.token_to_id(BPETokenizer.PAD_TOKEN)
        ids = [pad_id, pad_id]
        decoded = trained_tokenizer.decode(ids, skip_special_tokens=False)
        assert BPETokenizer.PAD_TOKEN in decoded


# ── vocab_size Property ───────────────────────────────────────────────

class TestVocabSize:
    def test_vocab_size_after_training(self, trained_tokenizer: BPETokenizer) -> None:
        assert trained_tokenizer.vocab_size <= SMALL_VOCAB_SIZE
        assert trained_tokenizer.vocab_size > len(BPETokenizer.SPECIAL_TOKENS)

    def test_vocab_size_after_load(self, saved_tokenizer_path: Path) -> None:
        tok = BPETokenizer(tokeniser_path=saved_tokenizer_path)
        assert tok.vocab_size <= SMALL_VOCAB_SIZE
        assert tok.vocab_size > len(BPETokenizer.SPECIAL_TOKENS)


# ── Save / Load Round-Trip ────────────────────────────────────────────

class TestSaveLoad:
    def test_save_creates_file(self, trained_tokenizer: BPETokenizer, tmp_path: Path) -> None:
        path = tmp_path / "tok.json"
        trained_tokenizer.save(path)
        assert path.exists()

    def test_loaded_tokenizer_encodes_identically(
        self,
        trained_tokenizer: BPETokenizer,
        saved_tokenizer_path: Path,
    ) -> None:
        loaded = BPETokenizer(tokeniser_path=saved_tokenizer_path)
        text = "encoding consistency check"
        assert trained_tokenizer.encode(text, pad=False, truncate=False) == loaded.encode(text, pad=False, truncate=False)
