# 🧠 GPT-1 From Scratch — Implementation Checklist

> **Paper:** "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)
> **Goal:** Reproduce GPT-1 faithfully → benchmark on the same evals → introduce modern improvements for SOTA.
> **Hardware:** WSL / NVIDIA RTX 3060 12 GB VRAM
> **Stack:** Python 3.12+, PyTorch, `uv` (package manager), `wandb` (experiment tracking)

---

## Phase 0 — Project Scaffolding & Tooling

- [ ] **0.1 — Project structure**
  - Set up a clean directory layout:
    ```
    gpt1/
    ├── notebooks/           # Jupyter notebooks for intuition-building & experiments
    │   ├── 01_tokenizer.ipynb
    │   ├── 02_attention.ipynb
    │   ├── 03_transformer_block.ipynb
    │   ├── 04_gpt_model.ipynb
    │   ├── 05_pretraining.ipynb
    │   └── 06_finetuning.ipynb
    ├── gpt1/                # Source package
    │   ├── __init__.py
    │   ├── tokenizer.py
    │   ├── model.py
    │   ├── data.py
    │   ├── train.py
    │   ├── finetune.py
    │   ├── evaluate.py
    │   └── utils.py
    ├── configs/             # YAML/JSON config files for experiments
    │   ├── pretrain.yaml
    │   └── finetune.yaml
    ├── scripts/             # CLI scripts for training, eval, etc.
    │   ├── pretrain.sh
    │   └── finetune.sh
    ├── evals/               # Evaluation harnesses per benchmark
    │   ├── glue.py
    │   ├── race.py
    │   └── storycloze.py
    ├── tests/               # Unit tests
    ├── data/                # Downloaded datasets (gitignored)
    ├── checkpoints/         # Saved model weights (gitignored)
    ├── pyproject.toml
    ├── implementation.md    # ← You are here
    └── README.md
    ```
  - Create `.gitignore` entries for `data/`, `checkpoints/`, `.venv/`, `wandb/`, `__pycache__/`

- [ ] **0.2 — Dependency management with `uv`**
  - Initialize with `uv init` (already done)
  - Add core dependencies:
    - `torch` (with CUDA 12.x support for RTX 3060)
    - `tokenizers` (HuggingFace, fast Rust-backed BPE training & inference)
    - `transformers` (only for tokenizer reference / comparison, not for the model)
    - `datasets` (HuggingFace, for downloading benchmark data)
    - `wandb`
    - `jupyter`, `ipykernel`
    - `pyyaml` (config loading)
    - `tqdm`
    - `scikit-learn` (for eval metrics like F1, Matthews corr)
    - `scipy` (for Spearman/Pearson on STS-B)

- [x] **0.3 — Weights & Biases integration**
  - `wandb login` and create project `gpt1-from-scratch`
  - Set up a `wandb.init()` wrapper in `gpt1/utils.py` that logs:
    - All hyperparameters (config dict)
    - Training loss curves (per step & per epoch)
    - Validation losses & perplexities
    - Evaluation benchmark scores
    - GPU memory usage, throughput (tokens/sec)
  - Plan: Every experiment gets a descriptive `wandb` run name (e.g., `pretrain-bookscorpus-v1`, `finetune-mnli-v1`)

- [ ] **0.4 — Config system**
  - Design a config dataclass or YAML schema covering:
    - Model hyperparams (n_layers, n_heads, d_model, d_ff, vocab_size, max_seq_len, dropout)
    - Training hyperparams (batch_size, lr, warmup_steps, total_steps, weight_decay, grad_clip)
    - Fine-tuning hyperparams (task name, num_epochs, lr, lambda_aux, label_smoothing)
    - Data paths, checkpoint paths, wandb project name

---

## Phase 1 — Tokenizer (BPE)

> **Paper spec:** Byte-Pair Encoding (BPE), vocabulary size ~40,000, based on the `ftfy` + `spaCy` tokenizer pipeline used by the original OpenAI implementation.

- [ ] **1.1 — Understand BPE** *(notebook: `01_tokenizer.ipynb`)*
  - Study the BPE algorithm from the original Sennrich et al. (2016) paper
  - Implement a toy BPE from scratch on a small corpus to build intuition
  - Understand the difference between character-level, word-level, subword (BPE) tokenization
  - Visualize merge operations and vocabulary growth

- [ ] **1.2 — Decide on tokenizer strategy**
  - **Option A (Recommended): Train a custom BPE using HuggingFace `tokenizers`** library
    - Use the `tokenizers.Tokenizer` + `tokenizers.models.BPE` API (or the legacy `ByteLevelBPETokenizer`)
    - Train on BooksCorpus (or your chosen pre-training corpus) with ~40,000 merges
    - Steps:
      1. Prepare a text iterator or list of text files from the corpus
      2. Configure pre-tokenizer: `ByteLevel` (GPT-style) or `Whitespace` + `Punctuation`
      3. Configure trainer: `BpeTrainer(vocab_size=40000, special_tokens=["<s>", "</s>", "<pad>", "<cls>", "<unk>"])`
      4. Train: `tokenizer.train_from_iterator(corpus_iterator)` or `tokenizer.train(files)`
      5. Save: `tokenizer.save("tokenizer.json")` — single file, fast to load
    - Advantages: extremely fast (Rust backend), battle-tested, supports all BPE variants, easy serialization
  - **Option B:** Use the OpenAI GPT-1 tokenizer (available via `transformers` library's `OpenAIGPTTokenizer`) — for exact reproduction / comparison baseline
  - **Option C:** Use `tiktoken` (OpenAI's fast BPE implementation) — note: this uses a different encoding than GPT-1
  - **Recommendation:** Use Option A to train your own tokenizer for the learning experience and flexibility. Keep Option B on hand to compare tokenization quality against the original.
  - Vocabulary should include special tokens: `<s>` (start), `</s>` (end/separator), `<pad>`, `<cls>` (for fine-tuning classification), `<unk>` (unknown)
  - The original paper uses `ftfy` for text cleaning and `spaCy` for tokenization before BPE

- [ ] **1.3 — Implement tokenizer wrapper** *(file: `gpt1/tokenizer.py`)*
  - `encode(text) → List[int]`
  - `decode(token_ids) → str`
  - `vocab_size` property
  - Special token IDs as constants
  - Efficient batched encoding with padding/truncation to `max_seq_len = 512`

---

## Phase 2 — Model Architecture

> **Paper spec:** 12-layer Transformer decoder, 768 hidden dim, 12 attention heads, 3072 FFN inner dim, GELU activation, learned positional embeddings, 117M parameters.

### 2.1 — Attention Mechanism

- [ ] **2.1.1 — Understand multi-head self-attention** *(notebook: `02_attention.ipynb`)*
  - Implement scaled dot-product attention from scratch: `Attention(Q, K, V) = softmax(QK^T / √d_k) V`
  - Implement the **causal mask** (lower-triangular mask to prevent attending to future tokens)
  - Implement multi-head attention: split into `h` heads, apply attention independently, concatenate & project
  - Verify shapes at every step
  - Notebook experiments:
    - Visualize attention patterns on a toy sequence
    - Compare masked vs unmasked attention
    - Show how head splitting works dimensionally

- [ ] **2.1.2 — Production attention module** *(file: `gpt1/model.py`)*
  - `MultiHeadAttention(d_model=768, n_heads=12, dropout=0.1)`
  - Causal masking built-in
  - Residual connection handled externally (in the block)

### 2.2 — Feed-Forward Network (Position-wise FFN)

- [ ] **2.2.1 — Understand GELU and FFN** *(notebook: `02_attention.ipynb` or `03_transformer_block.ipynb`)*
  - Implement GELU activation: `GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`
  - Compare GELU vs ReLU activation curves visually
  - Build FFN: `Linear(768, 3072) → GELU → Linear(3072, 768)`
  - Understand why FFN inner dim is 4× the model dim

- [ ] **2.2.2 — Production FFN module** *(file: `gpt1/model.py`)*
  - `PositionwiseFFN(d_model=768, d_ff=3072, dropout=0.1)`

### 2.3 — Transformer Decoder Block

- [ ] **2.3.1 — Understand the Transformer block** *(notebook: `03_transformer_block.ipynb`)*
  - **Pre-Norm vs Post-Norm:** The original GPT-1 uses **post-norm** (LayerNorm after residual add), but some implementations use pre-norm. Identify which one the paper uses.
    - Paper uses: `x = LayerNorm(x + Sublayer(x))` — **post-norm** (like original Transformer)
  - Build the block:
    ```
    h = x + MultiHeadAttention(LayerNorm_maybe(x))   # with causal mask
    output = h + FFN(LayerNorm_maybe(h))
    ```
  - Verify parameter count for a single block
  - Experiment: Apply dropout to attention weights and FFN output

- [ ] **2.3.2 — Production decoder block** *(file: `gpt1/model.py`)*
  - `TransformerBlock(d_model=768, n_heads=12, d_ff=3072, dropout=0.1)`
  - Includes: masked multi-head self-attention → add & norm → FFN → add & norm

### 2.4 — Full GPT-1 Model

- [ ] **2.4.1 — Embeddings** *(notebook: `04_gpt_model.ipynb`)*
  - **Token embedding:** `nn.Embedding(vocab_size, d_model)` — maps token IDs to 768-dim vectors
  - **Positional embedding:** `nn.Embedding(max_seq_len, d_model)` — LEARNED (not sinusoidal!)
    - Paper uses **learned** positional embeddings, unlike the original Transformer
  - Combined: `embed = token_embed + pos_embed`
  - Apply **dropout** to the combined embedding
  - Verify: positions are just `[0, 1, 2, ..., seq_len-1]`

- [ ] **2.4.2 — Language model head**
  - **Weight tying:** The output projection (LM head) shares weights with the token embedding matrix
  - `logits = hidden_state @ token_embed.weight.T` (no bias)
  - This is a key detail from the paper — reduces parameter count

- [ ] **2.4.3 — Assemble the full model** *(file: `gpt1/model.py`)*
  - `GPT1(vocab_size, max_seq_len=512, n_layers=12, d_model=768, n_heads=12, d_ff=3072, dropout=0.1)`
  - Forward pass:
    ```
    1. Token embedding + Positional embedding + Dropout
    2. 12 × TransformerBlock
    3. Final LayerNorm (if using pre-norm; for post-norm this is already in each block)
    4. LM head (weight-tied with token embedding)
    ```
  - Outputs: `logits` of shape `(batch, seq_len, vocab_size)`

- [ ] **2.4.4 — Parameter count verification**
  - Total should be ~117M parameters
  - Break down:
    - Token embedding: `40,000 × 768 = ~30.7M`
    - Positional embedding: `512 × 768 = ~0.4M`
    - Per Transformer block: `~7.1M` (attention + FFN + LayerNorms)
    - 12 blocks: `~85.2M`
    - LM head: weight-tied, so 0 extra
    - **Total: ~116.3M** ✓
  - Write a `count_parameters()` utility function

- [ ] **2.4.5 — Weight initialization**
  - The paper references a modified initialization:
    - Weights initialized from `N(0, 0.02)`
    - Biases initialized to 0
    - Residual projections scaled by `1/√N` where N is the number of residual layers
  - Implement `init_weights()` method

---

## Phase 3 — Data Pipeline

### 3.1 — Pre-training Data (BooksCorpus)

- [ ] **3.1.1 — Understand the dataset**
  - BooksCorpus: ~7,000 unique unpublished books, ~800M words
  - Provides long contiguous text — important for learning long-range dependencies
  - **Note:** The original BooksCorpus may be difficult to obtain now. Alternatives:
    - Use HuggingFace's `bookcorpusopen` dataset
    - Use OpenWebText (used by GPT-2) as a substitute
    - Use a subset for initial experiments given the 12GB VRAM constraint

- [ ] **3.1.2 — Data loading & preprocessing** *(notebook: `05_pretraining.ipynb`)*
  - Download / load dataset
  - Clean text with `ftfy` (fixes Unicode issues, encoding problems)
  - Tokenize entire corpus into a flat array of token IDs
  - Create training examples: contiguous chunks of `seq_len = 512` tokens
  - No need for sentence boundaries — just continuous text
  - Implement efficient `Dataset` and `DataLoader`:
    - Memory-mapped arrays (e.g., `np.memmap`) for large datasets
    - Random shuffling of chunks
    - Proper batching (no padding needed for pre-training since all sequences are 512)

- [ ] **3.1.3 — Production data module** *(file: `gpt1/data.py`)*
  - `PretrainingDataset(data_path, seq_len=512)`
  - `get_pretrain_dataloader(dataset, batch_size, num_workers)`

### 3.2 — Fine-tuning Data

- [ ] **3.2.1 — Dataset preparation for each task**
  - Each downstream task requires different input formatting (this is a KEY contribution of the paper):
    - **Classification** (CoLA, SST-2): `[start] text [extract]` → linear head
    - **Entailment/NLI** (MNLI, SNLI, QNLI, RTE, SciTail): `[start] premise [delim] hypothesis [extract]` → linear head
    - **Similarity** (MRPC, QQP, STS-B): Two orderings `[start] s1 [delim] s2 [extract]` AND `[start] s2 [delim] s1 [extract]` → element-wise add → linear head
    - **Multiple Choice / QA** (RACE, Story Cloze): `[start] context [delim] answer_i [extract]` for each answer → independently through model → softmax over answers
  - Create a unified `FinetuneDataset` class that handles all task types

- [ ] **3.2.2 — Download all benchmarks**
  - Use HuggingFace `datasets` library:
    - `load_dataset("glue", "cola")` — CoLA
    - `load_dataset("glue", "sst2")` — SST-2
    - `load_dataset("glue", "mrpc")` — MRPC
    - `load_dataset("glue", "qqp")` — QQP
    - `load_dataset("glue", "stsb")` — STS-B
    - `load_dataset("glue", "mnli")` — MNLI
    - `load_dataset("glue", "qnli")` — QNLI
    - `load_dataset("glue", "rte")` — RTE
    - `load_dataset("snli")` — SNLI
    - `load_dataset("scitail")` — SciTail
    - `load_dataset("race", "all")` — RACE (Middle + High)
    - Story Cloze Test — requires separate download (ROCStories), may need manual signup

---

## Phase 4 — Pre-Training

> **Paper spec:** Adam optimizer, LR warmup (linear, 2000 steps) → max LR 2.5e-4 → cosine annealing to 0. Batch size 64, sequence length 512. ~100 epochs on BooksCorpus (approx 800k steps).

- [ ] **4.1 — Training loop** *(notebook: `05_pretraining.ipynb` → then `gpt1/train.py`)*
  - Implement the language modeling objective:
    - `loss = CrossEntropyLoss(logits[:, :-1, :], targets[:, 1:])`
    - Standard next-token prediction (causal LM)
  - Adam optimizer: `β1=0.9, β2=0.999, ε=1e-8`
  - Learning rate schedule:
    - Linear warmup from 0 → 2.5e-4 over first 2,000 steps
    - Cosine decay from 2.5e-4 → 0 over remaining steps
  - Gradient clipping: clip global norm to 1.0 (standard practice)
  - Weight decay: 0.01 (L2 regularization on all params except biases and LayerNorm)

- [ ] **4.2 — Memory optimization for RTX 3060 (12 GB)**
  - Calculate memory budget:
    - Model: ~117M params × 4 bytes (fp32) = ~468 MB
    - Optimizer (Adam): 2× model size = ~936 MB
    - Gradients: ~468 MB
    - Activations: depends on batch size and seq_len
    - **Total minimum: ~1.9 GB + activations**
  - Strategies to fit in 12 GB:
    - **Mixed precision (AMP):** Use `torch.cuda.amp` with fp16 — halves activation memory
    - **Gradient accumulation:** Effective batch size = 64, micro batch size = 8 or 16 → accumulate 4-8 steps
    - **Gradient checkpointing:** Trade compute for memory on activations
    - **Efficient attention:** For 512 seq_len this is manageable, but consider `torch.nn.functional.scaled_dot_product_attention` (Flash Attention)
  - Estimate realistic micro batch size through experiments

- [ ] **4.3 — Training infrastructure**
  - Checkpointing: Save model, optimizer, scheduler, step number every N steps
  - Resume from checkpoint
  - Logging to wandb every K steps:
    - Train loss, learning rate, gradient norm
    - Tokens/second throughput
    - GPU memory usage
    - Validation loss & perplexity (evaluate every N steps on held-out set)
  - Simple text generation for qualitative monitoring (generate a few samples every N steps)

- [ ] **4.4 — Scaled-down pre-training for development**
  - Before committing to full training, run a "sanity check" training:
    - Small dataset (1% of BooksCorpus or similar)
    - Verify loss decreases
    - Verify generated text improves
    - Check for training instabilities (loss spikes, NaN, etc.)
    - Profile memory and throughput
  - This is your iterative development loop — make it fast!

- [ ] **4.5 — Full pre-training run**
  - Train on the full dataset
  - Monitor via wandb dashboard
  - Expected training time on RTX 3060: **This will be very long** (days to weeks)
    - Consider using a subset or reduced model size for practical reasons
    - Alternative: Download OpenAI's original GPT-1 pretrained weights and fine-tune from there for benchmarking purposes
  - Save final pretrained checkpoint

---

## Phase 5 — Fine-Tuning Framework

> **Paper spec:** Fine-tune with auxiliary LM objective. LR = 6.25e-5, linear warmup over 0.2% of training, linear decay. Batch size = 32, epochs = 3. Dropout = 0.1. λ (auxiliary LM weight) = 0.5.

- [ ] **5.1 — Task-specific heads** *(notebook: `06_finetuning.ipynb` → then `gpt1/finetune.py`)*
  - For each task type, add a thin linear head on top of the Transformer's final hidden state:
    - **Classification:** Take the hidden state at the `[extract]` token position → Linear(768, n_classes)
    - **Similarity:** Process both orderings, element-wise add hidden states → Linear(768, n_classes)
    - **Multiple Choice:** Process each (context, answer) pair → Linear(768, 1) → softmax over choices
  - The linear head is randomly initialized; the Transformer backbone is initialized from pre-training

- [ ] **5.2 — Combined objective**
  - **L₃ = L₂ + λ · L₁**
    - L₂ = Task-specific supervised loss (cross-entropy for classification, MSE for STS-B regression)
    - L₁ = Language modeling loss (next-token prediction on the fine-tuning text)
    - λ = 0.5
  - The LM auxiliary objective acts as a regularizer — prevents catastrophic forgetting of pre-trained representations

- [ ] **5.3 — Fine-tuning hyperparameters**
  - Adam optimizer: `lr = 6.25e-5`
  - Warmup: Linear over 0.2% of total training steps
  - LR schedule: Linear decay to 0
  - Batch size: 32
  - Epochs: 3 (for most tasks)
  - Dropout: 0.1 (applied to attention, FFN, embeddings, classifier head)
  - No weight decay mentioned for fine-tuning (check paper carefully)

- [ ] **5.4 — Fine-tuning pipeline**
  - Implement a general-purpose fine-tuning loop that:
    1. Loads pretrained checkpoint
    2. Adds task-specific head
    3. Prepares task-specific data (with proper input formatting from §3.2)
    4. Trains with combined objective L₃
    5. Evaluates on validation set each epoch
    6. Saves best checkpoint (by validation metric)
    7. Logs everything to wandb
  - Make it easy to switch between tasks via config

---

## Phase 6 — Evaluation Benchmarks

> **The paper evaluates on 12 tasks across 4 categories. Reproduce ALL of them.**

### 6.1 — Natural Language Inference (5 tasks)

- [ ] **6.1.1 — SNLI** (Stanford NLI)
  - 3-way classification: entailment / contradiction / neutral
  - ~570K training pairs
  - Metric: **Accuracy**
  - Paper result: 89.9%

- [ ] **6.1.2 — MNLI** (Multi-Genre NLI)
  - Same 3-way classification, but multiple genres
  - ~393K training pairs
  - Metric: **Accuracy** (matched & mismatched dev sets)
  - Paper result: 82.1% (matched)

- [ ] **6.1.3 — QNLI** (Question NLI, from SQuAD)
  - Binary: entailment / not-entailment
  - Metric: **Accuracy**
  - Paper result: 88.1%

- [ ] **6.1.4 — RTE** (Recognizing Textual Entailment)
  - Binary entailment
  - Small dataset (~2.5K training examples)
  - Metric: **Accuracy**
  - Paper result: 56.0%

- [ ] **6.1.5 — SciTail** (Science Entailment)
  - Binary: entails / neutral
  - Metric: **Accuracy**
  - Paper result: 88.3%

### 6.2 — Question Answering & Commonsense Reasoning (2 tasks)

- [ ] **6.2.1 — RACE** (Reading Comprehension from Exams)
  - Multiple choice QA (4 options per question)
  - Middle school + High school subsets
  - Metric: **Accuracy** (overall, middle, high)
  - Paper result: 59.0% (overall)

- [ ] **6.2.2 — Story Cloze Test**
  - Choose correct story ending from 2 options
  - Metric: **Accuracy**
  - Paper result: 86.5%
  - **Note:** This dataset requires signing up at the ROCStories website

### 6.3 — Semantic Similarity (3 tasks)

- [ ] **6.3.1 — MRPC** (Microsoft Research Paraphrase Corpus)
  - Binary: paraphrase / not paraphrase
  - Metric: **F1 score**
  - Paper result: 82.3%

- [ ] **6.3.2 — QQP** (Quora Question Pairs)
  - Binary: duplicate / not duplicate
  - Metric: **F1 score**
  - Paper result: 70.3%

- [ ] **6.3.3 — STS-B** (Semantic Textual Similarity Benchmark)
  - **Regression task** — predict similarity score from 1 to 5
  - Metric: **Pearson correlation**
  - Paper result: 82.0%

### 6.4 — Text Classification (2 tasks)

- [ ] **6.4.1 — CoLA** (Corpus of Linguistic Acceptability)
  - Binary: grammatically acceptable / not
  - Metric: **Matthews correlation coefficient (MCC)**
  - Paper result: 45.4%

- [ ] **6.4.2 — SST-2** (Stanford Sentiment Treebank)
  - Binary: positive / negative sentiment
  - Metric: **Accuracy**
  - Paper result: 91.3%

### 6.5 — Evaluation Infrastructure

- [ ] **6.5.1 — Unified evaluation harness** *(file: `gpt1/evaluate.py`, `evals/`)*
  - For each task:
    - Load fine-tuned checkpoint
    - Run inference on test/dev set
    - Compute the correct metric (accuracy / F1 / MCC / Pearson)
    - Log results to wandb as a summary table
  - Create a comparison table: **Your results vs Paper results**
  - Automate: single script to evaluate all 12 tasks

---

## Phase 7 — Results Analysis & Ablations

- [ ] **7.1 — Results comparison table**
  - Create a markdown/wandb table comparing your results to the paper's reported numbers for all 12 tasks
  - Analyze where your reproduction differs and why (dataset differences, tokenizer, training duration, etc.)

- [ ] **7.2 — Ablation studies** (from the paper)
  - The paper performs several ablations — reproduce them:
    - **No pre-training:** Train only on fine-tuning data (no transfer learning)
    - **No auxiliary LM objective (λ=0):** Fine-tune without the language modeling loss
    - **LSTM instead of Transformer:** Compare architecture (optional, for learning)
    - **Effect of number of pre-training layers transferred:** Transfer 1, 3, 6, 9, 12 layers and compare
  - Log all ablations to wandb for comparison

- [ ] **7.3 — Zero-shot evaluation** (from the paper)
  - The paper shows that pre-training alone (without fine-tuning) gives non-trivial performance on some tasks
  - Evaluate the pre-trained model (before fine-tuning) on all tasks
  - Compare zero-shot vs fine-tuned performance

---

## Phase 8 — Modern Improvements (Post-Reproduction)

> **Goal:** After faithful reproduction, introduce modern techniques to push performance toward current SOTA.

- [ ] **8.1 — Architecture improvements**
  - [ ] **Pre-Norm instead of Post-Norm** — GPT-2 and later models use pre-norm (LayerNorm before attention/FFN), which improves training stability
  - [ ] **RoPE (Rotary Positional Embeddings)** — Replace learned positional embeddings with RoPE for better length generalization
  - [ ] **SwiGLU activation** — Replace GELU FFN with SwiGLU (used in LLaMA), which tends to improve quality
  - [ ] **RMSNorm** — Replace LayerNorm with RMSNorm (faster, used in LLaMA)
  - [ ] **Grouped Query Attention (GQA)** — More memory-efficient attention (may not matter at this scale)
  - [ ] **KV-cache for inference** — Fast autoregressive generation

- [ ] **8.2 — Training improvements**
  - [ ] **Larger, better data** — Train on OpenWebText, The Pile, or RedPajama instead of BooksCorpus alone
  - [ ] **Better tokenizer** — Use GPT-2's BPE (50,257 vocab) or SentencePiece
  - [ ] **AdamW optimizer** — Use decoupled weight decay (AdamW) instead of Adam + L2
  - [ ] **Cosine schedule with restarts** or **warm-stable-decay** schedule
  - [ ] **Longer context** — Increase from 512 to 1024 or 2048 tokens
  - [ ] **Gradient accumulation tuning** — Larger effective batch sizes
  - [ ] **Mixed precision (bf16)** — If supported by RTX 3060 (it isn't natively; stick with fp16 + loss scaling)
  - [ ] **Flash Attention** — Use `torch.nn.functional.scaled_dot_product_attention` for memory-efficient attention
  - [ ] **Compile** — `torch.compile()` for faster training

- [ ] **8.3 — Fine-tuning improvements**
  - [ ] **LoRA / Parameter-efficient fine-tuning** — Fine-tune only a small number of adapter parameters
  - [ ] **Multi-task learning** — Fine-tune on multiple tasks simultaneously
  - [ ] **Better data augmentation** — Back-translation, synonym substitution, etc.
  - [ ] **Longer fine-tuning with early stopping** — Instead of fixed 3 epochs
  - [ ] **Hyperparameter sweeps** — Use wandb sweeps for automated HPO

- [ ] **8.4 — Evaluation improvements**
  - [ ] **Ensemble** — Average predictions from multiple fine-tuning runs
  - [ ] **Test-time augmentation** — Average predictions over different input orderings (for similarity tasks)
  - [ ] Compare against current SOTA on all benchmarks

---

## Phase 9 — Documentation & Reproducibility

- [ ] **9.1 — README** with:
  - Project overview
  - How to set up environment with `uv`
  - How to reproduce pre-training
  - How to fine-tune on each task
  - How to run evaluations
  - Results table
  - Wandb project link

- [ ] **9.2 — Notebooks as documentation**
  - Each notebook should tell a story: what you're building, why, and how
  - Include inline commentary, visualizations, and experimental results
  - Think of it as a blog post / tutorial

- [ ] **9.3 — Wandb reports**
  - Create a wandb report summarizing all experiments
  - Include training curves, ablation comparisons, and final benchmark results

---

## Suggested Order of Attack

> **This is the recommended order to implement things, from the ground up:**

| Step | What to Build | Where | Milestone |
|------|--------------|-------|-----------|
| 1 | Project setup, `uv`, `wandb` | Root | Can run `wandb.init()` ✓ |
| 2 | BPE tokenizer | Notebook → `tokenizer.py` | Can encode/decode text ✓ |
| 3 | Multi-head masked self-attention | Notebook → `model.py` | Correct shapes, causal mask works ✓ |
| 4 | FFN + GELU | Notebook → `model.py` | Forward pass works ✓ |
| 5 | Transformer block | Notebook → `model.py` | Single block forward pass ✓ |
| 6 | Full GPT-1 model | Notebook → `model.py` | ~117M params, generates random tokens ✓ |
| 7 | Data pipeline (pre-training) | Notebook → `data.py` | Batches of 512-token sequences ✓ |
| 8 | Pre-training loop | Notebook → `train.py` | Loss decreases, wandb logs ✓ |
| 9 | Sanity check training | Notebook | Overfits small dataset, generates text ✓ |
| 10 | Full pre-training | `scripts/pretrain.sh` | Converged model, good perplexity ✓ |
| 11 | Fine-tuning framework | Notebook → `finetune.py` | Can fine-tune on any GLUE task ✓ |
| 12 | All 12 eval benchmarks | `evals/` | Results for all tasks ✓ |
| 13 | Results comparison | README + wandb | Table: yours vs paper ✓ |
| 14 | Ablation studies | wandb | All paper ablations reproduced ✓ |
| 15 | Modern improvements | Iterate on model | Beat paper results ✓ |

---

## Key Reference Numbers (from the Paper)

| Parameter | Value |
|-----------|-------|
| Transformer layers | 12 |
| Hidden dimension (d_model) | 768 |
| Attention heads | 12 |
| FFN inner dimension | 3,072 |
| Max sequence length | 512 |
| Vocabulary size | ~40,000 (BPE) |
| Total parameters | ~117M |
| Activation function | GELU |
| Positional embeddings | Learned |
| Pre-training data | BooksCorpus (~800M words) |
| Pre-training optimizer | Adam (β₁=0.9, β₂=0.999) |
| Pre-training LR | 2.5e-4 (peak), cosine decay |
| Pre-training warmup | 2,000 steps (linear) |
| Pre-training batch size | 64 |
| Fine-tuning LR | 6.25e-5 |
| Fine-tuning warmup | 0.2% of steps (linear) |
| Fine-tuning batch size | 32 |
| Fine-tuning epochs | 3 |
| Auxiliary LM weight (λ) | 0.5 |
| Dropout | 0.1 |
| Weight init | N(0, 0.02) |

---

*Last updated: 2026-02-27*
