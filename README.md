# GPT-1: Implementing and Surpassing the Original

Reproducing the GPT-1 model from scratch, following the original paper — [*Improving Language Understanding by Generative Pre-Training*](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (Radford et al., 2018).

The goal is to faithfully implement the architecture and training pipeline described in the paper, reproduce its benchmark results, and then explore modern improvements to surpass the original evaluations.

## Setup

Requires **Python 3.12+**. Uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
# Clone the repo
git clone https://github.com/0xAnv/gpt1.git
cd gpt1

# Install dependencies
uv sync
```

## Project Structure

```
gpt1/
├── main.py              # Entry point
├── implementation.md    # Detailed implementation checklist
├── GPT1.pdf             # Original paper
├── pyproject.toml       # Project config & dependencies
└── README.md
```

## References

- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — Radford et al., 2018
