"""Tests for the config system (section 0.4).

Covers:
    - Default construction
    - YAML loading
    - Keyword overrides
    - CLI overrides
    - Override priority  (CLI > kwargs > YAML > defaults)
    - Unknown-field validation
    - Type casting
    - Computed properties (grad_accum_steps)
"""

import pytest
import textwrap
from pathlib import Path

from gpt.utils import ExperimentConfig, load_config


# ── Helpers ────────────────────────────────────────────────────────────
@pytest.fixture
def tmp_yaml(tmp_path):
    """Factory fixture — write a YAML string to a temp file and return its Path."""
    def _write(content: str) -> Path:
        p = tmp_path / "test_config.yaml"
        p.write_text(textwrap.dedent(content))
        return p
    return _write


# ── Defaults ───────────────────────────────────────────────────────────
class TestDefaults:
    def test_paper_model_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.n_layers == 12
        assert cfg.n_heads == 12
        assert cfg.d_model == 768
        assert cfg.d_ff == 3072
        assert cfg.vocab_size == 40_000
        assert cfg.max_seq_len == 512
        assert cfg.dropout == 0.1

    def test_paper_pretrain_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.batch_size == 64
        assert cfg.lr == 2.5e-4
        assert cfg.warmup_steps == 2_000
        assert cfg.total_steps == 800_000
        assert cfg.weight_decay == 0.01
        assert cfg.grad_clip == 1.0
        assert cfg.adam_beta1 == 0.9
        assert cfg.adam_beta2 == 0.999

    def test_paper_finetune_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.ft_lr == 6.25e-5
        assert cfg.ft_epochs == 3
        assert cfg.ft_batch_size == 32
        assert cfg.ft_warmup_frac == 0.002
        assert cfg.lambda_aux == 0.5
        assert cfg.label_smoothing == 0.0

    def test_infrastructure_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.wandb_project == "GPT1"
        assert cfg.checkpoint_dir == "checkpoints/"
        assert cfg.data_dir == "data/"
        assert cfg.resume_from == ""

    def test_logging_interval_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.log_interval == 50
        assert cfg.eval_interval == 1_000
        assert cfg.save_interval == 5_000


# ── Computed properties ────────────────────────────────────────────────
class TestComputed:
    def test_grad_accum_steps(self):
        cfg = ExperimentConfig(batch_size=64, micro_batch_size=8)
        assert cfg.grad_accum_steps == 8

    def test_grad_accum_steps_mismatch(self):
        cfg = ExperimentConfig(batch_size=64, micro_batch_size=7)
        with pytest.raises(AssertionError, match="divisible"):
            _ = cfg.grad_accum_steps


# ── YAML loading ───────────────────────────────────────────────────────
class TestYAML:
    def test_load_from_yaml(self, tmp_yaml):
        path = tmp_yaml("""
            lr: 1.0e-4
            batch_size: 32
            task_name: "mnli"
        """)
        cfg = load_config(path)
        assert cfg.lr == 1e-4
        assert cfg.batch_size == 32
        assert cfg.task_name == "mnli"
        # defaults still apply for unspecified fields
        assert cfg.n_layers == 12
        assert cfg.dropout == 0.1

    def test_missing_yaml_file(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config("nonexistent.yaml")

    def test_unknown_yaml_field(self, tmp_yaml):
        path = tmp_yaml("""
            lr: 1e-4
            bogus_field: 42
        """)
        with pytest.raises(ValueError, match="Unknown config fields"):
            load_config(path)

    def test_empty_yaml(self, tmp_yaml):
        path = tmp_yaml("")
        cfg = load_config(path)
        # should just return defaults
        assert cfg == ExperimentConfig()


# ── Keyword overrides ──────────────────────────────────────────────────
class TestKeywordOverrides:
    def test_override_defaults(self):
        cfg = load_config(lr=1e-3, n_layers=6)
        assert cfg.lr == 1e-3
        assert cfg.n_layers == 6

    def test_override_yaml(self, tmp_yaml):
        path = tmp_yaml("lr: 1.0e-4")
        cfg = load_config(path, lr=5e-5)
        # kwarg wins over YAML
        assert cfg.lr == 5e-5

    def test_unknown_kwarg(self):
        with pytest.raises(ValueError, match="Unknown config overrides"):
            load_config(bogus=123)


# ── CLI overrides ──────────────────────────────────────────────────────
class TestCLI:
    def test_cli_overrides(self):
        cfg = load_config(cli_args=["--lr", "1e-3", "--n_layers", "6"])
        assert cfg.lr == 1e-3
        assert cfg.n_layers == 6

    def test_cli_beats_yaml(self, tmp_yaml):
        path = tmp_yaml("lr: 1.0e-4")
        cfg = load_config(path, cli_args=["--lr", "9e-5"])
        assert cfg.lr == 9e-5

    def test_cli_beats_kwargs(self):
        cfg = load_config(lr=1e-4, cli_args=["--lr", "9e-5"])
        assert cfg.lr == 9e-5

    def test_unknown_cli_flag(self):
        with pytest.raises(ValueError, match="Unknown CLI flag"):
            load_config(cli_args=["--bogus", "42"])

    def test_cli_bool(self):
        cfg = load_config(cli_args=["--use_amp", "false"])
        assert cfg.use_amp is False

    def test_cli_missing_value(self):
        with pytest.raises(ValueError, match="requires a value"):
            load_config(cli_args=["--lr"])


# ── Type casting ───────────────────────────────────────────────────────
class TestTypeCasting:
    def test_yaml_int_to_float(self, tmp_yaml):
        """YAML parses '1' as int, but lr expects float."""
        path = tmp_yaml("lr: 1")
        cfg = load_config(path)
        assert isinstance(cfg.lr, float)
        assert cfg.lr == 1.0

    def test_cli_str_to_int(self):
        """CLI always gives strings; they must be cast."""
        cfg = load_config(cli_args=["--n_layers", "24"])
        assert isinstance(cfg.n_layers, int)
        assert cfg.n_layers == 24

    def test_cli_bool_true_variants(self):
        for val in ["true", "True", "1", "yes"]:
            cfg = load_config(cli_args=["--use_amp", val])
            assert cfg.use_amp is True

    def test_cli_bool_false_variants(self):
        for val in ["false", "False", "0", "no"]:
            cfg = load_config(cli_args=["--use_amp", val])
            assert cfg.use_amp is False


# ── Priority integration test ─────────────────────────────────────────
class TestPriority:
    def test_full_override_chain(self, tmp_yaml):
        """CLI > kwargs > YAML > defaults."""
        path = tmp_yaml("""
            lr: 1.0e-4
            n_layers: 6
            batch_size: 16
        """)
        cfg = load_config(
            path,
            cli_args=["--lr", "9e-5"],
            n_layers=24,
        )
        # lr: CLI wins (9e-5)
        assert cfg.lr == 9e-5
        # n_layers: kwarg wins (24), YAML said 6
        # BUT wait — CLI is parsed AFTER kwargs, and CLI didn't set n_layers,
        # so kwarg (24) should win over YAML (6). However, the current
        # implementation applies kwargs before CLI, so kwarg wins here.
        assert cfg.n_layers == 24
        # batch_size: YAML wins (16), no kwarg or CLI override
        assert cfg.batch_size == 16
        # d_model: defaults (768)
        assert cfg.d_model == 768


# ── Realistic config files ─────────────────────────────────────────────
class TestRealisticConfigs:
    def test_load_pretrain_yaml(self):
        """Load the actual pretrain.yaml we ship."""
        path = Path("configs/pretrain.yaml")
        if not path.exists():
            pytest.skip("pretrain.yaml not found (running from different cwd)")
        cfg = load_config(path)
        assert cfg.lr == 2.5e-4
        assert cfg.batch_size == 64
        assert cfg.lr_schedule == "cosine"
        assert cfg.use_amp is True

    def test_load_finetune_yaml(self):
        """Load the actual finetune.yaml we ship."""
        path = Path("configs/finetune.yaml")
        if not path.exists():
            pytest.skip("finetune.yaml not found (running from different cwd)")
        cfg = load_config(path)
        assert cfg.ft_lr == 6.25e-5
        assert cfg.task_name == "sst2"
        assert cfg.lambda_aux == 0.5
        assert cfg.ft_lr_schedule == "linear"
