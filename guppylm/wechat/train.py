"""WeChat persona model training.

Reuses the core training loop from guppylm/train.py with WeChat-specific
configuration: larger vocab for Chinese, longer sequence length, etc.
"""

from ..config import GuppyConfig, TrainConfig
from ..train import train_with_configs
from .config import WechatTrainConfig


def train(config: WechatTrainConfig | None = None) -> None:
    """Train the WeChat persona model.

    Builds GuppyConfig and TrainConfig from WechatTrainConfig and delegates
    to the shared train_with_configs() function.

    Key differences from fish-chat training:
    - Larger vocab (8192) for Chinese text coverage
    - Longer max_seq_len (256) for Chinese conversations
    - Data from wechat_data/ instead of data/
    - Checkpoints in wechat_checkpoints/ instead of checkpoints/
    - Model architecture is identical (GuppyLM with different config)
    """
    if config is None:
        config = WechatTrainConfig()

    mc = GuppyConfig(
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        ffn_hidden=config.ffn_hidden,
        dropout=config.dropout,
    )

    tc = TrainConfig(
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        min_lr=config.min_lr,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        eval_interval=config.eval_interval,
        save_interval=config.save_interval,
        grad_clip=config.grad_clip,
        device=config.device,
        seed=config.seed,
        data_dir=config.data_dir,
        output_dir=config.output_dir,
    )

    print("WeChat Persona Training")
    print(f"  vocab_size={mc.vocab_size}, max_seq_len={mc.max_seq_len}")
    print(f"  data_dir={tc.data_dir}, output_dir={tc.output_dir}")
    print()

    train_with_configs(mc, tc)
