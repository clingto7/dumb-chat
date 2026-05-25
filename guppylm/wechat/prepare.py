"""Tokenizer training for WeChat persona model.

Trains a ByteLevel BPE tokenizer on the augmented ChatML data with a larger
vocabulary size suitable for Chinese text.
"""

import json
import os

from .config import WechatTrainConfig

SPECIAL_TOKENS = [
    "<pad>",          # 0
    "<|im_start|>",   # 1
    "<|im_end|>",     # 2
]


def train_wechat_tokenizer(
    data_dir: str = "wechat_data/augmented",
    tokenizer_path: str = "wechat_data/tokenizer.json",
    vocab_size: int = 8192,
    min_frequency: int = 2,
) -> None:
    """Train a BPE tokenizer on the augmented ChatML data.

    Uses the same ByteLevel BPE approach as guppylm/prepare_data.py but
    with a larger vocab_size for Chinese text coverage.

    The ByteLevel pre-tokenizer handles CJK characters naturally by
    decomposing to UTF-8 bytes, so no special CJK pre-processing needed.
    BPE merges then learn common Chinese character/word patterns.

    Args:
        data_dir: Directory containing train.jsonl and eval.jsonl.
        tokenizer_path: Output path for the trained tokenizer.
        vocab_size: Vocabulary size (8192 for Chinese chat text).
        min_frequency: Minimum token frequency for BPE training.
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

    # Read all text from training and eval data
    texts = []
    for name in ["train.jsonl", "eval.jsonl"]:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    texts.append(data["text"])

    if not texts:
        print(f"No training data found in {data_dir}")
        return

    print(f"Training BPE tokenizer (vocab_size={vocab_size}) on {len(texts)} texts...")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        min_frequency=min_frequency,
    )

    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    os.makedirs(os.path.dirname(tokenizer_path) or ".", exist_ok=True)
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path} ({tokenizer.get_vocab_size()} tokens)")


def prepare(config: WechatTrainConfig | None = None) -> None:
    """Full WeChat data preparation pipeline: train tokenizer on augmented data.

    Steps:
    1. Read augmented train/eval JSONL
    2. Train BPE tokenizer on all text
    3. Quick tokenizer test with Chinese sample
    4. Save tokenizer to config.tokenizer_path
    """
    if config is None:
        config = WechatTrainConfig()

    # Train tokenizer
    train_wechat_tokenizer(
        data_dir=config.data_dir,
        tokenizer_path=config.tokenizer_path,
        vocab_size=config.vocab_size,
    )

    # Quick test
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(config.tokenizer_path)

    test_cases = [
        "<|im_start|>user\n你好<|im_end|>",
        "<|im_start|>assistant\n你好呀，今天怎么样？<|im_end|>",
        "<|im_start|>user\n你吃饭了吗<|im_end|>",
    ]

    print("\nTokenizer test:")
    for test in test_cases:
        ids = tokenizer.encode(test).ids
        decoded = tokenizer.decode(ids)
        print(f"  Input:   {test}")
        print(f"  Tokens:  {len(ids)} ids")
        print(f"  Decoded: {decoded}")
        print()

    # Stats
    vocab = tokenizer.get_vocab()
    # Count Chinese-ish tokens (UTF-8 byte sequences for CJK)
    cjk_tokens = sum(1 for t in vocab if any("\u4e00" <= c <= "\u9fff" for c in t))
    print(f"Vocabulary stats:")
    print(f"  Total tokens: {len(vocab)}")
    print(f"  CJK-related tokens: {cjk_tokens}")
