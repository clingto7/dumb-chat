"""WeChat persona configuration."""

from dataclasses import dataclass, field
from pathlib import Path

WECHAT_DATA_DIR = Path.home() / "Documents" / "xwechat_files"


@dataclass
class WechatDecryptConfig:
    """Configuration for WeChat DB decryption."""

    wechat_data_dir: str = str(WECHAT_DATA_DIR)
    # Auto-detect wxid subdirectory, or specify explicitly
    wxid: str = ""  # e.g. "wxid_0nvwqeeclz9u22_f63a"
    # Output directory for decrypted databases
    decrypted_dir: str = "wechat_data/decrypted"
    # Keys file path (from wechat-decrypt tool)
    keys_file: str = "wechat_data/all_keys.json"


@dataclass
class WechatExtractConfig:
    """Configuration for chat record extraction."""

    decrypted_dir: str = "wechat_data/decrypted"
    output_dir: str = "wechat_data/extracted"
    # Target contact/group wxid or alias (empty = list all)
    target_contacts: list[str] = field(default_factory=list)
    # Message types to include (1=text, 3=image, 34=voice, 43=video, 47=emoji, 10000=system)
    include_types: list[int] = field(default_factory=lambda: [1])
    # Max messages per contact (0 = unlimited)
    max_messages: int = 0
    # Time range filter (ISO format strings, empty = no filter)
    start_time: str = ""
    end_time: str = ""


@dataclass
class WechatChatMLConfig:
    """Configuration for ChatML sample generation from extracted chats."""

    input_dir: str = "wechat_data/extracted"
    output_dir: str = "wechat_data/chatml"
    # Target person's wxid (their messages become "assistant")
    target_wxid: str = ""
    # Max turns per sample (1 = single turn)
    max_turns: int = 1
    # Maximum context messages before target's response
    max_context_messages: int = 5
    # Minimum message length (chars) to include
    min_msg_length: int = 1
    # Maximum message length (chars) -- truncate longer messages
    max_msg_length: int = 200
    # Overlap stride for sliding window on long conversations (0 = no overlap)
    stride: int = 0
    # Eval split ratio
    eval_ratio: float = 0.05


@dataclass
class WechatAugmentConfig:
    """Configuration for data augmentation."""

    input_dir: str = "wechat_data/chatml"
    output_dir: str = "wechat_data/augmented"
    # LLM-based augmentation
    llm_enabled: bool = False
    llm_api_base: str = ""  # e.g. "https://api.openai.com/v1"
    llm_api_key: str = ""  # or use LLM_API_KEY env var
    llm_model: str = "gpt-4o-mini"
    llm_augment_ratio: float = 1.0  # generate N augmentation per original sample
    # Template-based augmentation
    template_enabled: bool = True
    synonym_replace_prob: float = 0.1
    random_delete_prob: float = 0.05
    sentence_reorder: bool = True


@dataclass
class WechatTrainConfig:
    """Training config for WeChat persona model -- overrides TrainConfig defaults."""

    data_dir: str = "wechat_data/augmented"
    output_dir: str = "wechat_checkpoints"
    tokenizer_path: str = "wechat_data/tokenizer.json"
    # Model config overrides (Chinese text needs larger vocab)
    vocab_size: int = 8192
    max_seq_len: int = 256  # longer for Chinese conversations
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    ffn_hidden: int = 768
    dropout: float = 0.1
    # Training overrides
    batch_size: int = 16
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 20000
    eval_interval: int = 200
    save_interval: int = 500
    grad_clip: float = 1.0
    device: str = "auto"
    seed: int = 42
