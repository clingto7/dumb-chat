"""Convert extracted WeChat messages into ChatML training samples.

Reads JSONL files produced by guppylm/wechat/extract.py and converts them
to ChatML format compatible with the existing GuppyLM training pipeline.

ChatML format:
    <|im_start|>user
    [Other person's message]<|im_end|>
    <|im_start|>assistant
    [Target person's response]<|im_end|>
"""

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import WechatChatMLConfig
from .extract import WechatMessage


@dataclass
class ChatMLSample:
    text: str           # Full ChatML formatted text
    category: str       # "wechat_persona"
    n_turns: int        # Number of conversation turns
    source_contact: str  # Original contact wxid


def _format_chatml_turn(role: str, content: str) -> str:
    """Format a single ChatML turn."""
    return f"<|im_start|>{role}\n{content}<|im_end|>"


def _format_single_turn(context_messages: list[str], target_message: str) -> str:
    """Format a single-turn ChatML sample with context."""
    parts = []
    if context_messages:
        user_content = "\n".join(context_messages)
        parts.append(_format_chatml_turn("user", user_content))
    parts.append(_format_chatml_turn("assistant", target_message))
    return "\n".join(parts)


def _format_multi_turn(turns: list[tuple[str, str]]) -> str:
    """Format a multi-turn ChatML sample."""
    parts = [_format_chatml_turn(role, content) for role, content in turns]
    return "\n".join(parts)


def _load_messages(jsonl_path: str) -> list[WechatMessage]:
    """Load messages from an extracted JSONL file."""
    messages = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            messages.append(WechatMessage(**data))
    return messages


def _is_valid_message(msg: WechatMessage, min_len: int, max_len: int) -> bool:
    """Check if a message is suitable for training."""
    if msg.type != 1:  # Only text messages
        return False
    content = msg.content.strip()
    if len(content) < min_len or len(content) > max_len:
        return False
    return True


def _is_target_message(msg: WechatMessage, target_wxid: str) -> bool:
    """Determine if a message was sent by the target person.

    For 1-on-1 chats where target_wxid is the other person:
    - is_self=False and sender=target_wxid means the target sent it
    - In 1-on-1 chats, the talker is the other person's wxid
    - When sender is empty (1-on-1), use is_self to determine:
      is_self=True means account owner sent, is_self=False means other person sent

    For group chats:
    - sender field has the actual sender wxid
    """
    if msg.sender == target_wxid:
        return True
    # 1-on-1 chat: sender may be empty, is_self distinguishes direction
    if not msg.sender and not msg.is_self and msg.talker == target_wxid:
        return True
    return False


def messages_to_single_turn(
    messages: list[WechatMessage],
    target_wxid: str,
    max_context_messages: int = 5,
    min_msg_length: int = 1,
    max_msg_length: int = 200,
) -> list[ChatMLSample]:
    """Convert a message list to single-turn ChatML samples.

    For each target person message:
    - Collect up to max_context_messages preceding messages as "user" input
    - Target person's message becomes "assistant" output
    """
    samples = []
    valid_msgs = [m for m in messages if _is_valid_message(m, min_msg_length, max_msg_length)]

    for i, msg in enumerate(valid_msgs):
        if not _is_target_message(msg, target_wxid):
            continue

        # Collect preceding context (non-target messages)
        context = []
        for j in range(i - 1, -1, -1):
            prev = valid_msgs[j]
            if _is_target_message(prev, target_wxid):
                break  # Stop at previous target message
            context.insert(0, prev.content.strip()[:max_msg_length])
            if len(context) >= max_context_messages:
                break

        if not context:
            continue

        target_content = msg.content.strip()[:max_msg_length]
        chatml_text = _format_single_turn(context, target_content)

        samples.append(ChatMLSample(
            text=chatml_text,
            category="wechat_persona",
            n_turns=1,
            source_contact=msg.talker,
        ))

    return samples


def messages_to_multi_turn(
    messages: list[WechatMessage],
    target_wxid: str,
    max_turns: int = 3,
    max_context_messages: int = 10,
    min_msg_length: int = 1,
    max_msg_length: int = 200,
    stride: int = 0,
    gap_seconds: int = 1800,
) -> list[ChatMLSample]:
    """Convert a message list to multi-turn ChatML samples.

    Creates sliding windows of conversation turns.
    Conversations are split at gaps > gap_seconds (30 min default).
    Consecutive messages from the same role are merged.
    """
    samples = []
    valid_msgs = [m for m in messages if _is_valid_message(m, min_msg_length, max_msg_length)]

    if not valid_msgs:
        return samples

    # Build conversation turns, splitting at time gaps
    segments = []
    current_segment = []

    for msg in valid_msgs:
        if current_segment:
            time_gap = abs(msg.timestamp - current_segment[-1].timestamp)
            # Handle both seconds and ms timestamps
            if msg.timestamp > 1e12:
                gap_threshold = gap_seconds * 1000
            else:
                gap_threshold = gap_seconds
            if time_gap > gap_threshold:
                if len(current_segment) >= 2:
                    segments.append(current_segment)
                current_segment = []
        current_segment.append(msg)

    if len(current_segment) >= 2:
        segments.append(current_segment)

    # Convert each segment into turns
    for segment in segments:
        turns = []
        current_role = None
        current_contents = []

        for msg in segment:
            role = "assistant" if _is_target_message(msg, target_wxid) else "user"
            content = msg.content.strip()[:max_msg_length]

            if role == current_role and current_contents:
                current_contents.append(content)
            else:
                if current_contents:
                    turns.append((current_role, "\n".join(current_contents)))
                current_role = role
                current_contents = [content]

        if current_contents:
            turns.append((current_role, "\n".join(current_contents)))

        if len(turns) < 2:
            continue

        # Create sliding windows
        effective_stride = stride if stride > 0 else max_turns
        for start in range(0, len(turns) - 1, effective_stride):
            window = turns[start:start + max_turns * 2]
            if len(window) < 2:
                continue
            # Ensure starts with user, ends with assistant
            if window[0][0] != "user":
                window = window[1:]
            if not window:
                continue
            if window[-1][0] != "assistant":
                window = window[:-1]
            if len(window) < 2:
                continue

            chatml_text = _format_multi_turn(window)
            samples.append(ChatMLSample(
                text=chatml_text,
                category="wechat_persona",
                n_turns=sum(1 for role, _ in window if role == "assistant"),
                source_contact=segment[0].talker,
            ))

    return samples


def convert_to_chatml(config: WechatChatMLConfig | None = None) -> None:
    """Convert extracted messages to ChatML training samples.

    Reads JSONL files from config.input_dir, converts to ChatML format,
    splits into train/eval, and writes to config.output_dir.
    """
    if config is None:
        config = WechatChatMLConfig()

    input_dir = config.input_dir
    output_dir = config.output_dir
    target_wxid = config.target_wxid

    if not target_wxid:
        print("Error: target_wxid must be specified in WechatChatMLConfig")
        print("Set target_wxid to the WeChat ID of the person whose style you want to imitate.")
        return

    all_samples = []
    jsonl_files = sorted(Path(input_dir).glob("*.jsonl")) if os.path.isdir(input_dir) else []

    if not jsonl_files:
        print(f"No .jsonl files found in {input_dir}")
        print("Run 'python -m guppylm wechat-extract' first.")
        return

    for jsonl_path in jsonl_files:
        print(f"Processing {jsonl_path.name}...")
        messages = _load_messages(str(jsonl_path))

        if config.max_turns <= 1:
            samples = messages_to_single_turn(
                messages, target_wxid,
                max_context_messages=config.max_context_messages,
                min_msg_length=config.min_msg_length,
                max_msg_length=config.max_msg_length,
            )
        else:
            samples = messages_to_multi_turn(
                messages, target_wxid,
                max_turns=config.max_turns,
                max_context_messages=config.max_context_messages,
                min_msg_length=config.min_msg_length,
                max_msg_length=config.max_msg_length,
                stride=config.stride,
            )

        all_samples.extend(samples)
        print(f"  {len(samples)} samples from {len(messages)} messages")

    if not all_samples:
        print("No training samples generated.")
        print("Check that target_wxid matches a sender in the extracted messages.")
        return

    # Shuffle and split
    random.seed(42)
    random.shuffle(all_samples)
    n_eval = int(len(all_samples) * config.eval_ratio)
    eval_samples, train_samples = all_samples[:n_eval], all_samples[n_eval:]

    # Write output
    os.makedirs(output_dir, exist_ok=True)
    for name, data in [
        (os.path.join(output_dir, "train.jsonl"), train_samples),
        (os.path.join(output_dir, "eval.jsonl"), eval_samples),
    ]:
        with open(name, "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps({"text": s.text, "category": s.category}, ensure_ascii=False) + "\n")

    print(f"\nGenerated {len(all_samples)} ChatML samples:")
    print(f"  Train: {len(train_samples)}, Eval: {n_eval}")
    print(f"  Output: {output_dir}/")
