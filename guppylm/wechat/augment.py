"""Data augmentation for WeChat persona training samples.

Two strategies:
1. Template-based: synonym replacement, random deletion, sentence reordering
2. LLM-based: use an LLM API to generate stylistic variations

Both operate on ChatML-formatted samples from guppylm/wechat/convert.py.
"""

import json
import os
import random
import re

from .config import WechatAugmentConfig
from .convert import ChatMLSample


# ── Chinese synonym dictionary (common casual/chat synonyms) ────────────────

_SYNONYM_MAP: dict[str, list[str]] = {
    # Common casual expressions
    "好的": ["好", "行", "OK", "ok", "可以"],
    "不行": ["不可以", "没戏"],
    "对": ["嗯", "是的", "对呀", "对啊"],
    "嗯": ["嗯嗯", "好", "对", "嗯呢"],
    "哈哈": ["哈哈哈", "笑死"],
    "谢谢": ["谢啦", "多谢", "感谢"],
    "不好意思": ["抱歉", "不好意思啊", "对不住"],
    "怎么样": ["如何", "咋样", "怎样"],
    "不知道": ["不清楚", "不了解", "不晓得"],
    "没问题": ["OK", "包在我身上"],
    "算了": ["罢了", "不管了"],
    "真的": ["确实", "真的假的", "真"],
    "没事": ["没关系", "没事儿", "不打紧"],
    "什么": ["啥", "啥子"],
    "为什么": ["为啥", "怎么"],
    "怎么": ["咋", "如何"],
    "这样": ["这样子", "这么"],
    "那样": ["那样子", "那么"],
    "可以": ["能", "行"],
    "应该": ["应当", "估计"],
    "可能": ["也许", "大概", "或许"],
    "觉得": ["感觉", "认为"],
    "知道": ["晓得", "了解"],
    "现在": ["目前", "这会儿"],
    "今天": ["今日", "今儿"],
    "明天": ["明儿", "次日"],
    "昨天": ["昨儿", "昨日"],
    "非常": ["特别", "超级"],
    "很": ["特别", "非常", "超"],
    "特别": ["尤其", "格外"],
    "然后": ["接着", "后来"],
    "所以": ["因此", "于是"],
    "但是": ["不过", "可是"],
    "因为": ["由于", "因"],
    "如果": ["假如", "要是"],
    "虽然": ["固然", "虽"],
    "而且": ["并且", "此外"],
    "还是": ["或者", "亦"],
    "已经": ["早就", "已"],
    "还没": ["尚未", "没"],
    "一下": ["一会儿", "稍等"],
    "等等": ["等下", "稍后"],
    "马上": ["立刻", "这就"],
    "其实": ["实际上", "说实话"],
    "当然": ["自然", "肯定"],
    "确实": ["的确", "真的"],
}


def synonym_replace(text: str, prob: float = 0.1, seed: int | None = None) -> str:
    """Replace words with synonyms using a Chinese synonym dictionary.

    Each matching word is replaced with probability `prob`.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    result = text
    for word, synonyms in _SYNONYM_MAP.items():
        if word in result and rng.random() < prob:
            replacement = rng.choice(synonyms)
            # Only replace the first occurrence to avoid over-augmentation
            result = result.replace(word, replacement, 1)
    return result


def random_delete(text: str, prob: float = 0.05, seed: int | None = None) -> str:
    """Randomly delete characters from the text with probability `prob`.

    Preserves Chinese punctuation and sentence boundaries.
    Does not delete the first or last character.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    if len(text) <= 3:
        return text

    # Characters to preserve (punctuation, special)
    preserve = set("，。！？、；：""''（）【】…— \n")

    chars = list(text)
    result = [chars[0]]  # Always keep first char
    for i in range(1, len(chars) - 1):
        if chars[i] in preserve:
            result.append(chars[i])
        elif rng.random() > prob:
            result.append(chars[i])
    result.append(chars[-1])  # Always keep last char
    return "".join(result)


def sentence_reorder(text: str, seed: int | None = None) -> str:
    """Reorder independent clauses in the message.

    Splits on Chinese punctuation (，。！？), shuffles non-adjacent clauses.
    Only reorder if there are >= 3 clauses to avoid nonsensical results.
    Preserves the first and last clause positions.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    # Split on sentence-ending punctuation while keeping the punctuation
    parts = re.split(r"([，。！？；])", text)
    if len(parts) < 5:  # Need at least 3 clauses with punctuation
        return text

    # Reconstruct clauses with their trailing punctuation
    clauses = []
    for i in range(0, len(parts) - 1, 2):
        clause = parts[i]
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        if clause.strip():
            clauses.append(clause + punct)

    # Handle trailing text without punctuation
    if len(parts) % 2 == 1 and parts[-1].strip():
        clauses.append(parts[-1])

    if len(clauses) < 3:
        return text

    # Keep first and last, shuffle middle
    first, middle, last = clauses[0], clauses[1:-1], clauses[-1]
    rng.shuffle(middle)
    return first + "".join(middle) + last


# ── Template augmentation ────────────────────────────────────────────────────


def augment_template(sample: ChatMLSample, config: WechatAugmentConfig) -> list[ChatMLSample]:
    """Apply template-based augmentation to a single sample.

    Returns list including the original + augmented variants.
    Only augments the assistant (target person) portion of the ChatML.
    """
    results = [sample]  # Always include original

    if not config.template_enabled:
        return results

    # Extract assistant content from ChatML
    text = sample.text
    assistant_match = re.search(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", text, re.DOTALL)
    if not assistant_match:
        return results

    original_content = assistant_match.group(1)

    # Apply each augmentation independently
    augmented_contents = []

    # 1. Synonym replacement
    if config.synonym_replace_prob > 0:
        new_content = synonym_replace(original_content, config.synonym_replace_prob)
        if new_content != original_content:
            augmented_contents.append(new_content)

    # 2. Random deletion
    if config.random_delete_prob > 0:
        new_content = random_delete(original_content, config.random_delete_prob)
        if new_content != original_content:
            augmented_contents.append(new_content)

    # 3. Sentence reorder
    if config.sentence_reorder:
        new_content = sentence_reorder(original_content)
        if new_content != original_content:
            augmented_contents.append(new_content)

    # Create augmented samples
    for new_content in augmented_contents:
        new_text = text.replace(
            f"<|im_start|>assistant\n{original_content}<|im_end|>",
            f"<|im_start|>assistant\n{new_content}<|im_end|>",
        )
        results.append(ChatMLSample(
            text=new_text,
            category="wechat_persona_aug",
            n_turns=sample.n_turns,
            source_contact=sample.source_contact,
        ))

    return results


# ── LLM-based augmentation ──────────────────────────────────────────────────


def augment_with_llm(
    sample: ChatMLSample,
    config: WechatAugmentConfig,
) -> list[ChatMLSample]:
    """Use an LLM API to generate variations of the target person's messages.

    The LLM is prompted to rewrite the message in the same style.
    Only the assistant (target) content is augmented.

    Returns list of augmented samples (does not include the original).
    """
    results = []

    # Extract assistant content
    text = sample.text
    assistant_match = re.search(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", text, re.DOTALL)
    if not assistant_match:
        return results

    original_content = assistant_match.group(1)

    try:
        from openai import OpenAI

        api_key = config.llm_api_key or os.environ.get("LLM_API_KEY", "")
        if not api_key:
            return results

        client = OpenAI(
            api_key=api_key,
            base_url=config.llm_api_base or None,
        )

        prompt = (
            "以下是一个人说的原话。请用同样的说话风格和语气改写这句话，"
            "保持意思相近但表达方式不同。只输出改写结果，不要加任何前缀：\n"
            f"原话：{original_content}\n改写："
        )

        n_variants = max(1, int(config.llm_augment_ratio))
        for _ in range(n_variants):
            response = client.chat.completions.create(
                model=config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=len(original_content) * 2,
            )
            new_content = response.choices[0].message.content.strip()
            if new_content and new_content != original_content:
                new_text = text.replace(
                    f"<|im_start|>assistant\n{original_content}<|im_end|>",
                    f"<|im_start|>assistant\n{new_content}<|im_end|>",
                )
                results.append(ChatMLSample(
                    text=new_text,
                    category="wechat_persona_llm",
                    n_turns=sample.n_turns,
                    source_contact=sample.source_contact,
                ))
    except ImportError:
        print("[WARN] openai package not installed, skipping LLM augmentation")
    except Exception as e:
        print(f"[WARN] LLM augmentation failed: {e}")

    return results


# ── Main augmentation pipeline ──────────────────────────────────────────────


def augment(config: WechatAugmentConfig | None = None) -> None:
    """Run the full augmentation pipeline.

    Reads wechat_data/chatml/train.jsonl, applies augmentation, and writes
    to wechat_data/augmented/train.jsonl. Copies eval data unchanged.
    """
    if config is None:
        config = WechatAugmentConfig()

    input_dir = config.input_dir
    output_dir = config.output_dir

    # Read training samples
    train_path = os.path.join(input_dir, "train.jsonl")
    if not os.path.exists(train_path):
        print(f"Training data not found at {train_path}")
        print("Run 'python -m guppylm wechat-prepare' (convert step) first.")
        return

    samples = []
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            samples.append(ChatMLSample(
                text=data["text"],
                category=data.get("category", "wechat_persona"),
                n_turns=data.get("n_turns", 1),
                source_contact=data.get("source_contact", ""),
            ))

    print(f"Loaded {len(samples)} training samples")

    # Apply template augmentation
    augmented = []
    for sample in samples:
        augmented.extend(augment_template(sample, config))

    print(f"After template augmentation: {len(augmented)} samples")

    # Apply LLM augmentation (if enabled)
    if config.llm_enabled:
        llm_augmented = []
        for sample in samples:
            llm_augmented.extend(augment_with_llm(sample, config))
        augmented.extend(llm_augmented)
        print(f"After LLM augmentation: {len(augmented)} samples")

    # Write output
    os.makedirs(output_dir, exist_ok=True)

    # Write augmented training data
    with open(os.path.join(output_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for s in augmented:
            f.write(json.dumps({"text": s.text, "category": s.category}, ensure_ascii=False) + "\n")

    # Copy eval data unchanged
    eval_src = os.path.join(input_dir, "eval.jsonl")
    eval_dst = os.path.join(output_dir, "eval.jsonl")
    if os.path.exists(eval_src):
        import shutil
        shutil.copy2(eval_src, eval_dst)
        print(f"Copied eval data to {eval_dst}")

    print(f"\nAugmentation complete:")
    print(f"  Input: {len(samples)} samples")
    print(f"  Output: {len(augmented)} samples")
    print(f"  Augmented data: {output_dir}/")
