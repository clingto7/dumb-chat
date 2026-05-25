"""Entry point for: python -m guppylm"""

import os
import sys

CHECKPOINT_PATH = "checkpoints/best_model.pt"
TOKENIZER_PATH = "data/tokenizer.json"
HF_REPO = "arman-bd/guppylm-9M"
HF_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main"


def download_model():
    """Download pre-trained GuppyLM from HuggingFace."""
    import urllib.request

    files = [
        (f"{HF_BASE}/pytorch_model.bin", CHECKPOINT_PATH),
        (f"{HF_BASE}/tokenizer.json", TOKENIZER_PATH),
        (f"{HF_BASE}/config.json", "checkpoints/config.json"),
    ]

    print(f"Downloading GuppyLM from {HF_REPO}...\n")
    for url, dest in files:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        name = os.path.basename(dest)
        print(f"  {name}...", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / 1e6
        print(f"{size_mb:.1f} MB")

    print("\nDone! Run: python -m guppylm chat")


def main():
    if len(sys.argv) < 2:
        print("GuppyLM — A tiny fish brain")
        print()
        print("Usage:")
        print("  python -m guppylm train            Train the model")
        print("  python -m guppylm prepare          Generate data & train tokenizer")
        print("  python -m guppylm chat             Chat with Guppy")
        print("  python -m guppylm download         Download pre-trained model from HuggingFace")
        print()
        print("WeChat persona commands:")
        print("  python -m guppylm wechat-decrypt   Extract keys & decrypt WeChat databases")
        print("  python -m guppylm wechat-extract   Extract chat records from decrypted DBs")
        print("  python -m guppylm wechat-prepare   Convert to ChatML + augment + train tokenizer")
        print("  python -m guppylm wechat-train      Train persona model")
        print("  python -m guppylm wechat-chat       Chat with persona model")
        return

    cmd = sys.argv[1]
    sys.argv = sys.argv[1:]

    if cmd == "prepare":
        from .prepare_data import prepare
        prepare()

    elif cmd == "train":
        from .train import train
        train()

    elif cmd == "download":
        download_model()

    elif cmd == "chat":
        if not os.path.exists(CHECKPOINT_PATH):
            print("Model not found. Download the pre-trained model first:\n")
            print("  python -m guppylm download\n")
            print("Or train your own:\n")
            print("  python -m guppylm prepare")
            print("  python -m guppylm train")
            return

        from .inference import main as inference_main
        inference_main()

    elif cmd == "wechat-decrypt":
        from .wechat.decrypt import decrypt_all
        from .wechat.config import WechatDecryptConfig
        decrypt_all(WechatDecryptConfig())

    elif cmd == "wechat-extract":
        from .wechat.extract import extract_and_save
        from .wechat.config import WechatExtractConfig
        extract_and_save(WechatExtractConfig())

    elif cmd == "wechat-prepare":
        from .wechat.convert import convert_to_chatml
        from .wechat.augment import augment
        from .wechat.prepare import prepare
        from .wechat.config import WechatChatMLConfig, WechatAugmentConfig, WechatTrainConfig
        # target_wxid can be passed as second arg: python -m guppylm wechat-prepare <wxid>
        chatml_cfg = WechatChatMLConfig()
        if len(sys.argv) > 1:
            chatml_cfg.target_wxid = sys.argv[1]
            sys.argv = sys.argv[1:]  # consume the wxid arg
        convert_to_chatml(chatml_cfg)
        augment(WechatAugmentConfig())
        prepare(WechatTrainConfig())

    elif cmd == "wechat-train":
        from .wechat.train import train
        train()

    elif cmd == "wechat-chat":
        from .wechat.inference import main as wechat_chat_main
        wechat_chat_main()

    else:
        print(f"Unknown command: {cmd}")
        print("Run 'python -m guppylm' for usage.")


main()
