"""WeChat persona inference -- chat with the trained persona model.

Thin wrapper around GuppyInference with WeChat-specific default paths
and longer max_tokens for Chinese responses.
"""

import os

from ..inference import GuppyInference


class WechatPersonaInference:
    """Chat inference for WeChat persona model.

    Reuses GuppyInference with WeChat-specific defaults.
    The core logic (tokenization, generation, boundary truncation) is identical.
    """

    def __init__(self, checkpoint_path="wechat_checkpoints/best_model.pt",
                 tokenizer_path="wechat_data/tokenizer.json", device="cpu"):
        self._engine = GuppyInference(checkpoint_path, tokenizer_path, device)

    def chat_completion(self, messages, temperature=0.7, max_tokens=128,
                        top_k=50, **kwargs):
        """Chat completion with persona model.

        Default max_tokens=128 (longer than fish-chat's 64) because Chinese
        responses typically need more tokens.
        """
        return self._engine.chat_completion(
            messages, temperature=temperature,
            max_tokens=max_tokens, top_k=top_k, **kwargs,
        )

    @property
    def config(self):
        return self._engine.config

    @property
    def model(self):
        return self._engine.model


def main():
    """CLI for persona chat."""
    import argparse

    p = argparse.ArgumentParser(description="Chat with WeChat persona model")
    p.add_argument("--checkpoint", default="wechat_checkpoints/best_model.pt")
    p.add_argument("--tokenizer", default="wechat_data/tokenizer.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--prompt", "-p", help="Single prompt mode: ask one question and exit")
    p.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")
    p.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    args = p.parse_args()

    if not os.path.exists(args.checkpoint):
        print("Persona model not found. Train one first:")
        print()
        print("  python -m guppylm wechat-prepare")
        print("  python -m guppylm wechat-train")
        return

    engine = WechatPersonaInference(args.checkpoint, args.tokenizer, args.device)

    if args.prompt:
        result = engine.chat_completion(
            [{"role": "user", "content": args.prompt}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        print(result["choices"][0]["message"]["content"])
        return

    print("\nWeChat Persona Chat (type 'quit' to exit)")
    while True:
        inp = input("\nYou> ").strip()
        if inp.lower() in ("quit", "exit", "q"):
            break
        if not inp:
            continue
        result = engine.chat_completion(
            [{"role": "user", "content": inp}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        msg = result["choices"][0]["message"]
        if msg.get("content"):
            print(f"Persona> {msg['content']}")


if __name__ == "__main__":
    main()
