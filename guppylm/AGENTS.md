# guppylm/ KNOWLEDGE BASE

## OVERVIEW
Core Python package: synthetic data, tokenizer prep, model/training loop, checkpoint loading, and chat inference.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| CLI commands | `__main__.py` | Manual `sys.argv` dispatch; shifts `sys.argv` before chat argparse |
| Defaults | `config.py` | `GuppyConfig`, `TrainConfig`; edit here before changing call sites |
| Transformer | `model.py` | Attention, FFN, Block, `GuppyLM.generate()` |
| Dataset loader | `dataset.py` | JSONL lines with `text`; pads in `collate_fn` |
| Synthetic data | `generate_data.py` | many `gen_*` topic functions; category strings drive exports/eval |
| Tokenizer prep | `prepare_data.py` | ByteLevel BPE, vocab 4096, min_frequency 2 |
| Training | `train.py` | checkpoint/config writes to `checkpoints/` |
| Inference | `inference.py` | HF config compatibility + ChatML formatting |
| Eval cases | `eval_cases.py` | held-out prompt/style expectations |

## CONVENTIONS
- Use relative imports inside the package: `from .config import GuppyConfig`.
- Keep model defaults aligned across `config.py`, README architecture table, notebooks, and `docs/index.html` config.
- `generate_data.py` naming: private phrase helpers use leading `_`; public topic generators use `gen_*`.
- Dataset categories are lowercase labels such as `greeting`, `temp_hot`, `glass_tap`.
- Training expects `data/train.jsonl`, `data/eval.jsonl`, and `data/tokenizer.json` unless configs change.
- `best_model.pt` embeds the usable model config; `train.py` also writes nested `checkpoints/config.json` as `{"model": ..., "train": ...}` metadata.
- `inference.py` and `tools/export_onnx.py` tolerate legacy checkpoints or HF-style state dicts; `tools/export_model.py` expects `model_state_dict` + `config`.
- `GuppyInference.chat_completion()` returns an OpenAI-like `{"choices": [{"message": ...}]}` shape.

## ANTI-PATTERNS
- Do not remove truncation after `<|im_end|>` or `<|im_start|>` in inference; it prevents next-turn leakage.
- Do not assume conversation history improves results; single-turn prompts are the intended reliable mode.
- Do not raise max sequence/architecture knobs without updating tokenizer/training/docs/browser config together.
- Do not rely on nested `checkpoints/config.json` for custom inference/export settings unless readers are updated to consume the nested `model` key.
- Do not replace the vanilla architecture with advanced transformer features unless the README design decision changes too.
- Avoid moving artifact paths under package-relative paths; current scripts and README assume repo-root relative paths.

## QUICK VALIDATION
```bash
python -m guppylm
python -m guppylm chat --prompt "hi guppy"
```

`chat` needs `checkpoints/best_model.pt` and `data/tokenizer.json`; run `python -m guppylm download` or train first.
