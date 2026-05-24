# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-11 Asia/Shanghai
**Commit:** a30df30
**Branch:** main

## OVERVIEW
GuppyLM is a small Python/PyTorch LLM teaching repo: synthetic fish-chat data, BPE tokenizer, vanilla transformer training, local chat, HuggingFace export, and a static ONNX/WASM browser demo.

## STRUCTURE
```
guppylm/
├── guppylm/              # core package: config, data, model, train, inference
├── tools/                # notebook, dataset, HF model, ONNX export scripts
├── docs/                 # GitHub Pages browser demo + committed ONNX/tokenizer
├── assets/               # README/model-card static images
├── train_guppylm.ipynb   # generated/curated Colab training path
├── use_guppylm.ipynb     # generated/curated Colab inference path
├── requirements.txt      # minimal runtime deps only
└── Makefile              # notebook regeneration only
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| CLI dispatch | `guppylm/__main__.py` | `python -m guppylm` commands: prepare/train/chat/download |
| Hyperparameters | `guppylm/config.py` | dataclasses; root-relative `data/`, `checkpoints/` |
| Model architecture | `guppylm/model.py` | vanilla transformer; tied token embedding/LM head |
| Data generation | `guppylm/generate_data.py` | 60 fish-topic generators; largest file |
| Tokenizer/data prep | `guppylm/prepare_data.py` | writes ChatML JSONL + BPE tokenizer |
| Training | `guppylm/train.py` | cosine LR, AMP on CUDA, checkpoint writes |
| Local chat | `guppylm/inference.py` | ChatML prompt format; boundary truncation |
| Manual eval cases | `guppylm/eval_cases.py` | data-driven checks, not pytest |
| Notebook generation | `tools/make_colab.py` | embeds source files into notebooks |
| HF exports | `tools/export_model.py`, `tools/export_dataset.py` | `.env` or args for tokens/repos |
| Browser export | `tools/export_onnx.py`, `docs/` | quantized ONNX + tokenizer for static demo |

## CODE MAP
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `GuppyConfig` | dataclass | `guppylm/config.py` | model shape + token IDs |
| `TrainConfig` | dataclass | `guppylm/config.py` | training defaults + artifact dirs |
| `GuppyLM` | `nn.Module` | `guppylm/model.py` | core transformer and generation |
| `GuppyDataset` | `Dataset` | `guppylm/dataset.py` | JSONL-to-token IDs loader |
| `train()` | function | `guppylm/train.py` | full training loop |
| `GuppyInference` | class | `guppylm/inference.py` | checkpoint load + chat completion |
| `generate_dataset()` | function | `guppylm/generate_data.py` | synthetic train/eval data |
| `export_and_push()` | function | `tools/export_model.py` | HF model layout/export |
| `export_onnx()` | function | `tools/export_onnx.py` | browser ONNX artifact |

## CONVENTIONS
- No package metadata: use source tree directly; no `pyproject.toml`, `setup.py`, console scripts, lockfile, linter config, or test config.
- Python style: stdlib imports, third-party imports, relative package imports; 4 spaces; short docstrings; mostly double quotes.
- Typing is light. Dataclasses in `config.py` are the typed center; most functions have no return annotations.
- Runtime paths are repo-root relative: `data/`, `dataset/`, `checkpoints/`, `hf_export/`, `docs/`.
- ChatML-like samples use `<|im_start|>user\n...<|im_end|>` and `<|im_start|>assistant\n...<|im_end|>`.
- Special token IDs: `0=<pad>`, `1=<|im_start|>`, `2=<|im_end|>`.
- Guppy output style: short, lowercase, fish/tank/world-through-water voice.

## ANTI-PATTERNS (THIS PROJECT)
- Do not make behavior depend on a system prompt; personality is baked into training samples/weights.
- Avoid multi-turn quality assumptions; README calls single-turn the reliable path because context is 128 tokens.
- Do not add GQA/RoPE/SwiGLU/early-exit casually; current design intentionally stays vanilla at ~9M params.
- Keep inference boundary truncation at `<|im_end|>` / `<|im_start|>` to prevent leaking into the next turn.
- Do not commit `.env`, generated `data/`, `dataset/`, `checkpoints/`, generic `*.pt`, `*.bin`, or generic `*.onnx` artifacts.
- Exception: `docs/*.onnx` and `docs/*.json` are intentionally unignored for GitHub Pages.

## COMMANDS
```bash
python -m pip install -r requirements.txt
python -m guppylm download
python -m guppylm chat --prompt "tell me a joke"
python -m guppylm prepare
python -m guppylm train
make notebook
python tools/export_model.py --local-only
python tools/export_dataset.py --local-only
python tools/export_onnx.py
```

## NOTES
- No CI/workflows found. Validation is manual unless added later.
- `requirements.txt` omits release/export extras: `huggingface_hub`, `onnx`, `onnxruntime`, `onnxscript`.
- `Makefile` only regenerates notebooks; it is not a build/test pipeline.
- `docs/download.sh` ends with stale text: `cd web && python -m http.server 8080`; actual directory is `docs`.
- `tools/export_dataset.py` swallows generator exceptions; inspect generated counts/categories if changing data generators.
