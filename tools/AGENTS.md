# tools/ KNOWLEDGE BASE

## OVERVIEW
Operational scripts for notebooks, HuggingFace dataset/model export, and ONNX browser artifacts.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Regenerate notebooks | `make_colab.py` | Writes `train_guppylm.ipynb`, `use_guppylm.ipynb` |
| HF model export | `export_model.py` | `hf_export/` layout, optional repo upload |
| HF dataset export | `export_dataset.py` | local JSONL first, optional dataset push |
| Browser ONNX export | `export_onnx.py` | default `docs/model.onnx`, copies tokenizer |
| Model card | `model_card.md` | copied to `hf_export/README.md` |
| Dataset card | `dataset_card.md` | uploaded as dataset README |

## CONVENTIONS
- Scripts are direct CLI files, not installed console entry points.
- `argparse` is used in export scripts; keep defaults aligned with root docs.
- `.env` is optional; CLI args override or complement `HF_TOKEN`, `HF_REPO`, `HF_DATASET`.
- Do not print or commit tokens. `.env` is gitignored.
- `export_model.py` writes HF standard files: `pytorch_model.bin`, `config.json`, `tokenizer.json`, `README.md`, `assets/guppy.png`.
- `export_onnx.py` quantizes to uint8 by default and writes assets beside the static demo.
- `make_colab.py` embeds source files into notebooks; source changes may require `make notebook`.

## ANTI-PATTERNS
- `export_dataset.py` says `HF_REPO` in the docstring but code uses `HF_DATASET`; follow code behavior.
- `export_dataset.py` intentionally saves local JSONL before optional push; preserve this safety behavior.
- `export_model.py` has an empty `except Exception: pass` around remote cleanup; be careful before broadening silent failures.
- `export_onnx.py` imports package internals through a `sys.path` pointing directly at `guppylm/`, so it uses top-level `from config import ...`.
- Do not assume export dependencies are installed by `requirements.txt`; install HF/ONNX extras when running release scripts.

## COMMANDS
```bash
python3 tools/make_colab.py
python tools/export_model.py --local-only
python tools/export_dataset.py --local-only
python tools/export_onnx.py
python tools/export_onnx.py --no-quantize
```
