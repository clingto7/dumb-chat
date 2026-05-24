# docs/ KNOWLEDGE BASE

## OVERVIEW
Static GitHub Pages browser demo: single HTML app using ONNX Runtime Web, committed `model.onnx`, tokenizer, and logo assets.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Browser app | `index.html` | CSS, UI, tokenizer loader, ONNX inference in one file |
| Model artifact | `model.onnx` | quantized uint8 export from `tools/export_onnx.py` |
| Tokenizer artifact | `tokenizer.json` | copied beside ONNX model |
| Asset downloader | `download.sh` | pulls `model.onnx` + `tokenizer.json` from HF |
| Logo | `guppy.png` | header/loading image |

## CONVENTIONS
- `index.html` loads assets from `MODEL_BASE = "."`; keep `model.onnx` and `tokenizer.json` in this directory for Pages.
- ONNX Runtime Web is loaded from jsDelivr: `onnxruntime-web@1.21.0`.
- Browser config duplicates model defaults: vocab 4096, max seq 128, d_model 384, 6 layers, 6 heads, FFN 768, token IDs 0/1/2.
- Generation defaults in browser: temperature 0.7, top_k 50, max_tokens 32.
- Demo is client-only: no server/API keys; inference runs locally after downloading artifacts.
- `docs/*.onnx` and `docs/*.json` are explicitly unignored despite global model-artifact ignores.

## ANTI-PATTERNS
- Do not move artifacts without changing `MODEL_BASE` and Pages assumptions.
- Do not update architecture constants in HTML without regenerating/exporting a compatible ONNX model.
- Do not rely on `requirements.txt` for browser export; ONNX tooling is an extra release dependency.
- `download.sh` currently says `cd web`; correct mental model is `docs`.

## COMMANDS
```bash
python tools/export_onnx.py
bash docs/download.sh
python3 -m http.server 8080 --directory docs
```
