# Repository Guidelines

## Project Structure & Module Organization
- `data_prep/`: end-to-end video processing (YOLO/ViTPose detection, VLM descriptions, temporal sampling, geometry utilities) with the pipeline entrypoint `data_prep/process_videos.py`; helpers live under `data_prep/pipeline/`.
- `training/`: Qwen-based VLA training loop (`train_vla.py`) and model definition (`vla_model.py`).
- `inference/`: Blender-based render tooling (`render_trace.py`) plus overlay/round-trip sanity scripts.
- `models/`: pretrained checkpoints such as MotionAGFormer; `data/` holds processed pose/trace outputs; `tests/` contains pytest suites; `notebooks/` is for exploratory workflows.

## Build, Test, and Development Commands
- Create an environment and install deps: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` (CUDA-enabled PyTorch recommended).
- Data pipeline (outputs to `data/pose_processed`): `python data_prep/process_videos.py --input-dir data/raw --output-dir data/pose_processed --enable_vlm --limit 2 --debug`.
- Train VLA: `python training/train_vla.py --config config_qwen3.yaml --checkpoint path/to/checkpoint.pth` (config controls dataset roots, schedule, and logging).
- Render traces headless: `blender --background --python inference/render_trace.py -- --trace data/kinetics_processed/example.npz --output-dir inference/renders --preview`.
- Tests: `pytest tests` for fast geometry/path coverage; enable the heavy VLM check with `pytest tests/test_vlm_integration.py::test_vlm_output -v --no-skip` after generating pose+description data.

## Coding Style & Naming Conventions
- Python, 4-space indent, type hints where possible; prefer snake_case for functions/variables and CapWords for classes.
- Keep modules importable from repo root (tests append the project root to `sys.path`); avoid hard-coded absolute paths outside `movement/`.
- Match existing docstring style (triple quotes, concise descriptions) and keep functions small with clear responsibilities.

## Testing Guidelines
- Pytest is the harness; favor small, deterministic unit tests in `tests/` alongside any new module.
- Add sample fixtures rather than large binaries; integration tests that require models (e.g., VLM) should be marked `@pytest.mark.skip` or guarded with fixtures similar to `ensure_vlm_test_data`.
- When adding data pipeline steps, supply at least one shape/value assertion mirroring `tests/test_data_prep.py` patterns.

## Commit & Pull Request Guidelines
- Follow the existing Conventional Commit style seen in history (`feat(data_prep): ...`, `fix(training): ...`, `chore: ...`); use present tense and a scoped prefix.
- PRs should include a concise summary, expected inputs/outputs, test evidence (`pytest ...` or sample render command), and any new data/model requirements. Add screenshots or short notes for visual outputs under `inference/renders` when relevant.

## Data, Models, and Security Notes
- Large artifacts (renders, `.npz` exports, Blender caches) should stay out of version control; keep outputs under `data/` or `inference/renders` and clean them before committing.
- Ensure required checkpoints exist locally (`models/motionagformer-b-h36m.pth.tr`, ViTPose weights) and avoid embedding access keys in configs or notebooks. Use environment variables for credentials when invoking AWS/CLI tooling.
