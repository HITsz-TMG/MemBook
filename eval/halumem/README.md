# HaluMem evaluation (MemBook)

Reproduce MemBook results on **HaluMem-Medium** / **HaluMem-Long**.

Data is **not** shipped in this repo. Place HaluMem JSON under `data/halumem/` (or set paths via the runner / env). See `.env-example`.

## Setup

```bash
# from repo root
pip install -e ".[bm25,dense]"   # extras as needed

cd eval/halumem
cp .env-example .env
# edit .env: OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
```

## Run

Preferred entry:

```bash
bash run_membook.bash
```

Edit the parameter block at the top of `run_membook.bash` (`dataset_list`, `model_*`, `max_users`, etc.).

Or call the Python runner directly:

```bash
python eval_membook.py --medium --frame membook --version medium
python evaluation.py --frame membook --version medium
```

## Layout (this release)

| File | Role |
|------|------|
| `run_membook.bash` | End-to-end: build memories → score |
| `eval_membook.py` | MemBook adapter + pipeline |
| `evaluation.py` | LLM-as-judge scoring |
| `llms.py` / `prompts.py` | API client + QA prompts |
| `base_adapter.py` | Shared adapter helpers |
| `eval_baseline.py` / `eval_tools.py` | Shared helpers imported by the MemBook runner |

## Notes

- Results go under `results/` (gitignored).
- Optional MemBook toggles are documented in `.env-example` (`MEMBOOK_*`).
