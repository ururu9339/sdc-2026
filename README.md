# sdc-2026

# Currvix

> Currvix forecasts which skills and jobs Malaysian employers will need, and shows each person the shortest reskilling path to a growing role.

---

## Team Information

| Field                  | Details                                     |
| ---------------------- | ------------------------------------------- |
| **Team Name**          | NoCoders.ai.                                |
| **Team Domain**        | Domain 1.                                   |
| **SDG Track (1–17)**   | SDG 4: Quality Education (Target 4.4) |
| **Team Leader**        | Nurali Zhan                                 |
| **University**         | Xiamen University Malaysia                  |
| **Number of Members**  | 4.                                          |

## Links

| Resource          | Link                                           |
| ----------------- | ---------------------------------------------- |
| GitHub Repository | https://github.com/ururu9339/sdc-2026.git      |
| Live Demo         | https://your-project.replit.app                |
| Slide Deck        | https://example.com/slides                     |
| Demo Video        | https://example.com/demo-video                 |

---

## Overview

### Problem

Workers, students and career counsellors in Malaysia have little forward-looking signal on which skills employers will need as AI reshapes jobs. Official labour statistics arrive late, and job ads are messy: the same skill appears as "Python", "python3" or "pengaturcaraan python", in English and Malay, so demand is hard to count. As a result, people pick courses and career moves on guesswork.

### Solution

Currvix turns raw job postings into a forecast of skill and job demand, then uses it to recommend reskilling paths:

1. **Normalize.** An embedding model (Qwen3-Embedding-4B) maps every skill or job mention in a posting or CV to a stable ID in a curated taxonomy aligned with MASCO, Malaysia's occupation classification.
2. **Count.** The mapped mentions become weekly demand series per skill and job.
3. **Forecast.** A LightGBM model forecasts each series' growth.
4. **Explain and match.** A job↔skill graph built from posting co-occurrence shows what each job needs. Matching compares a user's skills against it to find jobs within reach and the missing skills that are growing fastest.

Similarity only decides *which* skills and jobs are related. Growth numbers come only from the forecaster. When the evidence is too thin, Currvix says "insufficient data" instead of guessing.

### Key Features

- **Skill and job normalization:** messy mentions (abbreviations, typos, Malay) map to stable taxonomy IDs. Each mapping has a confidence score and records the model version that produced it. Unrecognized mentions are logged as candidate new skills.
- **Job↔skill graph:** every weight is labelled with its evidence, either measured co-occurrence in postings or an embedding-similarity fallback for jobs with few postings.
- **Cold start for new roles:** an unseen title such as "Cloud Security Engineer" borrows the skill profiles of its nearest known jobs, or abstains if none is close enough.
- **Reskilling match:** finds the jobs a user is close to and ranks their skill gaps by graph weight × forecast growth.
- **CV intake:** a CV can come in through a form or as an uploaded PDF or image, which is read with Tesseract OCR.
- **Evaluated and versioned:** a frozen gold set gates every change to the embedding model. Thresholds and scores are recorded in `models/version_log.json`.
- **Demand forecasting with a validation gate:** a LightGBM model forecasts each skill's and job's share of postings 12 weeks ahead, with 10–90% intervals. Walk-forward folds compare it with five baselines; a model that does not beat them is never published as validated, and its forecasts are served only as a labelled preview.
- **Honest outlook labels:** every forecast carries its as-of date, age, staleness flag and confidence. Series that are identical to another entity's are collapsed and flagged instead of being counted twice.
- **CV-to-report API:** one call turns a skill list or CV text into best-fit jobs, ranked skill gaps with a reason for each, a three-stage learning path and a market snapshot. Goals such as "interested in machine learning" are not counted as skills the user already has.

| Mention → ID mapping (305 gold mentions) | Top-1 | MRR |
| ---------------------------------------- | ----- | --- |
| Exact string match                       | 0.131 | 0.131 |
| Fuzzy match (rapidfuzz)                  | 0.887 | 0.920 |
| Qwen3-Embedding-0.6B                     | 0.931 | 0.962 |
| **Qwen3-Embedding-4B (used)**            | **0.967** | **0.982** |

### Tech Stack

| Layer         | Technology                                                        |
| ------------- | ----------------------------------------------------------------- |
| Frontend      | React 19, TypeScript, Vite, Tailwind CSS (`frontend/`), bilingual English / Bahasa Melayu; built on Replit and consuming the REST API in `docs/API_FOR_FRONTEND.md` |
| Backend       | Python, FastAPI, SQLModel                                         |
| Forecasting   | LightGBM, pandas, NumPy (weekly demand series, 12 week horizon)   |
| AI models     | Qwen3-Embedding-4B via Ollama; LightGBM (demand forecasting); Tesseract OCR |
| Database      | PostgreSQL + pgvector (taxonomy vectors, mappings, graph); SQLite (candidate API, for now) |
| CV intake     | Tesseract OCR, pdf2image, Pillow                                  |
| Data          | Open job posting datasets as CSV, loaded through a source registry |
| AI / Compute  | RunPod (GPU)                                                      |
| Hosting / Dev | Replit (API served from a committed data snapshot in `serving/`)  |

---

## Project Structure

```
.
├── embeddings/                  # Qwen3 embeddings: skill/job normalization, graph, matching
│   ├── embedder.py              # The only entry point for embeddings (model, dim, instruction, cache)
│   ├── db.py                    # Postgres connection + version health check
│   ├── schema.sql               # pgvector tables: entities, mappings, graph, forecasts
│   ├── build_index.py           # Embed the taxonomy into pgvector
│   ├── normalize.py             # Map posting/CV mentions to taxonomy IDs
│   ├── graph.py                 # Job↔skill weights (co-occurrence + embedding fallback)
│   ├── cold_start.py            # Skill profiles for unseen job titles
│   ├── match.py                 # Jobs within reach + ranked skill gaps for a user
│   ├── versioning.py            # models/version_log.json and the quality gate
│   ├── taxonomy/                # skills.csv, jobs.csv (MASCO unit-group codes)
│   └── eval/                    # Frozen gold sets + evaluate.py (metrics, baselines, thresholds)
├── frontend/                    # React website: profile form, CV upload, report, market signals
├── src/
│   ├── backend/                 # FastAPI app (serves the API and the built website)
│   │   ├── main.py              # App entry point, routers, CORS, error handling
│   │   ├── api/                 # forecast, report, health and candidate endpoints
│   │   ├── services/            # forecast store, CV-to-report engine, OCR, catalog
│   │   └── models/              # Pydantic / SQLModel response and request schemas
│   ├── sources/                 # Source adapters, taxonomy aliases, registry, validation
│   ├── training/                # Features, LightGBM training, baselines, walk-forward gate, audit
│   ├── pipeline/                # Agent loop, forecast export, serving-bundle refresh
│   └── validation/simulate.py   # Synthetic positive/negative controls for the gate
├── serving/                     # Committed snapshot the deployed API serves (forecasts, graph, report)
├── handoff/                     # Forecasts and job–skill graph as CSV, for loading into Postgres
├── models/version_log.json      # Active embedding version, metrics and thresholds
├── docs/                        # API contract, deployment, results, simulation
├── tests/                       # pytest suite
├── requirements.txt
└── README.md
```

### Architecture

```
Job postings ──► src/sources ──► observations ──► src/training (LightGBM + gate vs 5 baselines)
                                                        │
                                                        ▼
                                  serving/ bundle + handoff/ CSVs ──► entity_forecast (Postgres)
                                                                             │
User CV (form / PDF OCR / text) ──► FastAPI /api/v1/report ──► embeddings/match.py (Qwen3 + pgvector)
                                          │                         │  falls back to offline aliases
                                          ▼                         ▼
                 best-fit jobs · ranked skill gaps · learning path · market snapshot · outlook labels
```

The taxonomy is embedded once and stored in Postgres. Only a user's free-text skills need live embedding. Without `DATABASE_URL` the report uses the offline alias matcher and says so in its warnings.

---

## How to Run

### Environment variables

| Variable         | Purpose                                                     | Default                  |
| ---------------- | ----------------------------------------------------------- | ------------------------ |
| `DATABASE_URL`   | Postgres database with the `vector` extension available     | (required for `embeddings/`) |
| `OLLAMA_URL`     | Ollama server that hosts the embedding model                | `http://localhost:11434` |
| `OLLAMA_API_KEY` | Bearer token, if Ollama sits behind a key-checking proxy    | (none)                   |

Ollama has no authentication of its own. Never expose port 11434 to the internet. Use a firewall rule or a reverse proxy that checks `OLLAMA_API_KEY`. `python -m embeddings.db` warns if the configured endpoint is unprotected.

### Option 1 — Run on Replit

1. Import this repository via **Create Repl → Import from GitHub**. The committed `.replit` installs Python and
   Node, Tesseract and Poppler, builds the website and starts the API.
2. Click **Run**. The site and the API are served together on port 8000 (public port 80); the health check is
   `/api/v1/health`.
3. To publish, use **Deploy → Autoscale**. Optional secrets are listed above; without `DATABASE_URL` the report uses
   the offline skill matcher and says so.

### Option 2 — Run Locally

```bash
# 1. Clone and install
git clone https://github.com/ururu9339/sdc-2026.git
cd sdc-2026
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. OCR system packages (for PDF/image CV upload)
brew install tesseract poppler  # Ubuntu: apt install tesseract-ocr poppler-utils

# 3. Build the website (Node 20.19+ and pnpm 10)
cd frontend && pnpm install --frozen-lockfile && pnpm build && cd ..

# 4. Start the site and API on http://localhost:8000 (API docs at /docs)
python -m src.backend.main
```

For frontend work, `pnpm dev` in `frontend/` serves the site on port 5173 and forwards `/api` to port 8000.

### Embedding pipeline

Needs [Ollama](https://ollama.com) and PostgreSQL with [pgvector](https://github.com/pgvector/pgvector).

```bash
ollama pull qwen3-embedding:4b
export DATABASE_URL=postgresql://localhost/sdc26 PYTHONPATH=.

python -m embeddings.build_index                      # embed the taxonomy into pgvector
python -m embeddings.db                               # check stored vs live model version
python -m embeddings.normalize --type skill "k8s" "pengurusan projek"
python -m embeddings.normalize --input mentions.csv   # posting_id, month, mention, entity_type
python -m embeddings.graph                            # rebuild job↔skill weights
python -m embeddings.cold_start "Cloud Security Engineer"
python -m embeddings.match "python3" "SQL" "PowerBI"
```

### Run the tests

```bash
python3 -m pytest -q tests
```

The refresh tests expect a local `data/` folder with the raw sources, as on the forecasting team's machine.

### Evaluate the embedding model

```bash
python -m embeddings.eval.evaluate --compare qwen3-embedding:0.6b   # metrics vs baselines
python -m embeddings.eval.evaluate --record                         # gate, then log as active version
```

A new model, vector size or instruction text must score at least as well as the active version on the frozen gold set before `--record` accepts it. `build_index` refuses to replace the live index with settings that haven't passed this gate.

---

## How We Use Replit *(Required)*

- **Frontend:** built on Replit. It calls the REST API described in `docs/API_FOR_FRONTEND.md`.
- **Backend hosting:** the FastAPI backend runs on Replit with `uvicorn src.backend.main:app --host 0.0.0.0 --port 8000`. It serves a committed snapshot of the forecasts (`serving/`), so no model training runs on the server. See `docs/DEPLOY_REPLIT.md`.
- **Secrets:** `CORS_ORIGINS` and, when the Qwen matcher is connected, `DATABASE_URL`, `OLLAMA_URL` and `OLLAMA_API_KEY` are stored in Replit Secrets, never in the code.

---

## Challenges & Learnings

### Challenges

- **Building from scratch across many sources.** Job postings come from several sites, each with its own format, and skills are written inconsistently ("Python", "python3", "pengaturcaraan python") in both English and Malay. We had to design our own taxonomy of stable skill and job IDs, aligned with MASCO. We then built an embedding pipeline that maps every mention to one of those IDs, so demand isn't split across spellings.
- **No recent, date-ranged datasets.** We couldn't find open Malaysian job-posting data that is both recent and continuous. Our main series ends in September 2023, and the JobStreet listings (2024 to 2025) do not overlap it, so forecasts are served as a dated, unvalidated preview. We built and tested the pipeline on these files and on a labelled simulation, and designed it so newer data can be loaded without code changes.
- **Choosing thresholds without guessing.** The default similarity thresholds rejected almost every correct match. We hand-built a gold set of 305 mentions, including negatives with no correct ID, and picked the thresholds from measured precision.
- **Knowing when to abstain.** Unknown job titles always have *some* nearest neighbour. We added cut-offs taken from the evaluation, so the system answers "insufficient data" instead of inventing a profile.

### Learnings

- **Machine learning fundamentals:** how embedding models represent meaning, how to evaluate them against baselines (exact match, fuzzy match, a smaller model), and how a forecasting model such as LightGBM fits into the pipeline without being mixed up with similarity scores.
- **Tech stack:** Python, FastAPI and SQLModel for the backend; PostgreSQL with pgvector for vector search; Ollama to serve Qwen3-Embedding-4B; Tesseract OCR for CV uploads; rapidfuzz for baselines.
- **Agentic development:** we built with Replit and Claude Code. AI agents helped us plan, write, test and debug code quickly, while we stayed responsible for reviewing decisions and checking results against real evaluation numbers.
- **Data quality matters as much as the model.** Most remaining errors came from gaps in our taxonomy, not from the embedding model.

## Future Improvements

- [ ] Retrain the forecaster on real Malaysian postings data with enough history for the validation gate
- [ ] Grow the taxonomy from ESCO / O*NET and verify the MASCO codes against DOSM's MASCO 2020
- [ ] Add real posting mentions to the embedding gold set, then re-record the thresholds
- [ ] Add a small generative model for skill extraction and plain-language explanations
- [ ] Move the candidate database from SQLite to Postgres and run the embedding version check at startup
- [ ] Host live embedding on a protected endpoint

## Team

| Name        | Role                | GitHub                                   |
| ----------- | ------------------- | ---------------------------------------- |
| Nurali Zhan | Team Lead / Backend | [@username](https://github.com/username) |
| Shahzod Bakiev | Frontend / Backend | [@username](https://github.com/username) |
| Askaruly Ali | AI engine         | [@invalid-RJK](https://github.com/username) |
| Amirkhan Dossymov | AI / RunPod   | [@username](https://github.com/username) |

---

## Submission Checklist

- [ ] Team name, domain (Domain 1 / Domain 2) and SDG track filled in
- [ ] Slide deck link added
- [ ] Demo video link added
- [ ] Replit used, with proof of use in `submission/`
- [ ] RunPod used, with proof of use in `submission/`
- [ ] Preliminary submission completed (`SDC2026_TeamName_Preliminary.pdf`)
- [ ] Technology declaration completed (`SDC2026_TeamName_TechDeclaration.pdf`)
- [ ] Submission files renamed with your team name (replace `TeamName`)
- [ ] "How to Run" steps verified
- [ ] No API keys or secrets committed to the repository

## License

This project is licensed under the [Apache License 2.0](LICENSE).
