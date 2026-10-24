# sdc-2026

# Currvix

> Currvix forecasts which skills and jobs Malaysian employers will need, and shows each person the shortest reskilling path to a growing role.

---

## Team Information

| Field                  | Details                                     |
| ---------------------- | ------------------------------------------- |
| **Team Name**          | NoCoders.ai.                                |
| **Team Domain**        | Domain 1.                                   |
| **SDG Track (1–17)**   | e.g. SDG 11 — Sustainable Cities and Communities |
| **Team Leader**        | Nurali Zhan                                 |
| **University**         | Xiamen University Malaysia                  |
| **Number of Members**  | 4.                                          |

## Links

| Resource          | Link                                           |
| ----------------- | ---------------------------------------------- |
| GitHub Repository | https://github.com/ururu9339/sdc-2026.git      |
| Live Demo         | https://your-project.replit.app                |
| RunPod Endpoint   | https://api.runpod.ai/v2/your-endpoint-id      |
| Slide Deck        | https://example.com/slides                     |
| Demo Video        | https://example.com/demo-video                 |

---

## Overview

### Problem

Workers, students and career counsellors in Malaysia have little forward-looking signal on which skills employers will need as AI reshapes jobs. Official labour statistics arrive late, and job ads are messy: the same skill appears as "Python", "python3" or "pengaturcaraan python", in English and Malay, so demand is hard to count. As a result, people pick courses and career moves on guesswork.

### Solution

Currvix turns raw job postings into a forecast of skill and job demand, then uses it to recommend reskilling paths:

1. **Normalize.** An embedding model (Qwen3-Embedding-4B) maps every skill or job mention in a posting or CV to a stable ID in a curated taxonomy aligned with MASCO, Malaysia's occupation classification.
2. **Count.** The mapped mentions become monthly demand series per skill and job.
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

| Mention → ID mapping (305 gold mentions) | Top-1 | MRR |
| ---------------------------------------- | ----- | --- |
| Exact string match                       | 0.131 | 0.131 |
| Fuzzy match (rapidfuzz)                  | 0.887 | 0.920 |
| Qwen3-Embedding-0.6B                     | 0.931 | 0.962 |
| **Qwen3-Embedding-4B (used)**            | **0.967** | **0.982** |

### Tech Stack

| Layer         | Technology                                                        |
| ------------- | ----------------------------------------------------------------- |
| Frontend      | Planned                                                           |
| Backend       | Python, FastAPI, SQLModel                                         |
| Database      | PostgreSQL + pgvector (taxonomy vectors, mappings, graph); SQLite (candidate API, for now) |
| AI models     | Qwen3-Embedding-4B via Ollama; LightGBM (demand forecasting); Tesseract OCR |
| AI / Compute  | RunPod (GPU)                                                      |
| Hosting / Dev | Replit                                                            |

---

## Project Structure

```
.
├── embeddings/                  # Skill/job normalization, graph, matching
│   ├── embedder.py              # The only entry point for embeddings (model, dim, instruction, cache)
│   ├── db.py                    # Postgres connection + version health check
│   ├── schema.sql               # pgvector tables: entities, mappings, graph, forecasts
│   ├── build_index.py           # Embed the taxonomy into pgvector
│   ├── normalize.py             # Map posting/CV mentions to taxonomy IDs
│   ├── graph.py                 # Job↔skill weights (co-occurrence + embedding fallback)
│   ├── cold_start.py            # Skill profiles for unseen job titles
│   ├── match.py                 # Jobs within reach + ranked skill gaps for a user
│   ├── versioning.py            # Read/write models/version_log.json, quality gate
│   ├── taxonomy/
│   │   ├── skills.csv           # Canonical skills (id, name, alt_labels, description)
│   │   └── jobs.csv             # Canonical jobs with MASCO unit-group codes
│   └── eval/
│       ├── gold_mentions.csv    # Frozen gold set: mention → ID
│       ├── gold_job_pairs.csv   # Frozen gold set: similar / dissimilar job pairs
│       └── evaluate.py          # Metrics, baselines, threshold selection
├── models/
│   └── version_log.json         # Active model versions, metrics and thresholds
├── src/
│   ├── backend/
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── db.py                # SQLModel engine
│   │   ├── api/
│   │   │   ├── candidates.py    # CV intake: form and PDF/image upload
│   │   │   └── forecast.py      # Forecast endpoints (planned)
│   │   ├── models/candidate.py  # Candidate schema
│   │   └── services/ocr.py      # OCR and CV section parsing
│   ├── pipeline/agent_loop.py   # Posting ingestion pipeline (planned)
│   └── sources/sources_registry.json  # Job posting sources (planned)
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

### Architecture

```
Job postings ──► normalize.py ──► posting_entity (posting, month, skill/job ID)
                    │                     │
           Qwen3-Embedding-4B             ├──► monthly counts ──► LightGBM ──► entity_forecast
           (Ollama) + pgvector            │
                                          └──► graph.py ──► job_skill_edge
                                                                  │
User CV (form / PDF OCR) ──► FastAPI ──► match.py ◄───────────────┴── entity_forecast
                                           │
                                           └──► jobs within reach + ranked skill gaps
```

Embeddings are computed offline for the taxonomy and historical postings, then stored in Postgres. Only a user's free-text CV skills or a new job title need live embedding.

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

1. Import this repository via **Create Repl → Import from GitHub**.
2. In the **Secrets** tab (🔒), add the environment variables above.
3. Set the run command to `uvicorn src.backend.main:app --host 0.0.0.0 --port 8000` and click **Run**.

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

# 3. Start the API (docs at http://localhost:8000/docs)
python -m src.backend.main
```

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

### Evaluate the embedding model

```bash
python -m embeddings.eval.evaluate --compare qwen3-embedding:0.6b   # metrics vs baselines
python -m embeddings.eval.evaluate --record                         # gate, then log as active version
```

A new model, vector size or instruction text must score at least as well as the active version on the frozen gold set before `--record` accepts it. `build_index` refuses to replace the live index with settings that haven't passed this gate.

---

## How We Use Replit *(Required)*

Describe how your team used Replit. For example:

- **Development:** Collaborative coding in a shared Repl using Multiplayer.
- **Hosting:** The app is deployed with Replit Deployments at the Live Demo link above.
- **Secrets:** API keys are stored in Replit Secrets, never in the code.

📄 Proof of use: [`submission/SDC2026_TeamName_Replit_POU.pdf`](submission/SDC2026_TeamName_Replit_POU.pdf)

## How We Use RunPod *(Required)*

Describe how your team used RunPod. For example:

- **Workload:** Model inference (e.g. text generation, image processing) runs on a RunPod GPU.
- **Setup:** `runpod/handler.py` is deployed as a **RunPod Serverless Endpoint**.
- **Integration:** The Replit app sends requests to the endpoint using `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID`.

Example request:

```bash
curl -X POST https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "Hello"}}'
```

📄 Proof of use: [`submission/SDC2026_TeamName_Runpod_POU.pdf`](submission/SDC2026_TeamName_Runpod_POU.pdf)

---

## Challenges & Learnings

- Development exposed a serious problem in building such a system from the scratch that involves several sources to scan. Moreover, lack of current datasets that have a data from specified range of date
- Team acquired fundamental knowledge of machine learning concepts, especially embedding systems and forecasting models. Team mastered tech stack such as. Moreover, team gained a wide representation of agentic development involving Replit, Claude Code and other tools. 

## Future Improvements

- [ ] Ingest real job postings (`src/sources/`, `src/pipeline/agent_loop.py`) and map them through `normalize.py`
- [ ] Train the LightGBM forecaster and write its output to `entity_forecast`
- [ ] Build the frontend and the forecast / matching API endpoints
- [ ] Grow the taxonomy from ESCO / O*NET and verify the MASCO codes against DOSM's MASCO 2020
- [ ] Add real posting mentions to the gold set, then re-record the thresholds
- [ ] Add a small generative model for skill extraction and plain-language explanations
- [ ] Move the backend database to Postgres and run the version health check at startup
- [ ] Host live embedding on a protected GPU endpoint, and restrict CORS to the app's origin

## Team

| Name        | Role                | GitHub                                   |
| ----------- | ------------------- | ---------------------------------------- |
| Nurali Zhan | Team Lead / Backend | [@username](https://github.com/username) |
| Member Name | Frontend            | [@username](https://github.com/username) |
| Askaruly Ali| AI / RunPod         | [@invalid-RJK](https://github.com/username) |
| Member Name | AI / RunPod         | [@username](https://github.com/username) |

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
