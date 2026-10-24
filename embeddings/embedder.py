import csv
import hashlib
import os
import sqlite3
from pathlib import Path

import numpy as np
import requests

MODEL = "qwen3-embedding:4b"
DIM = 1024
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/embed"
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")  # for a remote Ollama behind a key-checking reverse proxy
TASK = "Given a skill or job phrase from a job posting or CV, retrieve the matching standardized skill or occupation"

HERE = Path(__file__).parent
TAXONOMY_DIR = HERE / "taxonomy"
CACHE_PATH = HERE / ".cache" / "embeddings.sqlite3"


def version_tag():
    return f"{MODEL}|dim={DIM}|task={hashlib.sha1(TASK.encode()).hexdigest()[:8]}"

def _cache():
    CACHE_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(CACHE_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")
    return con

def _key(text):
    return hashlib.sha1(f"{version_tag()}\n{text}".encode()).hexdigest()

def auth_headers():
    return {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}

def _request(texts):
    r = requests.post(OLLAMA, json={"model": MODEL, "input": texts}, headers=auth_headers(), timeout=300)
    r.raise_for_status()
    v = np.array(r.json()["embeddings"], dtype=np.float32)[:, :DIM]
    return v / np.linalg.norm(v, axis=1, keepdims=True)                

def _embed(texts, batch=32, use_cache=True):
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    if not use_cache:
        return np.vstack([_request(texts[i:i + batch]) for i in range(0, len(texts), batch)])

    con = _cache()
    keys = [_key(t) for t in texts]
    found = {}
    for i in range(0, len(keys), 500):
        chunk = keys[i:i + 500]
        rows = con.execute(f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(chunk))})", chunk)
        found.update((k, np.frombuffer(b, dtype=np.float32)) for k, b in rows)

    missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in found))
    for i in range(0, len(missing), batch):
        part = missing[i:i + batch]
        vecs = _request(part)
        rows = [(_key(t), v.tobytes()) for t, v in zip(part, vecs)]
        con.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
        con.commit()
        found.update((k, np.frombuffer(b, dtype=np.float32)) for k, b in rows)
    con.close()
    return np.vstack([found[k] for k in keys])

def embed_documents(texts, **kw):
    """Taxonomy entries: no instruction."""
    return _embed(list(texts), **kw)


def embed_queries(texts, **kw):
    """Raw mentions, CV lines, new job titles: instruction-prefixed, as Qwen3-Embedding expects."""
    return _embed([f"Instruct: {TASK}\nQuery: {t}" for t in texts], **kw)

def load_taxonomy(kind=None):
    """Rows from taxonomy/skills.csv and/or taxonomy/jobs.csv. kind: 'skill', 'job' or None for both."""
    files = {"skill": ["skills.csv"], "job": ["jobs.csv"], None: ["skills.csv", "jobs.csv"]}[kind]
    rows = []
    for f in files:
        with open(TAXONOMY_DIR / f, newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows

def entity_text(row):
    """name + alt labels + one-line description; embeds better than the name alone."""
    alts = row["alt_labels"].replace("|", ", ")
    return f"{row['name']} ({alts}). {row['description']}" if alts else f"{row['name']}. {row['description']}"
