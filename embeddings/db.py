import os
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import psycopg
import requests

from embeddings import embedder

SCHEMA = Path(__file__).parent / "schema.sql"


class VersionMismatch(RuntimeError):
    pass


def connect(url=None):
    url = url or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("Set DATABASE_URL to a Postgres database with pgvector")
    return psycopg.connect(url)


def init_schema(conn):
    conn.execute(SCHEMA.read_text())
    conn.commit()


def to_pgvector(v):
    return "[" + ",".join(f"{x:.7g}" for x in np.asarray(v, dtype=np.float32)) + "]"


def get_meta(conn, key):
    row = conn.execute("SELECT value FROM embedding_meta WHERE key = %s", (key,)).fetchone()
    return row[0] if row else None


def set_meta(conn, key, value):
    conn.execute(
        "INSERT INTO embedding_meta (key, value) VALUES (%s, %s) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        (key, value),
    )


def ollama_model_digest():
    base = embedder.OLLAMA.rsplit("/api/", 1)[0]
    r = requests.get(f"{base}/api/tags", headers=embedder.auth_headers(), timeout=10)
    r.raise_for_status()
    for m in r.json()["models"]:
        if m["name"] == embedder.MODEL:
            return m["digest"]
    raise RuntimeError(f"{embedder.MODEL} is not pulled on {base}")


def check_version(conn):
    stored = get_meta(conn, "model_version")
    if stored is None:
        raise VersionMismatch("No taxonomy index in the database; run python -m embeddings.build_index")
    if stored != embedder.version_tag():
        raise VersionMismatch(f"Database vectors are {stored!r} but this server uses {embedder.version_tag()!r}")
    mixed = conn.execute(
        "SELECT count(*) FROM entity_embedding WHERE model_version <> %s", (stored,)
    ).fetchone()[0]
    if mixed:
        raise VersionMismatch(f"{mixed} entity vectors were built with a different version; rebuild the index")
    stored_digest = get_meta(conn, "model_digest")
    live_digest = ollama_model_digest()
    if stored_digest and stored_digest != live_digest:
        raise VersionMismatch(f"Ollama model digest {live_digest[:12]} differs from index digest {stored_digest[:12]}")
    return stored


def check_ollama_exposure():
    warnings = []
    u = urlparse(embedder.OLLAMA)
    local = u.hostname in ("localhost", "127.0.0.1", "::1")
    if not local and not embedder.OLLAMA_API_KEY:
        warnings.append(f"OLLAMA_URL points at {u.hostname} with no OLLAMA_API_KEY; Ollama has no auth of its own")
    if not local and u.scheme != "https":
        warnings.append(f"OLLAMA_URL uses {u.scheme}:// to a remote host; queries and any API key travel in clear text")
    return warnings


if __name__ == "__main__":
    with connect() as conn:
        print("ok:", check_version(conn))
    for w in check_ollama_exposure():
        print("WARNING:", w)
