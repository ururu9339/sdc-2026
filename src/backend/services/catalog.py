from __future__ import annotations

from src.sources import taxonomy as T


def list_taxonomy(kind: str, q: str | None = None, offset: int = 0, limit: int = 500) -> dict:
    tax = T.load_taxonomy()
    rows = tax[tax["kind"] == kind]
    needle = (q or "").strip().lower()
    entries = []
    for r in rows.itertuples(index=False):
        alts = [a.strip() for a in str(r.alt_labels).split("|") if a.strip()]
        if needle and needle not in str(r.id).lower() and needle not in str(r.name).lower() \
                and not any(needle in a.lower() for a in alts):
            continue
        entries.append({"id": str(r.id), "name": str(r.name), "alt_labels": alts})
    entries.sort(key=lambda e: (e["name"].lower(), e["id"]))
    page = entries[offset: offset + limit]
    return {"kind": kind, "count": len(page), "total": len(entries), "offset": offset, "limit": limit,
            "items": page}
