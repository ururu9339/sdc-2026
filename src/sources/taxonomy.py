"""The shared vocabulary: the teammate taxonomy (embeddings/taxonomy/{skills,jobs}.csv) is the only one.

    load_vocabulary()              -> Vocabulary(entities, skill_aliases, job_aliases)
    skill_flags(text, aliases)     -> one bool column per skill id (any alias matches)
    job_of(titles, aliases)        -> one job id per value, NaN where no job alias matches

Entity ids are the taxonomy ids verbatim ('skill:python', 'job:software_engineer'); the prefix gives the kind.
Aliases per entity are its name and alt_labels (literal, case-insensitive), plus the regex extras in
src/sources/vocab_extra.json (migrated from the retired legacy lists; see that file's _note).

Matching rules:
  - Every alias is matched with alphanumeric boundaries, so 'sql' does not match inside 'mysql'.
  - Skills: a posting can carry several skills; any alias match counts.
  - Jobs: one job per value. The job whose matching alias is most specific wins (longest alias, after
    removing regex backslashes); ties go to the earlier row of jobs.csv. A flat taxonomy has no priority
    order, and a generic extra such as 'developer' must not override the title 'web developer'.
Unmatched values stay NaN (jobs) or False (skills). There is no 'other' bucket.
"""
from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.training import config as C

EXTRA_PATH = Path(__file__).with_name("vocab_extra.json")
TAXONOMY_FILES = {"skill": "skills.csv", "job": "jobs.csv"}
ID_RE = re.compile(r"^(skill|job):[a-z][a-z0-9_]*$")
_NON_ALNUM = r"[^a-z0-9&]+"  # job text is normalised the way the legacy family matcher normalised it
_BOUNDARY = "(?<![a-z0-9])(?:{})(?![a-z0-9])"


@dataclass(frozen=True)
class Vocabulary:
    entities: pd.DataFrame        # columns: id, name, kind
    skill_aliases: dict           # entity id -> list of regex aliases (skill text is lowercased)
    job_aliases: dict             # entity id -> list of regex aliases (job text is normalised)

    @property
    def names(self) -> dict:
        return dict(zip(self.entities["id"], self.entities["name"]))

    def ids(self, kind: str) -> set:
        return set(self.entities.loc[self.entities["kind"] == kind, "id"])


def _normalise(text: str) -> str:
    return re.sub(_NON_ALNUM, " ", text.lower()).strip()


def load_taxonomy(path: Path | None = None) -> pd.DataFrame:
    """Both taxonomy CSVs stacked, with kind derived from the id prefix. Raises ValueError on any problem."""
    root = Path(path if path is not None else C.TAXONOMY_DIR)
    frames = []
    for kind, fname in TAXONOMY_FILES.items():
        df = pd.read_csv(root / fname, dtype=str, keep_default_na=False)
        missing = {"id", "name", "type", "alt_labels"} - set(df.columns)
        if missing:
            raise ValueError(f"{fname}: missing columns {sorted(missing)}")
        bad = df.loc[~df["id"].str.match(ID_RE), "id"].tolist()
        if bad:
            raise ValueError(f"{fname}: ids must look like '{kind}:snake_case', got {bad[:5]}")
        if (df["type"] != kind).any():
            raise ValueError(f"{fname}: every row must have type '{kind}'")
        if (df["name"].str.strip() == "").any():
            raise ValueError(f"{fname}: every entity needs a name")
        frames.append(df.assign(kind=kind))
    tax = pd.concat(frames, ignore_index=True)
    dup = tax.loc[tax["id"].duplicated(), "id"].tolist()
    if dup:
        raise ValueError(f"duplicate taxonomy ids: {dup[:5]}")
    return tax


def load_extra(path: Path | None = None) -> dict:
    raw = json.loads((Path(path) if path is not None else EXTRA_PATH).read_text(encoding="utf-8"))
    return {k: list(v) for k, v in raw.items() if not k.startswith("_")}


def _dedupe(seq):
    return list(dict.fromkeys(s for s in seq if s))


def build_vocabulary(tax: pd.DataFrame, extra: dict) -> Vocabulary:
    unknown = sorted(set(extra) - set(tax["id"]))
    if unknown:
        raise ValueError(f"vocab_extra.json refers to ids not in the taxonomy: {unknown[:5]}")
    skill_aliases, job_aliases = {}, {}
    for r in tax.itertuples(index=False):
        names = [r.name] + [a.strip() for a in r.alt_labels.split("|") if a.strip()]
        if r.kind == "skill":
            literal = [re.escape(n.lower()) for n in names]
        else:
            literal = [re.escape(_normalise(n)) for n in names]
        target = skill_aliases if r.kind == "skill" else job_aliases
        target[r.id] = _dedupe(literal + list(extra.get(r.id, [])))
    entities = tax[["id", "name", "kind"]].reset_index(drop=True)
    return Vocabulary(entities=entities, skill_aliases=skill_aliases, job_aliases=job_aliases)


@functools.lru_cache(maxsize=4)
def _cached(taxonomy_dir: str, extra_path: str) -> Vocabulary:
    return build_vocabulary(load_taxonomy(Path(taxonomy_dir)), load_extra(Path(extra_path)))


def load_vocabulary(path: Path | None = None, extra_path: Path | None = None) -> Vocabulary:
    """The vocabulary for C.TAXONOMY_DIR (read at call time) and vocab_extra.json."""
    tdir = str(Path(path if path is not None else C.TAXONOMY_DIR))
    epath = str(Path(extra_path if extra_path is not None else EXTRA_PATH))
    return _cached(tdir, epath)


def entity_name(entity_id: str) -> str:
    """Taxonomy name for an id; falls back to the id's own words (for ids outside the taxonomy)."""
    try:
        return load_vocabulary().names.get(entity_id) or entity_id.split(":", 1)[-1].replace("_", " ")
    except (OSError, ValueError):
        return entity_id.split(":", 1)[-1].replace("_", " ")


def _required_literal(alias: str) -> str:
    """Longest literal prefix that every match of the regex `alias` must start with ('' if none)."""
    out, i = [], 0
    while i < len(alias):
        c = alias[i]
        if c == "\\":
            nxt = alias[i + 1] if i + 1 < len(alias) else ""
            if nxt in ("b", "B"):  # zero-width assertion: skip
                i += 2
                continue
            if nxt == "" or nxt.isalnum():  # class or escape such as \d or \s: stop
                break
            out.append(nxt)  # escaped punctuation such as \. or \# is a literal
            i += 2
            continue
        if c in "?*+{":  # quantifier: applies to the character before it, which is therefore optional
            if out:
                out.pop()
            break
        if c in "[](|)^$.":
            break
        out.append(c)
        i += 1
    return "".join(out).lower()


def _specificity(alias: str) -> int:
    return len(alias.replace("\\", ""))


def _compile(alias: str) -> re.Pattern:
    return re.compile(_BOUNDARY.format(alias))


def skill_flags(text: pd.Series, aliases: dict) -> pd.DataFrame:
    """Boolean frame, one column per skill id. Works on the distinct strings and maps the result back to rows.

    Each alias is first checked for its required literal prefix (a plain substring test), and the regex runs
    only on the texts that pass. The result equals running every regex on every text."""
    codes, uniques = pd.factorize(text.fillna("").astype(str))
    low = [u.lower() for u in uniques]
    out = {}
    for ent, alias_list in aliases.items():
        hit = np.zeros(len(low), dtype=bool)
        for alias in alias_list:
            pat = _compile(alias)
            lit = _required_literal(alias)
            cand = [i for i, t in enumerate(low) if not hit[i] and (not lit or lit in t)]
            for i in cand:
                if pat.search(low[i]):
                    hit[i] = True
        out[ent] = hit
    per_value = pd.DataFrame(out, dtype=bool, index=pd.RangeIndex(len(low)))
    flags = per_value.iloc[codes].reset_index(drop=True) if len(codes) else per_value.iloc[0:0].reset_index(drop=True)
    flags.index = text.index
    return flags


def job_of(values: pd.Series, aliases: dict) -> pd.Series:
    """One job id per value: the job with the most specific matching alias, ties to the earlier job.

    Works on the distinct normalised strings and maps back. Unmatched values are NaN."""
    s = values.fillna("").astype(str)
    uniq = pd.Series(s.unique(), dtype=object)
    norm = [_normalise(u) for u in uniq]
    best_job = [None] * len(norm)
    best_spec = [0] * len(norm)
    for ent, alias_list in aliases.items():
        for alias in alias_list:
            spec = _specificity(alias)
            pat = _compile(alias)
            lit = _required_literal(alias)
            for i, t in enumerate(norm):
                if spec > best_spec[i] and (not lit or lit in t) and pat.search(t):
                    best_job[i] = ent
                    best_spec[i] = spec
    mapping = {u: (j if j is not None else np.nan) for u, j in zip(uniq, best_job)}
    return s.map(mapping).astype(object)
