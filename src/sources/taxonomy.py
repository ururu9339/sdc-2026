"""Shared vocabulary so every source lines up on the same entity ids.

    family_of(titles)   -> role family id per job title / role string (NaN if unmatched)
    skill_flags(text)   -> one bool column per canonical skill (keyword extractor)

Every adapter imports from here, so there is one implementation of each. Families are matched on
normalised text (lowercase, punctuation to spaces) and the FIRST matching family wins, so order
matters: specific families come before broad ones. Unmatched titles stay NaN; there is no 'other'.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

LEXICON_PATH = Path(__file__).with_name("skills_lexicon.json")

ROLE_PATH = Path(__file__).with_name("role_families.json")


def load_role_families(path: Path = ROLE_PATH) -> dict[str, list[str]]:
    """canonical family id -> regex patterns over normalised text, in priority order."""
    return {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}


ROLE_FAMILIES: dict[str, list[str]] = load_role_families()

_NON_ALNUM = r"[^a-z0-9&]+"


def _classify(values: pd.Series) -> pd.Series:
    norm = values.str.lower().str.replace(_NON_ALNUM, " ", regex=True).str.strip()
    fam = pd.Series(np.nan, index=values.index, dtype=object)
    for name, pats in ROLE_FAMILIES.items():
        pat = re.compile("|".join(f"(?:{p})" for p in pats))
        hit = norm.map(lambda t: bool(pat.search(t))) & fam.isna()
        fam = fam.where(~hit, name)
    return fam


def family_of(series: pd.Series) -> pd.Series:
    """Vectorised, case-insensitive, first-match-wins family id. Unmatched -> NaN.

    Classifies the distinct values only, then maps back, so large columns stay fast.
    """
    s = series.fillna("").astype(str)
    uniq = pd.Series(s.unique(), dtype=object)
    mapping = dict(zip(uniq, _classify(uniq)))
    return s.map(mapping).astype(object)


def load_lexicon(path: Path = LEXICON_PATH) -> dict[str, list[str]]:
    return {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}


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
            out.append(nxt)  # escaped punctuation such as \. is a literal
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


def skill_flags(text: pd.Series, lexicon: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Boolean frame, one column per canonical skill. Keyword extractor, to be augmented by Qwen.

    Works on the distinct strings only and maps the result back to rows. Each alias is first checked
    for its required literal prefix (a plain substring test), and the regex runs only on those texts.
    The result is identical to running every regex on every text."""
    lexicon = lexicon if lexicon is not None else load_lexicon()
    codes, uniques = pd.factorize(text.fillna("").astype(str))
    low = [u.lower() for u in uniques]
    out = {}
    for skill, aliases in lexicon.items():
        hit = np.zeros(len(low), dtype=bool)
        for alias in aliases:
            pat = re.compile(f"(?<![a-z0-9])(?:{alias})(?![a-z0-9])")
            lit = _required_literal(alias)
            cand = [i for i, t in enumerate(low) if not hit[i] and (not lit or lit in t)]
            for i in cand:
                if pat.search(low[i]):
                    hit[i] = True
        out[skill] = hit
    per_value = pd.DataFrame(out, dtype=bool, index=pd.RangeIndex(len(low)))
    flags = per_value.iloc[codes].reset_index(drop=True) if len(codes) else per_value.iloc[0:0].reset_index(drop=True)
    flags.index = text.index
    return flags


def vocabulary_errors(family_ids, skill_ids) -> list[str]:
    """Family ids and skill ids share one key space (entity_id), so they may not collide."""
    return [f"skill id {s!r} collides with job family id {s!r}; rename the skill (e.g. '{s}_skill')"
            for s in sorted(set(family_ids) & set(skill_ids))]

