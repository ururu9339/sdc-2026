"""Taxonomy mechanics with hand-written strings (fixtures, not datasets)."""
import json
import re

import numpy as np
import pandas as pd

from src.sources import taxonomy as T

# the 12 job_market occupations must each land in a family, and the families must be distinct
JM_OCCUPATIONS = {
    "UX Designer": "ux_designer", "Product Manager": "product_manager", "Financial Analyst": "financial_analyst",
    "Data Analyst": "data_analyst", "AI Engineer": "ai_engineer", "Cloud Engineer": "cloud_engineer",
    "Software Engineer": "software_engineer", "Operations Manager": "operations_manager", "HR Analyst": "hr_specialist",
    "Marketing Specialist": "marketing", "Data Scientist": "data_scientist", "Business Analyst": "business_analyst",
}


def test_every_jm_occupation_maps_to_its_family():
    got = T.family_of(pd.Series(list(JM_OCCUPATIONS)))
    assert got.tolist() == list(JM_OCCUPATIONS.values())
    assert len(set(JM_OCCUPATIONS.values())) == 12


def test_role_families_load_from_json_in_priority_order():
    raw = json.loads(T.ROLE_PATH.read_text())
    assert list(T.ROLE_FAMILIES) == [k for k in raw if not k.startswith("_")]
    assert list(T.ROLE_FAMILIES)[:2] == ["ux_designer", "product_manager"]
    assert len(T.ROLE_FAMILIES) == 37


def test_case_insensitive_and_first_match_wins():
    got = T.family_of(pd.Series(["SOFTWARE ARCHITECT", "marketing analytics specialist", "Account-Executive"]))
    # 'Software Architect': software listed before architecture; marketing before data analytics
    assert got.tolist() == ["software_engineer", "marketing", "sales"]


def test_unmatched_is_nan_not_other():
    got = T.family_of(pd.Series(["Management Trainee", None, "Paralegal"]))
    assert pd.isna(got.iloc[0]) and pd.isna(got.iloc[1]) and got.iloc[2] == "legal"


def test_family_patterns_compile_and_ids_are_snake_case():
    for name, pats in T.ROLE_FAMILIES.items():
        assert re.fullmatch(r"[a-z][a-z0-9_]*", name), name
        re.compile("|".join(pats))


def test_lexicon_ids_are_snake_case_and_aliases_nonempty():
    def no_dupes(pairs):
        keys = [k for k, _ in pairs]
        assert len(keys) == len(set(keys)), "duplicate skill id in lexicon"
        return dict(pairs)

    json.loads(T.LEXICON_PATH.read_text(), object_pairs_hook=no_dupes)
    lex = T.load_lexicon()
    assert "_note" not in lex and len(lex) >= 60
    for skill, aliases in lex.items():
        assert re.fullmatch(r"[a-z][a-z0-9_]*", skill), skill
        assert aliases


def test_family_and_skill_ids_never_collide():
    # entity_id is the key across kinds, so a skill and a job family may not share an id
    assert not set(T.ROLE_FAMILIES) & set(T.load_lexicon())
    assert T.vocabulary_errors(T.ROLE_FAMILIES, T.load_lexicon()) == []


def test_vocabulary_collision_is_an_error():
    errs = T.vocabulary_errors(["sales", "teacher"], ["sales", "python"])
    assert len(errs) == 1 and "sales" in errs[0] and "_skill" in errs[0]


def test_skill_flags_respect_word_boundaries():
    text = pd.Series(["JavaScript and MySQL", "Java developer", "c# and .NET", None])
    flags = T.skill_flags(text, {"java": ["java"], "javascript": ["javascript"], "sql": ["sql", "mysql"],
                                 "c_sharp_dotnet": ["c#", "\\.net"]})
    assert flags["java"].tolist() == [False, True, False, False]
    assert flags["javascript"].tolist() == [True, False, False, False]
    assert flags["sql"].tolist() == [True, False, False, False]
    assert flags["c_sharp_dotnet"].tolist() == [False, False, True, False]
    assert flags.dtypes.unique().tolist() == [np.dtype(bool)]
    assert flags.index.equals(text.index)


def test_skill_flags_map_repeated_values_back_to_rows():
    text = pd.Series(["python sql", "excel", "python sql", None, "excel"], index=[10, 11, 12, 13, 14])
    flags = T.skill_flags(text, {"python": ["python"], "excel": ["excel"]})
    assert flags["python"].tolist() == [True, False, True, False, False]
    assert flags["excel"].tolist() == [False, True, False, False, True]
    assert flags.index.tolist() == [10, 11, 12, 13, 14]
