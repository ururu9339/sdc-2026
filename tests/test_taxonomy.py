"""Shared vocabulary from the teammate taxonomy: loading, aliases and matching rules.

Tiny hand-written taxonomies live in tmp_path. The real taxonomy is checked only for its structure and for
the market occupations the adapters depend on."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sources import taxonomy as T
from src.sources.validate import validate
from src.training import config as C

ID_OK = r"^(skill|job):[a-z][a-z0-9_]*$"
SKILL_HEADER = "id,name,type,alt_labels,description\n"
JOB_HEADER = "id,name,type,masco_code,alt_labels,description\n"


def _tax_dir(root: Path, skills: list[str], jobs: list[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "skills.csv").write_text(SKILL_HEADER + "\n".join(skills) + "\n", encoding="utf-8")
    (root / "jobs.csv").write_text(JOB_HEADER + "\n".join(jobs) + "\n", encoding="utf-8")
    return root


def _skill(id_, name, alts=""):
    return f'{id_},"{name}",skill,"{alts}",desc'


def _job(id_, name, alts=""):
    return f'{id_},"{name}",job,1234,"{alts}",desc'


# ---- the real taxonomy --------------------------------------------------------------------
def test_real_taxonomy_loads_with_kind_from_the_prefix():
    v = T.load_vocabulary()
    assert len(v.ids("skill")) == 108 and len(v.ids("job")) == 56
    ids = v.entities["id"]
    assert ids.str.match(ID_OK).all()
    assert set(v.entities["kind"]) == {"skill", "job"}
    assert v.names["skill:python"] == "Python" and v.names["job:accountant"] == "Accountant"


def test_real_extras_are_migrated_only_for_taxonomy_ids():
    extra = T.load_extra()
    assert set(extra) <= T.load_vocabulary().ids("skill") | T.load_vocabulary().ids("job")
    # 28 legacy skills and 16 legacy families had a taxonomy counterpart; the rest were dropped
    assert sum(k.startswith("skill:") for k in extra) == 28 and sum(k.startswith("job:") for k in extra) == 16


def test_legacy_vocabulary_files_are_gone():
    folder = Path(T.__file__).parent
    assert not (folder / "skills_lexicon.json").exists()
    assert not (folder / "role_families.json").exists()
    assert not hasattr(T, "family_of") and not hasattr(T, "load_lexicon")


def test_market_occupations_under_the_taxonomy():
    occ = ["UX Designer", "Product Manager", "Financial Analyst", "Data Analyst", "AI Engineer", "Cloud Engineer",
           "Software Engineer", "Operations Manager", "HR Analyst", "Marketing Specialist", "Data Scientist",
           "Business Analyst"]
    got = T.job_of(pd.Series(occ), T.load_vocabulary().job_aliases).tolist()
    assert got[:4] == ["job:ux_designer", "job:product_manager", "job:financial_analyst", "job:data_analyst"]
    assert got[4:7] == ["job:ml_engineer", "job:devops_engineer", "job:software_engineer"]
    assert got[10:] == ["job:data_scientist", "job:business_analyst"]
    # the three legacy families with no taxonomy counterpart stay unmatched, and are reported by the adapter
    assert all(pd.isna(x) for x in got[7:10])


# ---- loading and validation ---------------------------------------------------------------
def test_duplicate_id_is_rejected(tmp_path):
    d = _tax_dir(tmp_path / "t", [_skill("skill:python", "Python")], [_job("job:x", "X"), _job("job:x", "X2")])
    with pytest.raises(ValueError, match="duplicate taxonomy ids"):
        T.load_taxonomy(d)


def test_ids_need_the_kind_prefix(tmp_path):
    d = _tax_dir(tmp_path / "t", [_skill("python", "Python")], [_job("job:x", "X")])
    with pytest.raises(ValueError, match="skill:snake_case"):
        T.load_taxonomy(d)


def test_type_column_must_match_the_file(tmp_path):
    d = _tax_dir(tmp_path / "t", ['skill:python,"Python",job,"",desc'], [_job("job:x", "X")])
    with pytest.raises(ValueError, match="type 'skill'"):
        T.load_taxonomy(d)


def test_extra_aliases_must_refer_to_taxonomy_ids(tmp_path):
    tax = T.load_taxonomy(_tax_dir(tmp_path / "t", [_skill("skill:python", "Python")], [_job("job:x", "X")]))
    with pytest.raises(ValueError, match="not in the taxonomy"):
        T.build_vocabulary(tax, {"skill:nope": ["nope"]})


def test_taxonomy_dir_is_read_from_config(tmp_path, monkeypatch):
    d = _tax_dir(tmp_path / "t", [_skill("skill:go", "Golang")], [_job("job:x", "Clerk", "office clerk")])
    no_extras = tmp_path / "extra.json"
    no_extras.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(C, "TAXONOMY_DIR", d)
    v = T.load_vocabulary(extra_path=no_extras)
    assert v.ids("skill") == {"skill:go"} and v.job_aliases["job:x"][1] == "office\\ clerk"


def test_validator_flags_ids_outside_the_taxonomy():
    df = pd.DataFrame({"entity_id": ["skill:python", "skill:madeup"], "kind": ["skill", "skill"],
                       "period": ["2024-01-01", "2024-01-01"], "source_id": ["p", "p"], "value": [0.1, 0.2],
                       "support": [5, 5]})
    res = validate(df, {"source_id": "p", "granularity": "weekly"}, vocabulary_ids={"skill:python"})
    assert not res["ok"] and any("not in the shared taxonomy" in e for e in res["errors"])
    assert validate(df, {"source_id": "p", "granularity": "weekly"})["ok"]  # no vocabulary given: no check


# ---- aliases and matching ------------------------------------------------------------------
@pytest.fixture
def tiny(tmp_path, monkeypatch):
    d = _tax_dir(tmp_path / "t",
                 [_skill("skill:sql", "SQL", "structured query language"), _skill("skill:csharp", "C#", ""),
                  _skill("skill:python", "Python", "python3|python programming"),
                  _skill("skill:go", "Go", "")],
                 [_job("job:web_developer", "Web Developer", "frontend developer"),
                  _job("job:generic_dev", "Developer", ""),
                  _job("job:full_stack", "Full-Stack Engineer", ""),
                  _job("job:twin_a", "Analyst", ""), _job("job:twin_b", "Analyst", "")])
    monkeypatch.setattr(C, "TAXONOMY_DIR", d)
    return T.build_vocabulary(T.load_taxonomy(d), {})


def test_skill_names_and_alt_labels_match_with_word_boundaries(tiny):
    text = pd.Series(["SQL and MySQL", "structured query language", "python3 and Python", "C# and .NET", None])
    flags = T.skill_flags(text, tiny.skill_aliases)
    assert flags["skill:sql"].tolist() == [True, True, False, False, False]
    assert flags["skill:python"].tolist() == [False, False, True, False, False]
    assert flags["skill:csharp"].tolist() == [False, False, False, True, False]  # punctuation is literal
    assert flags.dtypes.unique().tolist() == [np.dtype(bool)]
    assert flags.index.equals(text.index)


def test_skill_flags_map_repeated_values_back_to_rows(tiny):
    text = pd.Series(["python sql", "go", "python sql", None, "go"], index=[10, 11, 12, 13, 14])
    flags = T.skill_flags(text, tiny.skill_aliases)
    assert flags["skill:python"].tolist() == [True, False, True, False, False]
    assert flags["skill:go"].tolist() == [False, True, False, False, True]
    assert flags.index.tolist() == [10, 11, 12, 13, 14]


def test_jobs_are_matched_on_normalised_text(tiny):
    got = T.job_of(pd.Series(["Full-stack engineer", "FULL STACK ENGINEER", "Nobody Here", None]), tiny.job_aliases)
    assert got.tolist()[:2] == ["job:full_stack", "job:full_stack"]
    assert pd.isna(got.iloc[2]) and pd.isna(got.iloc[3])


def test_most_specific_job_alias_wins_not_file_order(tiny):
    # 'Developer' is listed before 'Web Developer' in jobs.csv; the longer alias must still win
    got = T.job_of(pd.Series(["Web Developer", "frontend developer", "Developer", "Senior Developer"]), tiny.job_aliases)
    assert got.tolist() == ["job:web_developer", "job:web_developer", "job:generic_dev", "job:generic_dev"]


def test_equal_specificity_goes_to_the_earlier_job(tiny):
    assert T.job_of(pd.Series(["Analyst"]), tiny.job_aliases).tolist() == ["job:twin_a"]


def test_fallback_pattern_gives_the_second_column_a_chance(tiny):
    role = T.job_of(pd.Series(["Random"]), tiny.job_aliases).fillna(
        T.job_of(pd.Series(["Web Developer"]), tiny.job_aliases))
    assert role.tolist() == ["job:web_developer"]


def test_entity_name_falls_back_to_the_id_words(monkeypatch):
    monkeypatch.setattr(C, "TAXONOMY_DIR", Path("/definitely/not/here"))
    assert T.entity_name("skill:python") == "python"
    monkeypatch.undo()
    assert T.entity_name("skill:python") == "Python"
    assert T.entity_name("job:not_in_taxonomy") == "not in taxonomy"
