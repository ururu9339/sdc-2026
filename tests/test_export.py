import csv
import math
from datetime import datetime, timezone

import pytest

from src.pipeline import export_forecasts as E
from src.pipeline.store import AppendOnlyLog
from src.training import config as C

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
VALID = {"skill:python", "job:accountant"}


def _rec(entity, as_of="2026-10-18", p50=0.1, status="ok", version="v1", horizon=12, **kw):
    return {"entity_id": entity, "as_of": as_of, "p50": p50, "status": status, "model_version": version,
            "horizon_periods": horizon, **kw}


def test_horizon_is_converted_from_periods_to_months():
    assert E.horizon_months(12, 7.0) == 3
    assert E.horizon_months(26, 7.0) == 6
    assert E.horizon_months(3, 30.44) == 3


def test_growth_is_relative_change_of_the_smoothed_share():
    rows, counts = E.rows_from_records([_rec("skill:python", p50=0.12)], "live", VALID, NOW, period_days=7.0)
    assert len(rows) == 1 and counts["entities_exported"] == 1
    r = rows[0]
    assert r["entity_id"] == "skill:python" and r["horizon_months"] == 3
    assert r["growth"] == pytest.approx(math.exp(0.12) - 1) and r["growth"] == pytest.approx(0.1275, abs=1e-4)
    assert r["created_at"] == NOW


def test_model_version_is_bare_for_live_and_prefixed_for_preview():
    live, _ = E.rows_from_records([_rec("skill:python", version="v3")], "live", VALID, NOW)
    prev, _ = E.rows_from_records([_rec("skill:python", version="preview:v3")], "preview", VALID, NOW)
    assert live[0]["model_version"] == "v3"
    assert prev[0]["model_version"] == "preview:v3"


def test_newest_ok_record_per_entity_and_skips_are_counted():
    records = [
        _rec("skill:python", as_of="2026-10-04", p50=0.01),
        _rec("skill:python", as_of="2026-10-11", p50=0.02),
        _rec("job:accountant", as_of="2026-10-18", status="insufficient_data", p50=None),
        _rec("skill:not_in_taxonomy", as_of="2026-10-18", p50=0.3),
    ]
    rows, counts = E.rows_from_records(records, "live", VALID, NOW)
    assert [r["entity_id"] for r in rows] == ["skill:python"]
    assert rows[0]["growth"] == pytest.approx(math.exp(0.02) - 1)
    assert counts == {"records": 4, "not_ok": 1, "not_in_taxonomy": 1, "entities_exported": 1, "from_older_run": 0}


def test_entities_from_an_older_run_are_counted():
    records = [_rec("skill:python", as_of="2026-10-04"), _rec("job:accountant", as_of="2026-10-18")]
    rows, counts = E.rows_from_records(records, "live", VALID, NOW)
    assert len(rows) == 2 and counts["from_older_run"] == 1


class FakeCursor:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def executemany(self, sql, params):
        self.log.append(("executemany", sql, list(params)))


class FakeConn:
    def __init__(self):
        self.log = []

    def cursor(self):
        return FakeCursor(self.log)

    def commit(self):
        self.log.append(("commit",))


def test_upsert_sql_and_parameters_on_a_fake_connection():
    rows, _ = E.rows_from_records([_rec("skill:python", p50=0.1), _rec("job:accountant", p50=-0.2)], "live",
                                  VALID, NOW)
    conn = FakeConn()
    assert E.write_rows(conn, rows) == 2
    (_, sql, params), commit = conn.log
    assert "INSERT INTO entity_forecast (entity_id, horizon_months, growth, model_version, created_at)" in sql
    assert "ON CONFLICT (entity_id, horizon_months, model_version)" in sql
    assert "DO UPDATE SET growth = EXCLUDED.growth, created_at = EXCLUDED.created_at" in sql
    by_entity = {p[0]: p for p in params}
    assert [p[0] for p in params] == ["job:accountant", "skill:python"]
    assert by_entity["skill:python"] == ("skill:python", 3, pytest.approx(math.exp(0.1) - 1), "v1", NOW)
    assert commit == ("commit",)


def _point_logs(monkeypatch, tmp_path, live: bool):
    monkeypatch.setattr(C, "PREVIEW_LOG_PATH", tmp_path / "preview_log.jsonl")
    monkeypatch.setattr(C, "FORECAST_LOG_PATH", tmp_path / "forecast_log.jsonl")
    vlog = tmp_path / "version_log.json"
    vlog.write_text('{"live": "v1", "history": []}' if live else '{"live": null, "history": []}')
    monkeypatch.setattr(C, "VERSION_LOG_PATH", vlog)
    AppendOnlyLog(C.PREVIEW_LOG_PATH).append_many([
        _rec("skill:python", version="preview:v2", validated=False, feed="preview")])


def test_preview_export_is_refused_while_a_live_model_exists(monkeypatch, tmp_path):
    _point_logs(monkeypatch, tmp_path, live=True)
    with pytest.raises(E.ExportRefused, match="validated live model v1"):
        E.export("preview", now=NOW, valid_ids=VALID)
    assert E.main(["--feed", "preview", "--dry-run"]) == 2


def test_preview_export_works_when_no_live_model(monkeypatch, tmp_path, capsys):
    _point_logs(monkeypatch, tmp_path, live=False)
    rows, counts = E.export("preview", now=NOW, valid_ids=VALID)
    assert [r["model_version"] for r in rows] == ["preview:v2"] and counts["entities_exported"] == 1
    assert E.main(["--feed", "preview", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "skill:python" in out and "preview:v2" in out


def test_csv_output_matches_the_row_shape(monkeypatch, tmp_path):
    _point_logs(monkeypatch, tmp_path, live=False)
    rows, _ = E.export("preview", now=NOW, valid_ids=VALID)
    out = tmp_path / "rows.csv"
    E.write_csv(rows, str(out))
    with open(out, newline="", encoding="utf-8") as fh:
        got = list(csv.DictReader(fh))
    assert list(got[0]) == E.COLUMNS
    assert got[0]["entity_id"] == "skill:python" and got[0]["horizon_months"] == "3"
    assert got[0]["created_at"] == NOW.isoformat()
