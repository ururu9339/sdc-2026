# API for the frontend

For the developer building the Replit frontend. The backend is FastAPI; the live OpenAPI schema is at `/docs`
(Swagger UI) and `/openapi.json`. Examples below were captured from a running server (values trimmed).

**Read this first.** No model has passed the validation gate yet. `/api/v1/forecast` therefore serves an
**unvalidated PREVIEW** (`validated: false`, a `banner`). Every screen that shows an outlook must show that banner
and must never present the numbers as validated. This rule holds for every forecast-bearing response, including
`/explain`, `/forecast/{id}` and `POST /report`.

## 1. Base URL and environment

- Base URL: `<API_BASE_URL>`, for example `https://<your-replit-app>.replit.app`. Keep it in one frontend
  config value (for example `VITE_API_BASE_URL` or `NEXT_PUBLIC_API_BASE_URL`). Do not hard-code the host.
- All routes live under `/api/v1`. The unprefixed paths (`/forecast`, `/health`, ...) still work but are hidden from
  the schema. Use `/api/v1` in new code.
- The API is read-only except `POST /api/v1/report` (which computes a response from the request and stores nothing)
  and the `/api/v1/candidates/*` routes (which store personal data; see section 8).
- CORS is set by the backend's `CORS_ORIGINS` environment variable (comma-separated). Default `*`, which allows any
  origin without credentials. The frontend does not need cookies, so credentials are not used.
- Every response has an `X-Request-ID` header. Send your own `X-Request-ID` (letters, digits, `.:_-`, up to 128
  characters) to correlate a call with a log line; otherwise the server makes one. Quote it when reporting a bug.

## 2. Endpoints

| Method | Path | Purpose | Notes |
|---|---|---|---|
| GET | `/api/v1/health` | Is the service up, what is served | Always HTTP 200. Use it for a status badge |
| GET | `/api/v1/forecast` | Forecast list for the current feed | Paged, sorted, searchable. ETag |
| GET | `/api/v1/forecast/{entity_id}` | One entity's forecast | 404 for an unknown id |
| GET | `/api/v1/explain/{entity_id}` | Plain-language summary, drivers, caveats | Template text, no invented numbers |
| GET | `/api/v1/history/{entity_id}` | Past share of postings | Descriptive, not a forecast. `n_periods` 1 to 520 |
| GET | `/api/v1/snapshot` | Past share change between two blocks of periods | Descriptive, not a forecast. ETag |
| GET | `/api/v1/model` | Live model (null), gate verdict, audit | ETag |
| GET | `/api/v1/sources` | Sources, tiers, licences, coverage | |
| POST | `/api/v1/report` | Skills (+ optional CV text) to matches, gaps, learning order, outlook | Never cached |
| GET | `/api/v1/report/candidate/{candidate_id}` | Report from a stored candidate | Personal data; never cached |
| GET | `/api/v1/taxonomy/skills` | Skill vocabulary (ids, names, alternative labels) | Paged, searchable. ETag |
| GET | `/api/v1/taxonomy/jobs` | Job vocabulary | Paged, searchable. ETag |
| POST | `/api/v1/candidates/form` | Create a candidate from a form | Personal data |
| POST | `/api/v1/candidates/pdf` | Create a candidate from a PDF or image (OCR) | Needs Tesseract on the server; 503 if absent |
| GET | `/api/v1/candidates/` | List candidates (`skip`, `limit`) | Personal data |
| GET | `/api/v1/candidates/{candidate_id}` | One candidate | Personal data |

Entity ids are taxonomy ids with a prefix: `skill:python`, `job:accountant`. Use the taxonomy endpoints to fill
pickers, and send the id exactly as returned.

### 2.1 Query parameters on the listings

`GET /api/v1/forecast`:

| Parameter | Values | Default | Meaning |
|---|---|---|---|
| `sort` | `growth`, `-growth`, `name`, `-name` | `-growth` | `-growth` puts the largest rise first; `growth` the largest fall first. Entities with no number always come last. `name` is A to Z by taxonomy name |
| `q` | text, up to 100 characters | none | Case-insensitive match on the id or the taxonomy name |
| `kind` | `job`, `skill` | none | |
| `direction` | `emerging`, `declining`, `stable` | none | |
| `status` | `ok`, `insufficient_data` | none | `ok` has a number; `insufficient_data` does not |
| `min_confidence` | `low`, `moderate`, `high` | none | Keeps this level and above; records with no confidence are dropped |
| `offset` | integer 0 or more | 0 | Records to skip |
| `limit` | integer 1 to 500 | 100 | Records per page |

`GET /api/v1/taxonomy/skills` and `/jobs`: `q` (text), `limit` (1 to 500, default 500), `offset`.

On the listing routes (`/forecast`, `/snapshot`, `/taxonomy/*`) an unknown query parameter is a 422 error rather than
being ignored, so check spelling.

### 2.2 Paging

Listings return `total` (all matches), `count` (items on this page), `offset` and `limit`. Next page:
`offset = offset + count`, while `offset + count < total`.

## 3. Response examples (trimmed)

### `GET /api/v1/health`

```json
{
  "status": "ok",
  "ready": true,
  "api_version": "v1",
  "data": {
    "live_model": null,
    "feed": "preview",
    "forecast_as_of": "2023-09-11",
    "n_forecasts": 123,
    "gate_profile": "demo",
    "last_gate": {"passed": false, "reason": "failed: beats_baseline, fold_win_rate"},
    "data_layout": "repo",
    "bundle": {"as_of": "2023-09-11", "feed": "preview", "model_version": "preview:v18", "validated": false,
               "n_files": 6, "total_bytes": 1798025},
    "sources_with_data": ["job_postings", "jobstreet_my", "market_signal"],
    "matching_engine_available": {"qwen_pgvector": false, "offline_alias": true}
  }
}
```

`ready: false` means the stored files could not be read; show "service degraded". `feed` is `live`, `preview` or
`none`, and tells you which banner rule applies.

### `GET /api/v1/forecast?limit=3` (preview feed)

```json
{
  "status": "preview", "feed": "preview", "validated": false,
  "banner": "PREVIEW: unvalidated candidate forecasts; the model did not pass the validation gate (failed: beats_baseline, fold_win_rate). Not for decisions.",
  "reason": "failed: beats_baseline, fold_win_rate",
  "model_version": "preview:v18", "gate_profile": "demo",
  "as_of": "2023-09-11", "target_period": "2023-12-04", "horizon_periods": 12,
  "disclaimer": "All sources are open or student-accessible data ...",
  "count": 3, "total": 123, "offset": 0, "limit": 3,
  "forecasts": [
    {"entity_id": "skill:plc", "kind": "skill", "status": "ok", "p50": 0.1815, "p10": 0.0055, "p90": 0.2474,
     "direction": "emerging", "confidence": "low", "signal_quality": "noise_like", "dispersion_ratio": 0.87,
     "validated": false, "feed": "preview", "preview_reason": "failed: beats_baseline, fold_win_rate",
     "forecast_id": "preview:v18:skill:plc:2023-09-11", "as_of": "2023-09-11", "target_period": "2023-12-04"}
  ]
}
```

Only the first record is shown; the others have the same fields. `forecasts` items also carry `source_contributions`,
`baseline_persistence`, `baseline_trend`, `hash` and `logged_at` (for audit). You can ignore them.

With a live model, `status` and `feed` are `ok` and `live`, `validated` is `true`, and `banner` is `null`.
With no model and no preview, `status` is `no_live_model`, `feed` is `none`, `forecasts` is `[]`, and `reason` and
`gate` explain why.

### `GET /api/v1/forecast/{entity_id}`

```json
{
  "entity_id": "skill:python", "kind": "skill", "status": "preview", "feed": "preview", "validated": false,
  "banner": "PREVIEW: unvalidated candidate forecasts; ... Not for decisions.",
  "reason": null,
  "forecast": {"p50": -0.0019, "p10": -0.0209, "p90": 0.0227, "direction": "stable", "confidence": "low",
               "signal_quality": "noise_like", "as_of": "2023-09-11", "target_period": "2023-12-04"},
  "disclaimer": "All sources are open ..."
}
```

`status` values for an entity: `ok` (live), `preview` (unvalidated number), `insufficient_data` (no number; see
`reason`), `no_live_model` (feed none).

### `GET /api/v1/explain/skill:python`

```json
{
  "entity_id": "skill:python", "kind": "skill", "status": "preview", "feed": "preview", "validated": false,
  "banner": "PREVIEW: unvalidated candidate forecasts; ... Not for decisions.",
  "summary": "PREVIEW: ... Python (skill) is stable (no clear move) over the next 12 weeks (to 2023-12-04). Median change in its share of postings: -0.2%; 10-90% range -2.1% to +2.3%. Signal quality: noise-like; confidence: low. Its movement is indistinguishable from random variation, so treat the direction as unconfirmed.",
  "drivers": {"source_contributions": {"job_postings": -0.0002, "calendar_meta": -0.0018},
              "source_contributions_units": "log change over the horizon",
              "related": [{"entity_id": "job:data_scientist", "kind": "job", "weight": 1.0, "n_cooccur": 7003}]},
  "caveats": ["All sources are open ...", "PREVIEW: not a validated forecast. ..."]
}
```

Show `summary` as the paragraph; show `caveats` under it. `summary` starts with the banner text while in preview.

### `POST /api/v1/report`

Request:

```json
{"skills": ["skill:python", "skill:sql", "skill:excel", "skill:data_analysis"], "horizon_months": 3, "engine": "auto"}
```

Trimmed response:

```json
{
  "engine": "offline_alias", "feed": "preview", "validated": false, "banner": "PREVIEW: ...",
  "disclaimer": "All sources are open ...", "horizon_months": 3,
  "status": "nothing_within_reach",
  "message": "No job reaches 50% coverage with a non-declining outlook. The closest jobs by coverage are shown, and their gaps are used instead.",
  "resolved_skills": [{"input": "skill:python", "skill_id": "skill:python", "name": "Python", "status": "matched", "method": "taxonomy_id"}],
  "unresolved": [], "targets": ["job:business_analyst", "job:data_analyst", "job:dba"],
  "best_fit_jobs": [{"job_id": "job:business_analyst", "name": "Business Analyst", "coverage": 0.5719,
                     "matched": [{"skill_id": "skill:sql", "name": "SQL"}], "missing": [{"skill_id": "skill:tableau", "name": "Tableau"}],
                     "growth": -0.0062, "forecast_status": "ok", "direction": "stable", "confidence": "low",
                     "signal_quality": "noise_like"}],
  "skill_gaps": [{"skill_id": "skill:tableau", "name": "Tableau", "score": 0.8591, "weight": 0.8578, "share": 0.8578,
                  "growth": 0.0016, "forecast_status": "ok", "direction": "stable", "confidence": "low",
                  "signal_quality": "noise_like", "for_jobs": ["job:business_analyst", "job:data_analyst"],
                  "stage": "Start here", "why": "Appears in 86% of Data Analyst postings in our data; ..."}],
  "learning_path": {"note": "The order is a heuristic, not a curriculum: ...",
                    "stages": [{"title": "Start here", "skills": [{"skill_id": "skill:tableau", "name": "Tableau"}]},
                               {"title": "Next", "skills": []}, {"title": "Later", "skills": []}]},
  "market_snapshot": [{"entity_id": "skill:tableau", "name": "Tableau", "mean_share_first": 0.0042,
                       "mean_share_last": 0.0060, "percent_change": 42.25, "type": "descriptive_not_forecast"}],
  "caveats": ["Coverage is an overlap score ...", "Skill gaps come from job postings in our data ...", "..."],
  "warnings": []
}
```

`status` values: `ok` (matches and gaps found); `nothing_within_reach` (no job clears the coverage bar; `message`
explains, and the closest jobs and their gaps are shown); `no_overlap` (skills resolved but match no job);
`no_skills_resolved` (nothing could be matched; ask the user to rephrase; `message` says so). Always render
`message` when it is not null.

Errors: a skill text longer than 200 characters, more than 100 skills, text over 20,000 characters, or a
`horizon_months` outside 1 to 12 gives 422 (`validation_error`).

### `GET /api/v1/snapshot?limit=3`

```json
{"type": "descriptive_not_forecast", "status": "ok", "count": 153, "kind": null,
 "banner": "DESCRIPTIVE SNAPSHOT, NOT A FORECAST: average share of postings in the first and last contiguous blocks of periods of the source. Past counts only; no trend is inferred.",
 "sources": [{"source_id": "jobstreet_my", "tier": 1, "status": "ok", "contiguous_blocks": 2, "entities_compared": 153}],
 "items": [{"type": "descriptive_not_forecast", "source_id": "jobstreet_my", "entity_id": "skill:communication",
            "kind": "skill", "name": "Communication skills", "mean_share_first": 0.4111, "mean_share_last": 0.4274,
            "percent_change": 3.97, "support": 26474,
            "first_block": "2024-02-19..2024-06-17", "last_block": "2025-03-03..2025-05-05"}]}
```

### `GET /api/v1/history/skill:python?n_periods=3`

```json
{"entity_id": "skill:python", "kind": "skill", "source_id": "job_postings", "granularity": "weekly",
 "n_periods_returned": 3,
 "series": [{"period": "2023-08-28", "value": 0.0413, "support": 648.0}],
 "audit": {"source_id": "job_postings", "entity_id": "skill:python", "n_periods": 53, "noise_like": true}}
```

`value` is the share of postings in that period. Plot it as history.

## 4. Errors

Every failure, from every route, has one shape:

```json
{"error": {"code": "not_found", "message": "Entity not found in observations.", "details": null,
           "request_id": "62ee784d27234ef0851d0e3d07f2c7b8"}}
```

| HTTP | `code` | When | What to show |
|---|---|---|---|
| 404 | `not_found` | unknown entity, candidate or path | "Not in our data" |
| 422 | `validation_error` | bad query, body or path. `details` is a list of `{loc, msg, type}` (the submitted values are never echoed) or, for unknown query parameters, `{unknown, allowed}` | Explain the field; do not show the raw text |
| 415 | `unsupported_media_type` | upload type not allowed | Ask for PDF, PNG, JPEG or TIFF |
| 503 | `service_unavailable` | a dependency is missing (for example OCR) | "This feature is unavailable right now" |
| 500 | `internal_error` | unexpected failure; no details are exposed | "Something went wrong" plus the `request_id` |

Network failures and timeouts have no body; treat them as "service unreachable" and offer retry. `/health` gives
`ready: false` when the data cannot be read.

## 5. ETag and caching

- `GET /forecast`, `/snapshot`, `/model`, `/taxonomy/skills` and `/taxonomy/jobs` send an `ETag` and
  `Cache-Control: no-cache`. Store the ETag with the cached body and send `If-None-Match` on the next request.
  A 304 means the cached body is still current (no body is sent).
- Different query strings are different resources; cache per URL.
- `POST /report` and `/report/candidate/{id}` send `Cache-Control: no-store`. Do not cache them.
- CORS exposes `ETag` and `X-Request-ID` to the browser, so your code can read them.

## 6. How to render outlook and confidence

**The banner rule.** When `validated` is `false`, show the response's `banner` text at the top of the view and on
each card that shows a number. Do not hide it, do not collapse it behind a click, and do not use a "validated"
badge. The banner text is server-provided; show it verbatim.

**Outlook chip.** Map `(validated, forecast_status, direction)` to a label. These are suggestions.

| validated | `status` (forecast) or `forecast_status` (report) | direction | Chip label | Suggested colour |
|---|---|---|---|---|
| false | `ok` or `preview` | `emerging` | Rising (unvalidated) | amber outline |
| false | `ok` / `preview` | `declining` | Falling (unvalidated) | amber outline |
| false | `ok` / `preview` | `stable` | Stable (unvalidated) | grey outline |
| true | `ok` | `emerging` | Rising | green |
| true | `ok` | `declining` | Falling | red |
| true | `ok` | `stable` | Stable | neutral |
| any | `insufficient_data`, `no_forecast`, or null | null | No outlook | grey |

Add "unconfirmed" to the label when `confidence` is `low` or `signal_quality` is `noise_like`.

**What the numbers mean.**
- In forecasts, `p50` is the median **log** change in the share of postings over the horizon. The relative change
  is `exp(p50) - 1`. In reports, `growth` is already that relative change. Either way it is the change in the
  **share of postings**, not in posting counts: `+0.12` means the share is forecast to be 12% higher.
- `p10` and `p90` are the 10th and 90th percentiles; show them as a range, for example `exp(p10) - 1` to `exp(p90) - 1`.
- `horizon_periods` is 12 weekly periods (about 3 months). `target_period` is the period the forecast refers to.
- `confidence`: `high` when the whole 10 to 90% interval agrees with the direction, `moderate` otherwise, and `low`
  when the movement is noise-like.
- `signal_quality`: `trend_detected` means movement larger than sampling noise (not a verified trend);
  `noise_like` means it is indistinguishable from random variation (show as unconfirmed); `unknown` means not
  assessable.
- `coverage` in a report is an overlap score between the user's skills and a job's most common skills. It is not
  a judgement of ability.

**Empty and error states.**
- `feed: none`: "No outlook yet. No model has passed our validation checks." Show `reason` for detail.
- `status: insufficient_data` on an entity: show "Not enough history to forecast this yet", and `reason` in a
  tooltip. Do not show a number.
- Empty list (`forecasts: []`, `total: 0`): "No matches for these filters."
- Report `nothing_within_reach`, `no_overlap`, `no_skills_resolved`: show `message`, and suggest adding skills.

**Learning path.** Show `learning_path.stages` in order (`Start here`, `Next`, `Later`), with `learning_path.note`
visible. The note says the order is a heuristic from posting counts, not a curriculum or a time estimate.

**Snapshot.** Label the section "Past share of postings" with the snapshot banner visible. Its `percent_change` is
the change between two past blocks of periods. Never draw it as a forecast or with an arrow pointing to the future.

## 7. What not to do

- Do not show growth, direction, or any forecast number as validated while `validated` is false.
- Do not drop or hide the banner, and do not invent numbers, ranges or labels the response does not contain.
- Do not compute your own trend from `history` or `snapshot` and present it as a forecast.
- Do not treat `percent_change` in a snapshot as a prediction.
- Do not show `hash`, `logged_at` or `forecast_id` as user-facing content; they are for audit.
- Do not send CV text to any endpoint other than `POST /api/v1/report` or the candidate routes.
- Do not cache `POST /report` responses or store them in local storage with the user's skills.
- Do not render `message`, `summary` or `reason` as HTML; treat them as plain text.

## 8. Personal data

`/api/v1/candidates/*` and `/api/v1/report/candidate/{id}` handle personal data (name, email, skills, experience).
Show them only to the person concerned, use HTTPS, and do not log their contents. The candidate database is a local
SQLite file on the server and is not restored after a redeploy (see `docs/DEPLOY_REPLIT.md`).

## 9. Quick checklist for a new screen

1. Load `/api/v1/health` and read `data.feed` and `ready`.
2. Load `/api/v1/forecast` with `limit`, `sort` and paging; keep the banner from the envelope.
3. For one entity: `/api/v1/forecast/{id}` and `/api/v1/explain/{id}`. Show `summary` and `caveats`.
4. Map outlook chips with the table in section 6.
5. Handle the error shape in section 4 and show `request_id` on 500.
