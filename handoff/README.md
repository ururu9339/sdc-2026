# Hand-off files for the embeddings team

Generated from our forecasting pipeline; see `docs/INTEGRATION.md` for the contract and the load steps.

- `entity_forecast_preview.csv`: 123 rows for the `entity_forecast` table. **Unvalidated preview** (`model_version = preview:v17`): the model did not pass our validation gate, so do not use these for decisions. `horizon_months = 3`; call `embeddings.match` with `--horizon 3`. `growth = exp(p50) - 1`, the forecast relative change in the smoothed share of postings.
- `job_skill_graph_measured.csv`: 241 measured job-skill co-occurrence edges (taxonomy ids) from the job-postings source, for `job_skill_edge`. Built by keyword/alias matching, not by Qwen.

Both files are snapshots; regenerate with `python3 -m src.pipeline.export_forecasts --feed preview --csv ...` and `python3 -m src.sources.build_all`.
