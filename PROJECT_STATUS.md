# WatchMyBirds
Some Notes:
- Auth/TLS intentionally deferred. App currently runs LAN-only.
- SQLite locking: single-writer model confirmed. No additional locking required at this stage.

## Current Status

---

## Current Phase

---

## Next Steps

---

## Recent Changes
- 2026-02-03: Top-images script can process all species and multiple modes per run.
- 2026-02-03: Top-images script now injects repo root into PYTHONPATH for direct execution.
- 2026-02-03: Added a script to export top-N images per species by brightness/confidence/sharpness from the full database.
- 2026-02-03: No-detection daylight alert logs its accumulated/threshold counters when it triggers.
- 2026-02-03: Status alert checks now run during daylight gating to prevent overnight daylight accumulation for no-detection alerts.
- 2026-02-01: Added optional collage selection debug logging to explain filtering outcomes.
- 2026-02-01: Collage selection now buckets by time/hash to keep output monotonic when tightening filters.
- 2026-02-01: Collage tiles now trim dark borders when requested, reducing black padding in generated collages.
- 2026-02-01: Collage selection now supports configurable quality/diversity filters; zoomed crops avoid black padding and a helper script builds collages for the last three days.
- 2026-02-01: No-detection alert now accumulates daylight-only time and resets at night to avoid sunrise false positives.
- 2026-01-15: Daylight checks now apply TELEGRAM_TIMEZONE for coordinate locations to align no-detection alerts with local sunrise; updated daylight helper accordingly.
- 2026-01-13: Added automatic camera reinitialize on missing frames and Windows webcam backend cycling.
- 2026-01-13: Fixed daylight cache fallback to avoid NameError on invalid locations.
- 2026-01-13: Added Telegram status alerts for camera offline, missing frames, and 5h no-detection daylight window.
- 2026-01-13: Day/night daylight check now accepts coordinates in DAY_AND_NIGHT_CAPTURE_LOCATION.
- 2026-01-13: Daily Telegram summary now sends 20 minutes before sunset instead of a fixed time.
- 2026-01-04: Added classifier-based save gate (detector + classifier).
- 2026-01-04: Added day/night stream FPS overrides for capture and UI.
- 2026-01-04: Added a helper script to print current sunrise/sunset window and gate status.
- 2026-01-04: Added Telegram rule selection (basic vs daily summary) with daily stats and per-species collages.
- 2026-01-04: Normalized gallery image URLs to use forward slashes so thumbnails load on Windows.
- 2026-01-04: Resolve relative OUTPUT_DIR and MODEL_BASE_PATH against the project root to prevent missing images when the app is started from a different working directory.
- 2025-12-18: Added model provenance columns (`detector_model_id`, `classifier_model_id`) to SQLite and populate them from `latest_models.json`.
- 2025-12-18: Landing page now lazy-loads the gallery-style Daily Summary for today (cached) without delaying first render.
- 2025-12-18: Added daily species summary (per-day SQL + cache), exposed `/api/daily_species_summary`, and lazy-loaded it on the landing page.
- 2025-12-16: CODEX Planing-Regel ergänzt: BLOCKED-Status erfordert Begründung.
- 2025-12-16: Docker image now creates `/models` and `/output` directories at build time (no model/output copy).
- 2025-12-16: Entrypoint now respects `OUTPUT_DIR` for mkdir/chown (PUID/PGID ownership).
- 2025-12-16: Cleaned requirements.txt to direct dependencies only.
- 2025-12-16: Added runtime settings YAML (`OUTPUT_DIR/settings.yaml`), `/api/settings` GET/POST, and `/settings` UI page.
- 2025-12-16: Switched persistence to SQLite (`images.db` in OUTPUT_DIR) with the same columns as prior CSVs; writes remain in processing worker; reads routed via SQLite in UI.
- 2025-12-16: Added one-time CSV import script for legacy data (`scripts/import_csv_to_sqlite.py`).
- 2025-12-16: Documented full configuration reference table and confirmed no unused config keys.
- 2025-12-16: Completed Phase 3c (resources): CPU affinity reads shared config; capture/UI FPS normalization kept consistent.
- 2025-12-16: Completed Phase 3b (day/night caching): sun-position daylight check cached with TTL and fallback to daytime on error (behavior unchanged).
- 2025-12-16: Completed Phase 3a (detection vs I/O): detection loop stays lightweight; processing/EXIF/CSV/Telegram moved to worker queue; Telegram best-effort unchanged.
- 2025-12-16: Completed Phase 1; centralized config (`get_config`), DetectionManager-owned lifecycle, cache reuse without re-probe, freshest-frame delivery, separate capture/UI FPS settings, UI initial render fixed.
- 2025-12-12: Web UI startup decoupled from stream init; UI now available immediately with async stream status and placeholders.
- 2025-12-12: Stream settings cache now validated (URL/type/FFmpeg), atomically written, and updated on runtime resolution changes.
- 2025-12-12: Added stream settings cache to skip FFmpeg probing on restart and speed up startup.
- 2025-12-12: Hardened FFmpeg lifecycle to prevent orphaned processes and ensure clean restarts.
- 2025-08-06: Revised VideoCapture lifecycle and added FrameGenerator.
- 2025-08-05: Introduction of MODEL_BASE_PATH for unified model storage.
- 2025-08-05: Migration of model downloads to Hugging Face.
- 2025-07-28: Branch created.

---

## Notes for Codex
- Always read this file first to understand the current phase and history.
- Keep documentation in `README.md` and `PROJECT_STATUS.md` up to date.
- Update status here after each phase.
