# Codex Project Instructions - WatchMyBirds Inference App

## 1. Code Guidelines
- Follow `AGENTS.md` for style and workflow rules.
- Add TODO comments if uncertain.

## 2. Scope & Permissions
- Only modify production scripts; never touch files in `legacy/`.
- Always consider this file, `AGENTS.md`, and `PROJECT_STATUS.md` before making changes.

## 3. Environment & Dependencies
- Python 3.10+ (aligned with project README).
- Install dependencies via `pip install -r requirements.txt`.
- For new packages: update `requirements.txt` and document the change.

## 4. Documentation Maintenance
- Documentation must be updated after every code change.
- Update `README.md` and `PROJECT_STATUS.md` in the same commit as the code.
- Keep folder structure, config keys, tests, and usage notes in sync with reality. If unclear, regenerate the relevant sections to match the current state.

## 5. Testing & Quality
- Run `pytest` before merge/automation.
- Keep code formatted (Black 88 chars; ESLint airbnb-base for JS in `static/`).

## 6. Automation
- GitHub Actions or other automation should run only when documentation is up to date and tests pass.
- If tests fail, do not merge automatically.
