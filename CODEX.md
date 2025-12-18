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


## 7. Planning
For larger tasks, follow this strict mode:

7.1. Create a PLAN upfront:
   - Main Goal with clear success criteria
   - Numbered Subgoals
   - Concrete tasks per Subgoal
   - Initial status for every task: TODO

7.2. Use fixed STATUS values:
   - TODO
   - IN PROGRESS
   - DONE
   - BLOCKED (must include a short reason)

7.3. After EVERY work step:
   - Update the plan
   - Mark finished tasks as DONE
   - Adjust tasks or subgoals if new findings arise
   - Document changes explicitly under “Plan changes”

7.4. Always work in this order:
   - Show the plan
   - Execute one task
   - Update the plan
   - Only then continue

7.5. If assumptions change or new problems appear:
   - Update the plan first
   - Briefly explain what changed and why

7.6. Never reply with pure code/text only; always show the current plan with status.

### Example: structured and checkable
## Goal
UI latency with 100k SQLite entries < 500 ms on Gallery load
No changes to UI or output

Success criteria:
- Gallery loads without visible delay
- Subgallery pagination < 300 ms
- /species initial load < 1 s

---

## Subgoals

### SG1: Optimize DB queries
- T1: fetch_image_summaries without coco_json → DONE
- T2: fetch_day_images(date) → DONE
- T3: fetch_hourly_counts(date) → TODO
Status: IN PROGRESS

### SG2: Decouple UI from full scans
- T4: Gallery uses daily covers → TODO
- T5: Subgallery uses day-scoped queries → TODO
Status: TODO

### SG3: Performance for Edit/Delete
- T6: Index on optimized_name → DONE
Status: DONE

### SG4: Adjust caching strategy
- T7: Cache summary rows only → TODO
- T8: Increase TTL → TODO
Status: TODO

### SG5: Fix /species initial load
- T9: Long-TTL cache → TODO
Status: TODO

---

## Current Focus
SG1.T3

## Plan changes
None
