# AGENTS.md – Guidelines for Codex and Agents

> Purpose: Keep all agents aligned on quality, documentation, and safety.

---

## 1. Setup
- Python 3.10+ (align with project README).
- Install missing packages with `pip install -r requirements.txt`.
- When adding dependencies: update `requirements.txt` and document the change.

## 2. Code Style
| Language | Formatter / Linter | Key Rules |
| --- | --- | --- |
| Python | Black (88 chars), UTF-8 | Do not use `from ... import *`. |
| JavaScript (ES6) | ESLint (airbnb-base) | JS files live only in `static/`. |

### 2.1 Naming Conventions
- `snake_case` for variables and functions.
- `PascalCase` for classes.
- Boolean names: `is_...`, `has_...`, `should_...`.
- Loop indices `i/j` only in loops shorter than 5 lines.
- No cryptic abbreviations (`df` → `pandas_dataframe`).

### 2.2 Docstrings
- Triple-quoted `"""` directly under the definition.
- First line: verb in 3rd person singular (German).
- Multiline for complex functions:

```
Args:
    name (type): Description.
Returns:
    type: Description.
Raises:
    ErrorType: When the error occurs.
```

Example:
```python
def lade_daten(pfad: str) -> pd.DataFrame:
    """Lädt Daten aus einer CSV-Datei."""
```

## 3. Commit Messages
- Format: `<type>: <short summary>`.
- Types: feat, fix, refactor, docs, style, test, build, ci.
- Example: `feat: add hyperparameter tuning with Optuna`.

## 4. Testing & Quality
- Run `pytest` before merging.
- Keep code formatted (`black` for Python; ESLint for JS in `static/`).

## 5. Documentation
- Every code change must update documentation in the same commit.
- Update `README.md` and `PROJECT_STATUS.md`; add new configuration parameters when applicable.

## 6. Scope & Consistency
- Do not modify files in `legacy/`.
- Always read `CODEX.md`, `AGENTS.md`, and `PROJECT_STATUS.md` before starting.
- Follow instructions from both `CODEX.md` and this file.

## 7. CI/CD and Automation
- Automation runs only when documentation is up to date and tests succeed.
- If tests fail, do not merge automatically.

---

🔑 Maintaining documentation is a fixed part of every development task. Codex must not finalize changes without these updates. Changes to the codebase must implement the current objectives.
