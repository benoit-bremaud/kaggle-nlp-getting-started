# Contributing Guide

This document defines the conventions for working on this Kaggle project.

## Getting Started

```bash
make setup      # Install dependencies and configure hooks
make notebook   # Start Jupyter Lab
```

## Git Workflow

### Branches

- **`main`** — Stable, validated code only. Protected (requires PR).
- **`feat/{name}`** — Feature/experiment branches (e.g., `feat/xgboost-tuning`).

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>
```

**Types:**
| Type | Usage |
|---|---|
| `feat` | New feature or experiment |
| `fix` | Bug fix |
| `refactor` | Code restructuring without behavior change |
| `docs` | Documentation changes |
| `chore` | Maintenance tasks (deps, config) |
| `data` | Data pipeline changes |
| `model` | Model training/tuning changes |

**Scopes:** `notebook`, `src`, `data`, `model`, `infra`

**Examples:**
```
feat(notebook): add correlation heatmap to EDA section
fix(src): handle missing values in load_data utility
model(notebook): train XGBoost with tuned hyperparameters
data(notebook): add target encoding for categorical features
chore(infra): update ruff to v0.9
```

## Adding Dependencies

```bash
# Add to requirements.txt, then:
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Code Quality

- **Linting:** `make lint` (ruff check)
- **Formatting:** `make format` (ruff format)
- **Pre-commit hooks** run automatically on `git commit`
- **Notebook outputs** are stripped automatically by nbstripout

## Submission Workflow

1. Generate predictions in the notebook (Section 9)
2. The submission cell checks if predictions differ from the latest file — only saves a new numbered CSV if they do
3. Submit to Kaggle: `make submit COMPETITION=nlp-getting-started FILE=outputs/submissions/submission_NNN.csv` or upload via web
4. Log both scores (CV F1 + LB F1) in `outputs/submissions/log.md`
5. Update `PROJECT_LOG.md` with a Submission entry (ADR-029)

## Pull Request Template

PRs should follow the template in `.github/PULL_REQUEST_TEMPLATE.md`.

## Labels

When creating issues, assign relevant labels from the DS-oriented label system (see CLAUDE.md for full list).
