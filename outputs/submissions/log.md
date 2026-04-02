# Submission Log

Track all Kaggle submissions for this competition.

| # | Date | CV F1 | LB F1 | Model | Features | Notes |
|---|------|-------|-------|-------|----------|-------|
| 1 | 2026-03-31 | 0.7477 | 0.7941 | LogisticRegression (C=1.0) | TF-IDF (5000) + KeywordEncoder + numeric (text_len, word_count) | Baseline — LB > CV due to full-data retrain |
| 2 | 2026-04-01 | 0.7567 | 0.7873 | LogisticRegression (C=1, balanced, lbfgs) | Same as #1 | Tuned via GridSearchCV — CV improved but LB regressed. class_weight=balanced too aggressive for mild 57/43 imbalance |
| 3 | 2026-04-02 | 0.7567 | — | LogReg (tuned) — best of 4-model comparison | Same as #1 | XGBoost (0.7248) and LightGBM (0.7216) both below LogReg on sparse TF-IDF. No new submission — same as #2 |
