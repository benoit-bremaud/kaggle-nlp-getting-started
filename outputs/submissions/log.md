# Submission Log

Track all Kaggle submissions for this competition.

| # | Date | CV F1 | LB F1 | Model | Features | Notes |
|---|------|-------|-------|-------|----------|-------|
| 1 | 2026-03-31 | 0.7477 | 0.7941 | LogisticRegression (C=1.0) | TF-IDF (5000) + KeywordEncoder + numeric (text_len, word_count) | Baseline — LB > CV due to full-data retrain |
| 2 | 2026-04-01 | 0.7567 | 0.7873 | LogisticRegression (C=1, balanced, lbfgs) | Same as #1 | Tuned via GridSearchCV — CV improved but LB regressed. class_weight=balanced too aggressive for mild 57/43 imbalance |
| — | 2026-04-02 | — | — | XGBoost (0.7248) / LightGBM (0.7216) | Same as #1 | Experiment only — both tree-based models below LogReg baseline on sparse TF-IDF. No submission generated |
| 3 | 2026-04-02 | 0.7677 | 0.8039 | LogReg (C=1, balanced, liblinear) | Word TF-IDF (5000) + Char TF-IDF (3000) + keywords + numeric (text_len, word_count, mention_count, hashtag_count) | Advanced features — char n-grams + mention/hashtag counts. New best LB (+0.010 vs baseline) |
| 4 | 2026-04-04 | 0.7710 | 0.8002 | LogReg (C=1, balanced, liblinear) + SelectKBest(k=3000) | Best 3000 of 8226 features (chi2 selection) | Feature selection — CV 0.7710 (Pipeline, no leakage). Initial leaky CV was 0.7855. LB -0.004 vs #3 |
