# Project Log

Chronological record of all project activity — PRs, decisions, backlog changes, submissions.
Agent-agnostic, DIP-compliant. This file is the single source of truth for project history.

| Date | Type | Ref | Description |
|------|------|-----|-------------|
| 2026-03-27 | Setup | — | Project created from template, environment configured |
| 2026-03-30 | Backlog update | #1–#5 | Initial issue plan created: EDA, Data Cleaning, Feature Engineering, Baseline Model, First Submission |
| 2026-03-30 | PR merged | #6 | EDA — exploratory analysis of disaster tweets dataset (target distribution, text lengths, keyword signal) |
| 2026-03-30 | PR merged | #7 | Data cleaning — text preprocessing pipeline in `src/text.py` (URL removal, lowercasing, keyword decoding) |
| 2026-03-30 | PR merged | #8 | Feature engineering — TF-IDF + KeywordEncoder + numeric features in `src/features.py` |
| 2026-03-30 | PR merged | #9 | Baseline model — LogisticRegression (C=1.0), CV F1: 0.7477 |
| 2026-03-31 | PR merged | #10 | First Kaggle submission — retrain on 100% data, generate `submission_001.csv` |
| 2026-03-31 | Submission | #1 | Baseline submitted to Kaggle — CV F1: 0.7477 / LB F1: 0.7941 |
| 2026-04-01 | Backlog update | #11–#15 | Phase 3 issues created: HP tuning, gradient boosting, advanced features, feature selection, ensemble |
| 2026-04-01 | PR merged | #16 | HP tuning — GridSearchCV on LogisticRegression (C, solver, class_weight), best: C=1, balanced, lbfgs |
| 2026-04-01 | Submission | #2 | Tuned LogReg submitted — CV F1: 0.7567 / LB F1: 0.7873 (regression vs baseline, class_weight=balanced too aggressive) |
| 2026-04-02 | PR merged | #17 | Model comparison — XGBoost and LightGBM vs LogReg. Both tree-based models underperform on sparse TF-IDF (0.72 vs 0.75). LogReg remains best |
| 2026-04-02 | PR merged | #18 | Advanced text features — char n-grams (3000), mention_count, hashtag_count in `src/text.py` and `src/features.py` |
| 2026-04-02 | Submission | #3 | Advanced features submitted — CV F1: 0.7677 / LB F1: 0.8039. New best LB, first above 0.80 |
| 2026-04-04 | Decision | #15 | Closed ensemble issue as wontfix — tree-based models too weak for useful blend (separate from #14) |
| 2026-04-04 | PR merged | #19 | Feature selection — SelectKBest(chi2) tested, leakage bug fixed via Pipeline. No improvement over full features. Full set (8226) confirmed optimal |
| 2026-04-04 | Submission | #4 | Feature selection submitted — CV F1: 0.7710 (k=3000 with Pipeline) / LB F1: 0.8002 (regression vs #3). Full features remain best |
| 2026-04-04 | PR merged | #20 | Documentation update — README, CONTRIBUTING, CLAUDE.md aligned with project state. Broken links fixed, CV F1 #4 corrected |
