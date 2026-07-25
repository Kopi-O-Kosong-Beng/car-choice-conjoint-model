---
title: Analytics Edge — Hub
type: index
course: The Analytics Edge (SUTD, Term 5)
updated: 2026-07-26
tags: [hub, competition, analytics-edge]
---

# The Analytics Edge — Hub

> **Setup:** open the **repository root** as your Obsidian vault (the folder containing
> `Vault/`, `model/`, and `Raw Dump/`). That way the lecture PDFs under `Raw Dump/`
> resolve as links and open inside Obsidian.
>
> **For an AI assistant reading this vault:** start at [[Vault/Competition/Modeling Strategy & Results|Modeling Strategy & Results]]
> for what we built, and `CLAUDE.md` in the repository root for how to work in the codebase.

## 🏁 Competition — live status

**Public leaderboard: 1.201** · local nested CV 1.13883 · benchmark 1.38629 · rival 1.210
Kaggle closes **1 Aug 2026 12:00 SGT**; report due **10 Aug 2026 12:00 SGT**.

- [[Vault/Competition/Brief & Rules|Brief & Rules]] — metric, leaderboard mechanics, grading
- [[Vault/Competition/Data Dictionary|Data Dictionary]] — every column, structure, train↔test shift
- [[Vault/Competition/Modeling Strategy & Results|Modeling Strategy & Results]] — the two winning models, scores, what's queued
- [[Vault/Competition/Key Findings|Key Findings]] — the four results worth putting in the report

Outside the vault: `EXPERIMENTS.md` (research log with reflections) ·
`report_notes.md` (report draft material) · `submissions/log.md` (every score) ·
`README.md` (how to run the code)

## 📚 Course topics

| # | Note | Case studies | Used in competition? |
|---|---|---|---|
| 0 | [[Vault/Topics/Topic 0 - Stats Refresher\|Stats Refresher]] | Old Faithful, t-tests | indirectly |
| 1 | [[Vault/Topics/Topic 1 - Linear Regression\|Linear Regression]] | Wine prices, Moneyball | no |
| 2 | [[Vault/Topics/Topic 2 - Logistic Regression\|Logistic Regression]] | Framingham, Challenger | foundation for Topic 3 |
| 3 | [[Vault/Topics/Topic 3 - Discrete Choice\|Discrete Choice]] ⭐ | **the competition dataset** | **yes — core** |
| 4 | [[Vault/Topics/Topic 4 - Model Selection\|Model Selection]] | Hitters, economic growth | **yes — CV, regularization** |
| 6 | [[Vault/Topics/Topic 6 - CART and Random Forests\|CART & Random Forests]] | Supreme Court votes | **yes — boosting** |
| 7 | [[Vault/Topics/Topic 7 - Clustering and RecSys\|Clustering & RecSys]] | MovieLens | considered, not used |
| 8 | [[Vault/Topics/Topic 8 - Matrix Factorization and PCA\|Matrix Factorization & PCA]] | SVD, Social Progress | no |
| 9 | [[Vault/Topics/Topic 9 - Survival Analysis\|Censored Data & Survival]] | Heart transplant | no |

*(No Topic 5 folder exists in the course materials — likely recess/midterm week.)*

## ⭐ Start here for the competition

The dataset **is** the discrete-choice "safety" data from class. The ideas that actually
scored — conditional logit, part-worth utilities, heterogeneity, willingness-to-pay — all
live in [[Vault/Topics/Topic 3 - Discrete Choice]].
