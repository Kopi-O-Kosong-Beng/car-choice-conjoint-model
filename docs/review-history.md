# Review history — the five pull requests

This project was developed across two GitHub accounts and reviewed through pull requests. The
repository was renamed to `car-choice-conjoint-model` on 1 Aug 2026; pull requests do not carry
across a repository migration, so this file is the index to them.

Nothing of substance was in the PR pages that is not in the repository. Every commit, every
branch and full authorship are intact, and in this project the **merge commits carry the review
record** — the reasoning, the verification gates run before merging, and the conflict
resolutions. `git show 068ba66` is a fuller account than the PR description was.

| # | title | state | branch | merge commit |
|---|---|---|---|---|
| 1 | The combiner could not represent a negative weight (1.12819 → 1.12341) | merged | `sheil/free-sign-blend-1.12341` | `d94f4d4` (27 Jul) |
| 2 | The free-sign blend was refuted on the leaderboard (1.12341 local → 1.209 public) | merged | `sheil/free-sign-refuted-1.209` | folded into `d94f4d4` lineage |
| 3 | The design-share encoding is leaking — it invalidates the tree tuning, and it explains three failed submissions | merged | `sheil/encoding-leak-1.194` | `e8e1d22` (29 Jul) |
| 4 | Remove the delisted iter63 artifact and everything derived from it | **closed, not merged** | — | — |
| 5 | Iteration 80 — the model-probe, and the state it settles | merged | `sheil/docs-state-update` | `068ba66` (31 Jul) |

All four branches are present in this repository and all four are ancestors of `main`.

## Reading the review record

```bash
git log main --merges                    # the three merge commits, with full review notes
git show 068ba66                         # PR #5's verification gates and conflict resolution
git log origin/sheil/encoding-leak-1.194 # PR #3's commit-by-commit development
git log --format='%an' main | sort -u    # authorship, preserved
```

## Why PR #4 matters even though it was closed

It proposed removing a delisted artifact and everything derived from it. It was closed rather
than merged — which is itself the record of a decision, and consistent with how this project
treated negative results throughout: `EXPERIMENTS.md` documents failures at the same length as
successes, because on this problem the failures carried more information than the wins.

## Contributors

| author | commits |
|---|---|
| Zhi Feng | 48 |
| Sheil | 9 |

Plus Kavya and Nicole on the modelling and report, and the second modelling track
(`experiments/iter62_nnblend/`) which contributed half of the final submission.
