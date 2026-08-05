"""Refresh the verified-result line in README.md straight from the Kaggle API.

Reads KAGGLE_USERNAME and KAGGLE_KEY from the environment (GitHub Actions secrets),
downloads the competition leaderboard, finds our team's row, and rewrites the block
between the KAGGLE:START and KAGGLE:END markers in README.md.

Only our own team's rank and score are written out. The downloaded leaderboard also
contains every other team's name and score, and none of that is published: those are
other students, and the file stays in the runner's temp directory.
"""

import csv
import io
import os
import re
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

COMPETITION = "the-analytics-edge-competition-2026"
TEAM = "sheil_mistry_team_3"
COMP_URL = f"https://www.kaggle.com/competitions/{COMPETITION}"

README = Path("README.md")
START = "<!-- KAGGLE:START -->"
END = "<!-- KAGGLE:END -->"


def fail(message):
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_leaderboard(workdir):
    """Download the leaderboard and return its rows, newest Kaggle format or older zip."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.competition_leaderboard_download(COMPETITION, path=str(workdir))

    for archive in sorted(workdir.glob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not names:
                continue
            with zf.open(names[0]) as fh:
                return list(csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")))

    for plain in sorted(workdir.glob("*.csv")):
        with plain.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    fail(
        "Kaggle returned no leaderboard file. The usual causes are an account that "
        "never joined this competition, or expired API credentials."
    )


def find_team(rows):
    """Kaggle returns the leaderboard already ordered by the competition metric."""
    for position, row in enumerate(rows, start=1):
        if (row.get("teamName") or "").strip().lower() == TEAM.lower():
            return position, row
    fail(
        f"team {TEAM!r} was not found among {len(rows)} leaderboard rows. "
        "If the team was renamed on Kaggle, update TEAM in this script."
    )


def main():
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        fail("KAGGLE_USERNAME and KAGGLE_KEY must both be set in the environment.")

    with tempfile.TemporaryDirectory() as tmp:
        rows = load_leaderboard(Path(tmp))

    if not rows:
        fail("the leaderboard came back empty.")

    rank, row = find_team(rows)
    total = len(rows)
    score = (row.get("score") or "").strip() or "unavailable"
    checked = datetime.now(timezone.utc).strftime("%d %B %Y")

    block = (
        f"{START}\n"
        f"> **Confirmed against the [Kaggle leaderboard]({COMP_URL}) on {checked}:** "
        f"team `{TEAM}` finished **{rank} of {total}** with a final score of "
        f"**{score}**. This line is rewritten by "
        f"[a workflow](.github/workflows/kaggle-result.yml) that reads the Kaggle API, "
        f"not typed by hand.\n"
        f"{END}"
    )

    text = README.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        fail(f"could not find the {START} / {END} markers in README.md.")

    updated = pattern.sub(lambda _: block, text)
    if updated == text:
        print(f"No change: still {rank} of {total}, score {score}.")
        return

    README.write_text(updated, encoding="utf-8")
    print(f"Updated: {rank} of {total}, score {score}, checked {checked}.")


if __name__ == "__main__":
    main()
