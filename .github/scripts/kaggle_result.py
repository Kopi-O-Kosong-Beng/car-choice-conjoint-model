"""Refresh the verified-result line in README.md straight from the Kaggle API.

Reads KAGGLE_USERNAME and KAGGLE_KEY from the environment (GitHub Actions secrets),
downloads the competition leaderboard, finds our team's row, and rewrites the block
between the KAGGLE:START and KAGGLE:END markers in README.md.

Two things worth knowing about what Kaggle returns:

  * The download is the PUBLIC leaderboard, even after a competition closes. Our own
    private placing (4th) differs from our public placing (3rd), so this line is
    labelled public rather than final.
  * Only our own team's row is written out. The same file carries every other team's
    name, score and member usernames; none of that is published, and it stays in the
    runner's temporary directory.
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
TEAM = "Sheil_Mistry_Team_3"
COMP_URL = f"https://www.kaggle.com/competitions/{COMPETITION}"

README = Path("README.md")
START = "<!-- KAGGLE:START -->"
END = "<!-- KAGGLE:END -->"


def fail(message):
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def norm(name):
    """Kaggle ships the header capitalised and with a BOM; compare on neither."""
    return (name or "").lstrip("﻿").strip().lower()


def pick(row, *candidates):
    """Fetch a column by case-insensitive name, tolerating the BOM on the first one."""
    lookup = {norm(k): v for k, v in row.items()}
    for candidate in candidates:
        if norm(candidate) in lookup:
            return lookup[norm(candidate)]
    return None


def explain_http_error(err):
    """Turn Kaggle's bare HTTP codes into something actionable."""
    status = getattr(getattr(err, "response", None), "status_code", None)
    if status == 401:
        fail(
            "Kaggle rejected the credentials (401).\n"
            "  The secrets are reaching the runner, so the values themselves are wrong.\n"
            "  Check, in this order:\n"
            "   1. KAGGLE_USERNAME must be the profile slug from kaggle.com/<username>, "
            "not an email address and not the display name.\n"
            "   2. KAGGLE_KEY must be only the value of \"key\" from kaggle.json, "
            "without quotes, braces or a trailing newline.\n"
            "   3. Creating a new API token on Kaggle invalidates the previous one, so "
            "if a token was regenerated after the secret was saved, save the new value."
        )
    if status == 403:
        fail(
            "Kaggle authenticated the account but refused this competition (403).\n"
            f"  Open {COMP_URL} while signed in as that account and accept the rules. "
            "In-class competitions stay closed to accounts that never joined."
        )
    if status == 404:
        fail(
            f"Kaggle has no competition called {COMPETITION!r} (404). "
            "Check the slug in the competition URL and update COMPETITION in this script."
        )
    fail(f"Kaggle API call failed: {err}")


def load_leaderboard(workdir):
    """Download the leaderboard and return its rows, newest Kaggle format or older zip."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    from requests.exceptions import HTTPError

    try:
        api = KaggleApi()
        api.authenticate()
        api.competition_leaderboard_download(COMPETITION, path=str(workdir))
    except HTTPError as err:
        explain_http_error(err)

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

    fail("Kaggle returned no leaderboard file for this competition.")


def find_team(rows):
    for position, row in enumerate(rows, start=1):
        if norm(pick(row, "TeamName")) == norm(TEAM):
            rank = (pick(row, "Rank") or "").strip() or str(position)
            score = (pick(row, "Score") or "").strip() or "unavailable"
            return rank, score
    columns = ", ".join(sorted(norm(k) for k in rows[0])) if rows else "none"
    fail(
        f"team {TEAM!r} was not found among {len(rows)} leaderboard rows.\n"
        f"  Columns Kaggle returned: {columns}\n"
        "  If the team was renamed on Kaggle, update TEAM in this script."
    )


def main():
    user = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")
    if not (user and key):
        fail("KAGGLE_USERNAME and KAGGLE_KEY must both be set in the environment.")

    # Catch the two mistakes that produce a confusing 401, without echoing any value.
    if "{" in user or "{" in key:
        fail(
            "A secret contains '{', so the whole kaggle.json was probably pasted in. "
            "KAGGLE_USERNAME takes only the username, KAGGLE_KEY only the key."
        )
    if "@" in user:
        fail(
            "KAGGLE_USERNAME looks like an email address. Kaggle wants the profile "
            "slug shown at kaggle.com/<username>."
        )

    with tempfile.TemporaryDirectory() as tmp:
        rows = load_leaderboard(Path(tmp))

    if not rows:
        fail("the leaderboard came back empty.")

    rank, score = find_team(rows)
    total = len(rows)
    checked = datetime.now(timezone.utc).strftime("%d %B %Y")

    block = (
        f"{START}\n"
        f"> **Read from the [Kaggle leaderboard]({COMP_URL}) on {checked}:** team "
        f"`{TEAM}` placed **{rank} of {total}** on the public board with a score of "
        f"**{score}**. Kaggle serves the public leaderboard through its API, so the "
        f"private placing of 4th is not shown here. This line is written by "
        f"[a workflow](.github/workflows/kaggle-result.yml), not by hand.\n"
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
