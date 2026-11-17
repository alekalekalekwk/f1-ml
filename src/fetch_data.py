"""Fetch F1 race results with FastF1 and save them to data/raw/results.csv.

Usage:
    python src/fetch_data.py --test              # one race only, print it
    python src/fetch_data.py                     # all seasons 2022-2026
    python src/fetch_data.py --years 2025 2026   # selected seasons

Runs are incremental: rounds already in results.csv are not fetched again.
If the FastF1 API rate limit (500 calls/h) is hit, the run saves what it has
and stops; run it again later to fetch the rest.
"""

import argparse
import logging
from pathlib import Path

import fastf1
import pandas as pd
from fastf1.exceptions import F1AuthError, RateLimitExceededError

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
OUT_FILE = ROOT / "data" / "raw" / "results.csv"
DEFAULT_YEARS = [2022, 2023, 2024, 2025, 2026]

log = logging.getLogger("fetch_data")

RESULT_COLUMNS = {
    "Abbreviation": "driver",
    "FullName": "driver_name",
    "TeamName": "team",
    "GridPosition": "GridPosition",
    "Position": "Position",
    "Points": "Points",
}


def setup_logging() -> None:
    # Configure only this script's logger. FastF1 warns on every race that the
    # F1 live-timing API needs an account; results come from the Jolpica/Ergast
    # API anyway, so FastF1 is limited to errors to keep the output readable.
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def _no_f1_login():
    raise F1AuthError("F1 account sign-in is disabled in this project")


def disable_f1_login() -> None:
    """Never sign in to (or prompt for) a Formula 1 account.

    In an interactive terminal FastF1 asks for an F1 account email and password
    before calling the F1 live-timing API. This project only needs race and
    qualifying results, which come from the Jolpica/Ergast API without an
    account, so the sign-in is replaced by one that fails at once. FastF1 then
    continues without authentication, exactly as in a non-interactive run.
    Relies on FastF1 internals (fastf1==3.9.0 is pinned in requirements.txt).
    """
    try:
        from fastf1.internals import f1account
        for method in f1account.AUTH_METHODS:
            f1account._AUTHENTICATORS[method] = _no_f1_login
    except (ImportError, AttributeError):
        log.warning("Could not disable the F1 account sign-in for this FastF1 "
                    "version; if asked for an F1 email, press Ctrl+C.")


def setup_fastf1() -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.configure(cache_dir=str(CACHE_DIR))
    fastf1.set_log_level(logging.ERROR)
    disable_f1_login()


def fetch_race(year: int, round_no: int) -> pd.DataFrame:
    """Return one row per driver for the race session of (year, round_no)."""
    session = fastf1.get_session(year, round_no, "R")
    session.load(laps=False, telemetry=False, weather=False, messages=False)

    df = session.results[list(RESULT_COLUMNS)].rename(columns=RESULT_COLUMNS)
    df.insert(0, "year", year)
    df.insert(1, "round", round_no)
    df.insert(2, "event_name", session.event["EventName"])
    df.insert(3, "event_date", pd.Timestamp(session.event["EventDate"]).date())
    return df.reset_index(drop=True)


def completed_rounds(year: int) -> list[int]:
    """Round numbers of races whose date is already in the past."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    today = pd.Timestamp.now().normalize()
    done = schedule[schedule["EventDate"] < today]
    return [int(r) for r in done["RoundNumber"]]


def fetch_seasons(years: list[int], have: set[tuple[int, int]]) -> tuple[list[pd.DataFrame], bool]:
    """Fetch every completed round not already in `have`.

    Returns the new race frames and whether the run hit the API rate limit.
    A rate limit stops the run (every later call would fail too); any other
    error only skips that one round.
    """
    frames = []
    try:
        for year in years:
            rounds = completed_rounds(year)
            todo = [r for r in rounds if (year, r) not in have]
            log.info("%d: %d completed rounds, %d to fetch", year, len(rounds), len(todo))
            for round_no in todo:
                try:
                    race = fetch_race(year, round_no)
                except RateLimitExceededError:
                    raise
                except Exception as exc:  # skip a broken round, keep going
                    log.warning("skip %d round %d: %r", year, round_no, exc)
                    continue
                if race.empty or race["Position"].isna().all():
                    log.warning("skip %d round %d: no results", year, round_no)
                    continue
                frames.append(race)
                log.info("ok %d round %d %s (%d drivers)",
                         year, round_no, race["event_name"].iloc[0], len(race))
    except RateLimitExceededError as exc:
        log.error("API rate limit hit (%s). Saving what was fetched; "
                  "rerun this script in about an hour to fetch the rest.", exc)
        return frames, True
    return frames, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="store_true",
                        help="fetch one race (2024 round 1) and print it, save nothing")
    parser.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    args = parser.parse_args()

    setup_logging()
    setup_fastf1()

    if args.test:
        df = fetch_race(2024, 1)
        print(df.to_string())
        return

    # Incremental: keep rounds already saved, fetch only the missing ones.
    old = pd.read_csv(OUT_FILE) if OUT_FILE.exists() else pd.DataFrame()
    have = set(zip(old["year"], old["round"])) if not old.empty else set()
    log.info("%d rounds already in %s", len(have), OUT_FILE.relative_to(ROOT))

    new_frames, rate_limited = fetch_seasons(args.years, have)
    df = pd.concat([old, *new_frames], ignore_index=True)
    df = df.sort_values(["year", "round"], kind="stable").reset_index(drop=True)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"\nsaved {OUT_FILE.relative_to(ROOT)} (+{len(new_frames)} new rounds)")
    if rate_limited:
        print("INCOMPLETE: stopped by the API rate limit, rerun later.")
    print("shape:", df.shape)
    print(df.groupby("year")["round"].nunique().rename("races").to_string())
    print(df.head().to_string())


if __name__ == "__main__":
    main()
