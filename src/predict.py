"""Predict each driver's win probability for one race.

Usage:
    python src/predict.py --year 2026 --round 20
    python src/predict.py --year 2026 --round 21                    # upcoming race
    python src/predict.py --year 2026 --round 20 --from-qualifying  # "after quali" view

Grid source:
    - race already in data/raw/results.csv -> its real starting grid
    - race not run yet (or --from-qualifying) -> qualifying order from FastF1
      (grid penalties applied after qualifying are not known here)
Features are always built from races BEFORE this one only.
Probabilities are normalized to sum to 1 across the field.
Output: outputs/prediction_<year>_<round>.csv
        (outputs/prediction_<year>_<round>_quali.csv with --from-qualifying)
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from features import add_features, load_results, team_key

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "data" / "model.joblib"
OUT_DIR = ROOT / "outputs"


class PredictionError(Exception):
    """A prediction cannot be made yet; the message is meant for the user."""


def qualifying_rows(year: int, round_no: int) -> pd.DataFrame:
    """Rows for a race built from its qualifying result (grid = quali position)."""
    import fastf1

    from fetch_data import setup_fastf1

    setup_fastf1()
    session = fastf1.get_session(year, round_no, "Q")
    try:
        session.load(laps=False, telemetry=False, weather=False, messages=False)
    except Exception as exc:
        raise PredictionError(
            f"Qualifying for {year} round {round_no} is not available yet ({exc}). "
            "Run this again after qualifying.") from exc
    res = session.results
    if res is None or res.empty or res["Position"].isna().all():
        raise PredictionError(f"No qualifying result yet for {year} round {round_no}. "
                              "Run this again after qualifying.")

    rows = pd.DataFrame({
        "year": year,
        "round": round_no,
        "event_name": session.event["EventName"],
        "event_date": pd.Timestamp(session.event["EventDate"]).normalize(),
        "driver": res["Abbreviation"].values,
        "driver_name": res["FullName"].values,
        "team": res["TeamName"].values,
        "GridPosition": res["Position"].values,
        "Position": float("nan"),
        "Points": 0.0,
    })
    rows["team_key"] = rows["team"].map(team_key)
    rows["won"] = 0
    rows["race_id"] = year * 100 + round_no
    return rows


def race_features(year: int, round_no: int, from_qualifying: bool) -> tuple[pd.DataFrame, str]:
    history = load_results()
    in_data = ((history["year"] == year) & (history["round"] == round_no)).any()

    if in_data and not from_qualifying:
        source = "race starting grid"
        full = history
    else:
        source = "qualifying order"
        history = history[~((history["year"] == year) & (history["round"] == round_no))]
        upcoming = qualifying_rows(year, round_no)
        # Only races before this one may feed its features.
        history = history[history["event_date"] < upcoming["event_date"].iloc[0]]
        full = pd.concat([history, upcoming], ignore_index=True)
        full = full.sort_values(["event_date", "race_id", "Position"]).reset_index(drop=True)

    feats = add_features(full)
    race = feats[(feats["year"] == year) & (feats["round"] == round_no)].copy()
    return race, source


def output_stem(year: int, round_no: int, from_qualifying: bool) -> str:
    return f"prediction_{year}_{round_no}" + ("_quali" if from_qualifying else "")


def predict_race(year: int, round_no: int,
                 from_qualifying: bool = False) -> tuple[pd.DataFrame, str, str]:
    """Win probability per driver, sorted best first.

    Returns (prediction table, grid source, model name).
    Raises PredictionError when there is no model or no grid yet.
    """
    if not MODEL_FILE.exists():
        raise PredictionError("No model found. Run `python src/train.py` first.")
    bundle = joblib.load(MODEL_FILE)

    race, source = race_features(year, round_no, from_qualifying)
    proba = bundle["model"].predict_proba(race[bundle["features"]])[:, 1]
    total = proba.sum()
    race["win_probability"] = proba / total if total > 0 else 1 / len(race)

    out = (race.sort_values("win_probability", ascending=False)
           .assign(predicted_rank=lambda d: range(1, len(d) + 1))
           [["predicted_rank", "year", "round", "event_name", "driver", "driver_name",
             "team", "grid", *[f for f in bundle["features"] if f != "grid"],
             "win_probability", "Position"]]
           .rename(columns={"Position": "actual_position"})
           .reset_index(drop=True))
    return out, source, bundle["name"]


def save_prediction(out: pd.DataFrame, year: int, round_no: int,
                    from_qualifying: bool) -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    out_file = OUT_DIR / f"{output_stem(year, round_no, from_qualifying)}.csv"
    out.to_csv(out_file, index=False)
    return out_file


def upcoming_rounds(year: int) -> pd.DataFrame:
    """Rounds of `year` that are not in the results data yet (round, event_name, event_date)."""
    import fastf1

    from fetch_data import setup_fastf1

    setup_fastf1()
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    history = load_results()
    done = set(history.loc[history["year"] == year, "round"])
    todo = schedule[~schedule["RoundNumber"].isin(done)]
    return pd.DataFrame({
        "round": todo["RoundNumber"].astype(int).values,
        "event_name": todo["EventName"].values,
        "event_date": pd.to_datetime(todo["EventDate"]).values,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_no")
    parser.add_argument("--from-qualifying", action="store_true",
                        help="use the qualifying order as grid even if the race is in the data")
    args = parser.parse_args()

    try:
        out, source, model_name = predict_race(args.year, args.round_no, args.from_qualifying)
    except PredictionError as exc:
        sys.exit(str(exc))
    out_file = save_prediction(out, args.year, args.round_no, args.from_qualifying)

    name = out["event_name"].iloc[0]
    print(f"{args.year} round {args.round_no}: {name}  (model: {model_name}, "
          f"grid from {source})\n")
    show = out.head(10)[["predicted_rank", "driver_name", "team", "grid",
                         "win_probability", "actual_position"]]
    show = show.assign(actual_position=show["actual_position"].map(
        lambda v: "-" if pd.isna(v) else f"{v:.0f}"))
    print(show.to_string(index=False, formatters={
        "win_probability": "{:.1%}".format,
        "grid": "{:.0f}".format}))
    print(f"\nprobabilities sum to {out['win_probability'].sum():.3f}")

    winner = out[out["actual_position"] == 1]
    if not winner.empty:
        w = winner.iloc[0]
        print(f"actual winner: {w['driver_name']} (model rank {w['predicted_rank']}, "
              f"{w['win_probability']:.1%})")
    print(f"saved {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
