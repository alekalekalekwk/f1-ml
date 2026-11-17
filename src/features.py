"""Build model features from data/raw/results.csv and save data/features.csv.

One row per driver per race. Label: `won` (finished P1).
Features (all computed ONLY from races before the current one):
    grid              starting position; pit-lane start or missing = back of grid
    driver_form_rank  rank in this race's field by average finish over the
                      driver's last 5 races (1 = best recent form)
    team_form_rank    rank in this race's field by the team's average points
                      over its last 5 races (1 = strongest team right now)
    driver_wins_10    driver's wins in their last 10 races

Ranks are relative to the drivers in the same race, so they keep their meaning
when the pecking order or the points scale changes between eras. Recent wins
(last 10 races) instead of all-time wins stop past dominance from counting
forever. The raw values (driver_avg_finish_5, team_avg_points_5) stay in the
CSV for reference.

Usage:
    python src/features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN_FILE = ROOT / "data" / "raw" / "results.csv"
OUT_FILE = ROOT / "data" / "features.csv"
WINDOW = 5
WINS_WINDOW = 10

FEATURES = ["grid", "driver_form_rank", "team_form_rank", "driver_wins_10"]
RAW_FORM = ["driver_avg_finish_5", "team_avg_points_5"]

# Teams renamed across seasons map to one key so their form carries over.
TEAM_KEYS = {
    "Red Bull Racing": "red_bull",
    "Red Bull": "red_bull",
    "AlphaTauri": "racing_bulls",
    "RB F1 Team": "racing_bulls",
    "Alfa Romeo": "sauber_audi",
    "Sauber": "sauber_audi",
    "Audi": "sauber_audi",
    "Alpine": "alpine",
    "Alpine F1 Team": "alpine",
    "Haas F1 Team": "haas",
    "Cadillac F1 Team": "cadillac",
    "Aston Martin": "aston_martin",
    "Ferrari": "ferrari",
    "McLaren": "mclaren",
    "Mercedes": "mercedes",
    "Williams": "williams",
}


def team_key(name: str) -> str:
    if name not in TEAM_KEYS:
        print(f"WARNING unknown team name {name!r}: add it to TEAM_KEYS")
        return name.lower().replace(" ", "_")
    return TEAM_KEYS[name]


def load_results() -> pd.DataFrame:
    df = pd.read_csv(IN_FILE, parse_dates=["event_date"])
    df["team_key"] = df["team"].map(team_key)
    df["won"] = (df["Position"] == 1).astype(int)
    # Chronological order of races; ties inside a race keep finishing order.
    df["race_id"] = df["year"] * 100 + df["round"]
    return df.sort_values(["event_date", "race_id", "Position"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    field_size = df.groupby("race_id")["driver"].transform("size")

    # Grid: 0 means pit-lane start; NaN means did not start. Both -> back of grid.
    df["grid"] = df["GridPosition"].where(df["GridPosition"] > 0, field_size)

    # Driver form: shift(1) drops the current race, so only earlier races count.
    by_driver = df.groupby("driver", sort=False)
    df["driver_avg_finish_5"] = by_driver["Position"].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=1).mean())
    df["driver_wins_10"] = by_driver["won"].transform(
        lambda s: s.shift(1).rolling(WINS_WINDOW, min_periods=1).sum())

    # Team form: total team points per race, then the same shifted rolling mean.
    team_races = (df.groupby(["team_key", "race_id"], as_index=False)["Points"].sum()
                  .sort_values("race_id"))
    team_races["team_avg_points_5"] = team_races.groupby("team_key")["Points"].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=1).mean())
    df = df.merge(team_races[["team_key", "race_id", "team_avg_points_5"]],
                  on=["team_key", "race_id"], how="left")

    # No history yet (first race in the data, rookies, new teams): neutral defaults
    # that use no future information.
    df["driver_avg_finish_5"] = df["driver_avg_finish_5"].fillna((field_size + 1) / 2)
    df["team_avg_points_5"] = df["team_avg_points_5"].fillna(0.0)
    df["driver_wins_10"] = df["driver_wins_10"].fillna(0).astype(int)

    # Rank inside each race's field (1 = best). Uses only the pre-race values above.
    by_race = df.groupby("race_id")
    df["driver_form_rank"] = by_race["driver_avg_finish_5"].rank(method="min")
    df["team_form_rank"] = by_race["team_avg_points_5"].rank(method="min", ascending=False)
    return df


def leakage_check(df: pd.DataFrame, n: int = 300, seed: int = 0) -> int:
    """Recompute features for random rows using ONLY races dated before them.

    Returns the number of mismatches (0 means no leakage).
    """
    rng = np.random.default_rng(seed)
    rows = df.iloc[rng.choice(len(df), size=min(n, len(df)), replace=False)]
    mismatches = 0
    for _, row in rows.iterrows():
        past = df[df["event_date"] < row["event_date"]]
        d = past[past["driver"] == row["driver"]]
        exp_finish = d["Position"].tail(WINDOW).mean()
        exp_wins = int(d["won"].tail(WINS_WINDOW).sum())
        t = (past[past["team_key"] == row["team_key"]]
             .groupby("race_id")["Points"].sum().sort_index())
        exp_team = t.tail(WINDOW).mean() if len(t) else 0.0
        if np.isnan(exp_finish):  # no earlier race for this driver
            exp_finish = row["driver_avg_finish_5"]
        ok = (np.isclose(row["driver_avg_finish_5"], exp_finish)
              and row["driver_wins_10"] == exp_wins
              and np.isclose(row["team_avg_points_5"], exp_team))
        if not ok:
            mismatches += 1
            print("MISMATCH", row[["year", "round", "driver"]].to_dict())
    return mismatches


def main() -> None:
    df = add_features(load_results())
    cols = ["year", "round", "event_name", "event_date", "driver", "driver_name",
            "team", "team_key", *FEATURES, *RAW_FORM, "Position", "won"]
    out = df[cols]
    out.to_csv(OUT_FILE, index=False)

    print(f"saved {OUT_FILE.relative_to(ROOT)}  shape: {out.shape}")
    print("missing values in features:", int(out[FEATURES].isna().sum().sum()))
    print("\nexample (2026 São Paulo GP, top 5 finishers):")
    ex = out[(out["year"] == 2026) & (out["round"] == 20)].head(5)
    print(ex[["driver", "team_key", *FEATURES, "Position", "won"]].to_string(index=False))

    print("\nfeature means, winners vs others:")
    print(out.groupby("won")[FEATURES].mean().round(2).to_string())

    bad = leakage_check(df)
    print(f"\nleakage check: recomputed 300 random rows from earlier races only -> "
          f"{bad} mismatches")


if __name__ == "__main__":
    main()
