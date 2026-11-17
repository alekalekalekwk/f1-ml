"""Train win-probability models with a time-based split and compare to the baseline.

Split (never random):
    train  2022-2024
    test   2025          (main test season)
    extra  2026 so far   (new regulations, kept as an honest extra test)

Metrics, per race:
    accuracy       the driver with the highest predicted win probability won
    top-3 accuracy the real winner is among the 3 highest-probability drivers

Usage:
    python src/train.py
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import baseline_scores
from features import FEATURES

ROOT = Path(__file__).resolve().parent.parent
FEATURES_FILE = ROOT / "data" / "features.csv"
TRAIN_YEARS = [2022, 2023, 2024]
TEST_YEARS = [2025, 2026]


def build_models() -> dict:
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)),
    }


def race_scores(df: pd.DataFrame, proba) -> dict:
    """Per-race accuracy and top-3 accuracy from predicted win probabilities."""
    df = df.assign(proba=proba)
    hits, top3_hits = 0, 0
    races = df.groupby(["year", "round"])
    for _, race in races:
        ranked = race.sort_values("proba", ascending=False, kind="stable")
        hits += int(ranked["won"].iloc[0] == 1)
        top3_hits += int(ranked["won"].head(3).sum() == 1)
    n = races.ngroups
    return {"races": n, "accuracy": hits / n, "top3_accuracy": top3_hits / n}


def main() -> None:
    df = pd.read_csv(FEATURES_FILE)
    train = df[df["year"].isin(TRAIN_YEARS)]
    print(f"train: {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}, {train.shape[0]} rows, "
          f"{train.groupby(['year', 'round']).ngroups} races, features: {FEATURES}\n")

    rows = []
    for year in TEST_YEARS:
        s = baseline_scores(df[df["year"] == year])
        rows.append({"model": "Baseline (pole wins)", "season": year, **s})

    for name, model in build_models().items():
        model.fit(train[FEATURES], train["won"])
        for year in TEST_YEARS:
            test = df[df["year"] == year]
            s = race_scores(test, model.predict_proba(test[FEATURES])[:, 1])
            rows.append({"model": name, "season": year, **s})

        if name == "LogisticRegression":
            coefs = model[-1].coef_[0]
            print("LogisticRegression coefficients (on standardized features):")
            for f, c in sorted(zip(FEATURES, coefs), key=lambda x: -abs(x[1])):
                print(f"  {f:<22}{c:+.2f}")
            print()

    table = pd.DataFrame(rows)
    for year in TEST_YEARS:
        t = table[table["season"] == year]
        print(f"== test season {year} ({int(t['races'].iloc[0])} races)")
        print(t[["model", "accuracy", "top3_accuracy"]].to_string(
            index=False, formatters={"accuracy": "{:.1%}".format,
                                     "top3_accuracy": "{:.1%}".format}))
        print()


if __name__ == "__main__":
    main()
