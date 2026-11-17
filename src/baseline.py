"""Baseline: "the driver starting from pole wins".

Evaluated separately on the 2025 season and the 2026 season so far.
The prediction per race is the driver with the best (lowest) grid position,
which is the pole-sitter unless they had to start from the pit lane.

Also reports a top-3 version (winner started in the top 3 of the grid) so it
can be compared with the model's top-3 accuracy in train.py.

Usage:
    python src/baseline.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FEATURES_FILE = ROOT / "data" / "features.csv"
TEST_YEARS = [2025, 2026]


def baseline_scores(df: pd.DataFrame) -> dict:
    """Accuracy and top-3 accuracy of 'best grid position wins' over the races in df."""
    races = df.groupby(["year", "round"])
    hits, top3_hits = 0, 0
    for _, race in races:
        ranked = race.sort_values("grid", kind="stable")
        hits += int(ranked["won"].iloc[0] == 1)
        top3_hits += int(ranked["won"].head(3).sum() == 1)
    n = races.ngroups
    return {"races": n, "accuracy": hits / n, "top3_accuracy": top3_hits / n}


def main() -> None:
    df = pd.read_csv(FEATURES_FILE)
    print("Baseline: the driver starting from pole wins\n")
    print(f"{'season':<8}{'races':>6}{'pole won':>10}{'accuracy':>10}{'top-3':>8}")
    for year in TEST_YEARS:
        s = baseline_scores(df[df["year"] == year])
        print(f"{year:<8}{s['races']:>6}{round(s['accuracy'] * s['races']):>10}"
              f"{s['accuracy']:>10.1%}{s['top3_accuracy']:>8.1%}")

    print("\nFor context, earlier seasons (training data in train.py):")
    for year in sorted(set(df["year"]) - set(TEST_YEARS)):
        s = baseline_scores(df[df["year"] == year])
        print(f"{year:<8}{s['races']:>6}{round(s['accuracy'] * s['races']):>10}"
              f"{s['accuracy']:>10.1%}{s['top3_accuracy']:>8.1%}")


if __name__ == "__main__":
    main()
