"""Train win-probability models with a time-based split and compare to the baseline.

Split (never random):
    train  2022-2024
    test   2025          (main test season)
    extra  2026 so far   (new regulations, kept as an honest extra test)

Metrics, per race:
    accuracy       the driver with the highest predicted win probability won
    top-3 accuracy the real winner is among the 3 highest-probability drivers

The best model is chosen on 2025 only and saved to data/model.joblib.

Usage:
    python src/train.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baseline import baseline_scores
from features import FEATURES

ROOT = Path(__file__).resolve().parent.parent
FEATURES_FILE = ROOT / "data" / "features.csv"
MODEL_FILE = ROOT / "data" / "model.joblib"
TRAIN_YEARS = [2022, 2023, 2024]
TEST_YEARS = [2025, 2026]
SELECT_YEAR = 2025
BASELINE = "Baseline (pole wins)"


def build_models() -> dict:
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
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
        rows.append({"model": BASELINE, "season": year, **s})

    fitted = {}
    for name, model in build_models().items():
        model.fit(train[FEATURES], train["won"])
        fitted[name] = model
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
    wide = table.pivot(index="model", columns="season", values=["accuracy", "top3_accuracy"])
    wide.columns = [f"{'acc' if m == 'accuracy' else 'top3'} {y}" for m, y in wide.columns]
    wide = wide[[f"{m} {y}" for y in TEST_YEARS for m in ("acc", "top3")]]
    wide = wide.loc[[BASELINE, *fitted]]
    races = table.groupby("season")["races"].first()
    print("Comparison (per-race accuracy; races: "
          + ", ".join(f"{y}={races[y]}" for y in TEST_YEARS) + ")")
    print(wide.map("{:.1%}".format).to_string())

    # Pick the best model on 2025 only, so 2026 stays an honest extra test.
    sel = table[(table["season"] == SELECT_YEAR) & (table["model"] != BASELINE)]
    best = sel.sort_values(["accuracy", "top3_accuracy"], ascending=False).iloc[0]
    base = table[(table["season"] == SELECT_YEAR) & (table["model"] == BASELINE)].iloc[0]
    print(f"\nbest model on {SELECT_YEAR}: {best['model']} "
          f"(acc {best['accuracy']:.1%}, top-3 {best['top3_accuracy']:.1%}; "
          f"baseline acc {base['accuracy']:.1%}, top-3 {base['top3_accuracy']:.1%})")
    if best["accuracy"] < base["accuracy"]:
        print("note: no model beats the pole-position baseline on accuracy yet")

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": fitted[best["model"]],
        "name": best["model"],
        "features": FEATURES,
        "train_years": TRAIN_YEARS,
        "scores": table.to_dict("records"),
    }, MODEL_FILE)
    print(f"saved {MODEL_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
