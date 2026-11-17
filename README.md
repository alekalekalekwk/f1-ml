# F1 Race Winner Predictor

Predicts each driver's chance of winning a Formula 1 race from FastF1 race data
(seasons 2022-2026), and exports the prediction as an Excel table and a chart.
A small local web app lets you browse results and seasons.

![Example prediction: 2026 São Paulo Grand Prix](docs/example_prediction.png)

## How it works

1. **Data**: race results (driver, team, grid, finish, points) for every race
   since 2022, pulled with [FastF1](https://docs.fastf1.dev/).
2. **Features**, built for each driver in each race from earlier races only:

   | Feature | Meaning |
   |---|---|
   | `grid` | starting position (pit-lane start = back of grid) |
   | `driver_form_rank` | rank in this race's field by average finish over the driver's last 5 races |
   | `team_form_rank` | rank in this race's field by the team's average points over its last 5 races |
   | `driver_wins_10` | the driver's wins in their last 10 races |

   Ranks are relative to the other drivers in the same race, so they keep their
   meaning when the pecking order changes (for example the 2026 regulations).
   Renamed teams are mapped to one key (Alfa Romeo → Sauber → Audi,
   AlphaTauri → RB, …) so team form carries over.
3. **Model**: LogisticRegression, RandomForest and GradientBoosting are trained on
   2022-2024 and tested on 2025 and on 2026 so far, never with a random split.
   The best model on 2025 is saved; 2026 stays an untouched extra test.
4. **Prediction**: the model's per-driver probabilities are normalized so they
   add up to 100% across the field.

## Setup

Requires Python 3.11+ (developed on 3.13).

```bash
cd ~/f1-ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

FastF1 caches downloads in `./cache`. Raw data goes to `data/raw/`; both are
ignored by git.

## Usage

Run from the project folder with the venv active, in this order:

```bash
python src/fetch_data.py                       # 1. download race results -> data/raw/results.csv
python src/features.py                         # 2. build features        -> data/features.csv
python src/baseline.py                         # 3. "pole wins" baseline for 2025 and 2026
python src/train.py                            # 4. train + compare models -> data/model.joblib
python src/predict.py --year 2026 --round 20   # 5. predict one race      -> outputs/prediction_2026_20.csv
python src/export.py  --year 2026 --round 20   # 6. Excel + chart         -> outputs/prediction_2026_20.xlsx/.png
streamlit run src/app.py                       # web app at http://localhost:8501
```

- `fetch_data.py` is incremental: it only downloads races not yet in
  `results.csv`. Use `python src/fetch_data.py --test` to try a single race.
- **Upcoming race**: after qualifying, run `predict.py` for that round; the grid
  is taken from the qualifying order. `--from-qualifying` does the same for a
  past race (writes a separate `_quali` file).
- **After a new race**: run steps 1, 2 and 4 again to refresh data and model.
- Every script has `--help`.

## Results

Per-race accuracy: the model's top pick won. Top-3: the real winner was among
the model's three most likely drivers. Baseline: "the driver on pole wins".

| Model | 2025 accuracy | 2025 top-3 | 2026 accuracy | 2026 top-3 |
|---|---|---|---|---|
| Baseline (pole wins) | **66.7%** | 95.8% | **70.0%** | 90.0% |
| LogisticRegression | 37.5% | 91.7% | 40.0% | 90.0% |
| **RandomForest** (saved) | 45.8% | **100.0%** | 60.0% | **95.0%** |
| GradientBoosting | 33.3% | 83.3% | 65.0% | 90.0% |

2025: 24 races; 2026: 20 races (up to the São Paulo GP). Trained on 2022-2024.

**Reading it:** the pole-sitter is still the best single guess for the winner.
Where the model helps is the shortlist: its top 3 contained the winner in every
2025 race and in 19 of 20 races in 2026, better than the top 3 on the grid.

## Known limitations

- **Small test sets.** One race is about 4-5% of a test season, so small
  differences between models are not meaningful.
- **Does not beat pole position on picking the exact winner.** The features
  cannot tell when the pole-sitter will fail to win (strategy, safety cars,
  reliability, weather, first-lap incidents).
- **Few signals.** No qualifying pace gaps, practice or sprint results, tyre
  strategy, weather, track type or reliability data.
- **Qualifying grid ignores penalties.** For upcoming races the grid is the
  qualifying order; grid penalties applied afterwards are not included.
- **Regulation changes.** The model is trained on 2022-2024; the 2026 rules
  reshuffled the field, so early-2026 form lags behind reality for a few races.
- **New team names** must be added to `TEAM_KEYS` in `src/features.py`
  (the script warns when it sees an unknown one).
- **API limits.** FastF1 allows about 500 API calls per hour. A full first
  download can hit it; `fetch_data.py` saves what it has and can simply be rerun
  later. No F1 account is needed: the scripts turn off FastF1's F1-account
  sign-in (it would otherwise ask for an email and password in a terminal), and
  race and qualifying results load from the Jolpica/Ergast API without one.

## Project layout

```
src/fetch_data.py   download race results with FastF1
src/features.py     leakage-free features + leakage check
src/baseline.py     pole-position baseline
src/train.py        time-based training, model comparison, saves best model
src/predict.py      win probabilities for one race
src/export.py       formatted .xlsx and .png chart
src/app.py          Streamlit web app
PLAN.md             the step-by-step build plan
```
