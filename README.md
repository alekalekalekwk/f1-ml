# PLAN.md: F1 Race Winner Predictor

## RULE 0 (REPO LOCK)
Work ONLY inside ~/f1-ml. Before EVERY step run `pwd` and `git remote -v` and confirm both are correct. If not, STOP and report.

## RULES
- Write ONE file per tool call. Never write big files at once.
- Always do a full read_file before editing any existing file.
- Never say "done" without evidence. Evidence for every sub-task:
  1. the last ~15 lines of the output of the script you ran (or the error)
  2. `git log --oneline -3`
  3. `git status --short`
- ONE commit per sub-task, conventional message (feat:, chore:, docs:).
- After each sub-task: report evidence, then STOP and wait for me to say "next".
- Do not push to GitHub until I say "push".
- Do not install anything outside the project venv.

## GOAL
Python project that uses FastF1 race data to predict the probable winner of an F1 race, and exports the result as a spreadsheet (.xlsx) and a chart image (.png).
Stack: Python 3, fastf1, pandas, scikit-learn, matplotlib, openpyxl. Data seasons: 2022-2026 (2026 is in progress; it brings new technical regulations and new teams, so treat it as a separate test).

## STEPS

### Step 1: Skeleton
Create venv instructions, .gitignore (venv/, cache/, __pycache__/, .DS_Store, data/raw/), requirements.txt, folders src/ data/ outputs/ (with .gitkeep). Install deps in venv and show `pip list | grep -i -E "fastf1|pandas|scikit|matplotlib|openpyxl"`.

### Step 2: Fetch data
src/fetch_data.py: using FastF1 with cache enabled in ./cache, pull race results (driver, team, GridPosition, Position, Points, year, round, event name). First test on ONE race only and show the output. Then run for 2022-2026 (only rounds whose race date is in the past), skip failed rounds with a logged warning, save to data/raw/results.csv. Show df.shape and head().

### Step 3: Features
src/features.py: read results.csv, build the label `won` (Position == 1) and features: GridPosition, driver's average finish over last 5 races, team's average points over last 5 races, driver's win count in prior races. NO DATA LEAKAGE: every feature must use only races BEFORE the current one. Team names change across seasons (e.g. Sauber -> Audi in 2026): map them to one team key so team form carries over correctly; brand-new teams (e.g. Cadillac) start with no history. Save to data/features.csv. Show shape and a leakage check explanation.

### Step 4: Baseline
src/baseline.py: baseline = "pole position (grid 1) wins". Evaluate separately on 2025 races and on 2026 races so far, and print how often it is right for each.

### Step 5: Train v1
src/train.py: LogisticRegression, TIME-BASED split (train 2022-2024, test 2025, extra test 2026 so far), never random. Print accuracy AND top-3 accuracy (is the real winner within the 3 drivers with highest predicted probability per race) for each test season. Compare to the baseline.

### Step 6: Model comparison
Add RandomForest and GradientBoosting to train.py (still one model per tool call when editing). Print a comparison table of all models vs baseline for 2025 and 2026. Pick the best model on 2025 (not 2026, to keep 2026 an honest extra test). Save the best model with joblib to data/model.joblib.

### Step 7: Predict a race
src/predict.py: CLI args --year --round. Load model, compute win probability for each driver in that race, normalize so probabilities sum to 1, save outputs/prediction_<year>_<round>.csv. If the race has not been run yet (upcoming 2026 rounds), take GridPosition from that weekend's qualifying session and build features from all races before it.

### Step 8: Export
src/export.py: from the prediction CSV, produce (a) outputs/prediction_<year>_<round>.xlsx with a clean formatted table (header style, column widths, percent format, sorted desc) and (b) outputs/prediction_<year>_<round>.png horizontal bar chart of win probability per driver with title. Show the file list in outputs/.

### Step 9: README
README.md: project purpose, setup, how to run each script in order, results table, known limitations.

### Step 10: Final check
Run the full pipeline from a clean clone-like state (fresh venv not required, just rerun scripts) for one race, confirm outputs are generated, show `git log --oneline`. STOP and wait for my "push".
