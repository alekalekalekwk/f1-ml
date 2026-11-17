"""Local web UI for the F1 Race Winner Predictor.

Run:
    streamlit run src/app.py
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "raw" / "results.csv"
MODEL = ROOT / "data" / "model.joblib"
BAR_COLOR = "#2a78d6"

st.set_page_config(page_title="F1 Race Winner Predictor", page_icon="🏁", layout="wide")


@st.cache_data
def load_results(mtime: float) -> pd.DataFrame:
    """Load results.csv; `mtime` busts the cache when the file is refreshed."""
    df = pd.read_csv(RESULTS, parse_dates=["event_date"])
    df["won"] = df["Position"] == 1
    df["pole"] = df["GridPosition"] == 1
    return df


def grid_label(grid: float) -> str:
    if pd.isna(grid):
        return "-"
    return "Pit lane" if grid == 0 else str(int(grid))


def race_tab(race: pd.DataFrame) -> None:
    winner = race[race["won"]]
    pole = race[race["pole"]]
    w = winner.iloc[0] if not winner.empty else None
    p = pole.iloc[0] if not pole.empty else None

    c1, c2, c3 = st.columns(3)
    c1.metric("Winner", w["driver_name"] if w is not None else "-",
              w["team"] if w is not None else None, delta_color="off", delta_arrow="off")
    c2.metric("Pole position", p["driver_name"] if p is not None else "-",
              p["team"] if p is not None else None, delta_color="off", delta_arrow="off")
    if w is not None and p is not None:
        pole_won = w["driver"] == p["driver"]
        c3.metric("Pole → win?", "Yes" if pole_won else "No",
                  None if pole_won else f"won from grid {grid_label(w['GridPosition'])}",
                  delta_color="off", delta_arrow="off")

    table = race.sort_values("Position").assign(
        Grid=race["GridPosition"].map(grid_label),
        Gained=(race["GridPosition"] - race["Position"]).where(race["GridPosition"] > 0),
    )
    st.dataframe(
        table[["Position", "driver", "driver_name", "team", "Grid", "Gained", "Points"]],
        hide_index=True,
        width="stretch",
        height=35 * (len(table) + 1) + 3,
        column_config={
            "Position": st.column_config.NumberColumn("Pos", format="%d"),
            "driver": "Code",
            "driver_name": "Driver",
            "team": "Team",
            "Gained": st.column_config.NumberColumn("+/- places", format="%+d"),
            "Points": st.column_config.NumberColumn("Points", format="%g"),
        },
    )


def season_tab(season: pd.DataFrame, year: int) -> None:
    winners = season[season["won"]].sort_values("round")
    n_races = season["round"].nunique()
    pole_won = (winners["GridPosition"] == 1).mean() if len(winners) else float("nan")

    c1, c2, c3 = st.columns(3)
    c1.metric("Races in data", n_races)
    c2.metric("Different winners", winners["driver"].nunique())
    c3.metric("Won from pole", f"{pole_won:.0%}" if pd.notna(pole_won) else "-",
              help="How often the pole-sitter won. This is the baseline the model has to beat.")

    wins = (winners.groupby(["driver", "driver_name", "team"], as_index=False)
            .size().rename(columns={"size": "wins"})
            .sort_values("wins", ascending=False))
    st.subheader(f"Wins per driver, {year}")
    chart = (
        alt.Chart(wins)
        .mark_bar(color=BAR_COLOR, cornerRadiusEnd=4)
        .encode(
            x=alt.X("wins:Q", title="Wins", axis=alt.Axis(tickMinStep=1, grid=True)),
            y=alt.Y("driver_name:N", sort="-x", title=None,
                    scale=alt.Scale(paddingInner=0.4),
                    axis=alt.Axis(labelLimit=240, ticks=False, domain=False)),
            tooltip=[alt.Tooltip("driver_name:N", title="Driver"),
                     alt.Tooltip("team:N", title="Team"),
                     alt.Tooltip("wins:Q", title="Wins")],
        )
        .properties(height=alt.Step(32))  # 32px per driver row
    )
    st.altair_chart(chart, width="stretch")

    st.subheader("Race winners")
    st.dataframe(
        winners.assign(Grid=winners["GridPosition"].map(grid_label))[
            ["round", "event_name", "event_date", "driver_name", "team", "Grid"]],
        hide_index=True,
        width="stretch",
        column_config={
            "round": st.column_config.NumberColumn("Round", format="%d"),
            "event_name": "Grand Prix",
            "event_date": st.column_config.DateColumn("Date"),
            "driver_name": "Winner",
            "team": "Team",
            "Grid": "Started",
        },
    )


def prediction_tab() -> None:
    if not MODEL.exists():
        st.info(
            "No trained model yet. Predictions appear here once the model is trained "
            "(plan Steps 5–7: `src/train.py`, then `src/predict.py`)."
        )
        return
    st.success("Model found at data/model.joblib. Prediction view not built yet.")


def main() -> None:
    st.title("🏁 F1 Race Winner Predictor")

    if not RESULTS.exists():
        st.warning("No data yet. Run `python src/fetch_data.py` first.")
        st.stop()

    df = load_results(RESULTS.stat().st_mtime)

    with st.sidebar:
        st.header("Pick a race")
        years = sorted(df["year"].unique(), reverse=True)
        year = st.selectbox("Season", years)
        season = df[df["year"] == year]
        events = (season[["round", "event_name"]].drop_duplicates()
                  .sort_values("round", ascending=False))
        labels = {r: f"R{r} · {name}" for r, name in events.itertuples(index=False)}
        round_no = st.selectbox("Race", list(labels), format_func=labels.get)

        st.divider()
        coverage = df.groupby("year")["round"].nunique()
        st.caption("Races in data: " + ", ".join(f"{y}: {n}" for y, n in coverage.items()))

    race = season[season["round"] == round_no]
    st.caption(f"{labels[round_no]} · {race['event_date'].iloc[0]:%d %b %Y}")

    t_race, t_season, t_pred = st.tabs(["Race result", "Season", "Prediction"])
    with t_race:
        race_tab(race)
    with t_season:
        season_tab(season, year)
    with t_pred:
        prediction_tab()


main()
