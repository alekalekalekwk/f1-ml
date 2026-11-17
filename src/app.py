"""Local web UI for the F1 Race Winner Predictor.

Run:
    streamlit run src/app.py
"""

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import sibling scripts

from export import export_prediction  # noqa: E402
from predict import (  # noqa: E402
    PredictionError,
    output_stem,
    predict_race,
    save_prediction,
    upcoming_rounds,
)

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


@st.cache_data(ttl=3600, show_spinner=False)
def cached_upcoming(year: int) -> pd.DataFrame:
    """Races of `year` not in the data yet; empty if the schedule can't be loaded."""
    try:
        return upcoming_rounds(year)
    except Exception:
        return pd.DataFrame(columns=["round", "event_name", "event_date"])


def prediction_tab(df: pd.DataFrame, year: int, round_no: int, label: str) -> None:
    if not MODEL.exists():
        st.info("No trained model yet. Run `python src/features.py` and "
                "`python src/train.py` first, then come back here.")
        return

    mode = st.radio("Which race?", ["Selected race", "Upcoming race"], horizontal=True,
                    help="Selected race: the race picked in the sidebar. "
                         "Upcoming race: a race of the current season that hasn't run yet.")
    if mode == "Selected race":
        st.caption(f"{label} · change it in the sidebar")
        use_quali = st.checkbox(
            "Use the qualifying order as grid",
            help="What the model would have said right after qualifying, before the race "
                 "(grid penalties are not included).")
        target = (year, round_no, use_quali)
    else:
        season = int(df["year"].max())
        upcoming = cached_upcoming(season)
        if upcoming.empty:
            st.info(f"No upcoming {season} races found.")
            return
        names = {r: f"R{r} · {n} · {pd.Timestamp(d):%d %b %Y}"
                 for r, n, d in upcoming.itertuples(index=False)}
        pick = st.selectbox("Upcoming race", list(names), format_func=names.get)
        st.caption("The grid comes from qualifying, so this works once qualifying "
                   "has finished for that weekend.")
        target = (season, pick, False)

    if st.button("Predict", type="primary"):
        y, r, quali = target
        with st.spinner("Building features and running the model…"):
            try:
                out, source, model_name = predict_race(y, r, quali)
            except PredictionError as exc:
                st.session_state.pop("prediction", None)
                st.warning(str(exc))
                return
            csv = save_prediction(out, y, r, quali)
            xlsx, png = export_prediction(out, y, output_stem(y, r, quali), quali)
        st.session_state["prediction"] = {
            "target": target, "out": out, "source": source, "model": model_name,
            "files": {"CSV": csv, "Excel": xlsx, "Chart (PNG)": png},
        }

    pred = st.session_state.get("prediction")
    if pred and pred["target"] == target:
        show_prediction(pred)


def show_prediction(pred: dict) -> None:
    out = pred["out"]
    top = out.iloc[0]
    st.subheader(f"{int(top['year'])} {top['event_name']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Favourite", top["driver_name"], top["team"],
              delta_color="off", delta_arrow="off")
    c2.metric("Win chance", f"{top['win_probability']:.1%}",
              f"starts P{int(top['grid'])}", delta_color="off", delta_arrow="off")
    winner = out[out["actual_position"] == 1]
    if not winner.empty:
        w = winner.iloc[0]
        c3.metric("Actual winner", w["driver_name"],
                  "model's top pick ✓" if w["predicted_rank"] == 1
                  else f"model ranked them #{int(w['predicted_rank'])}",
                  delta_color="off", delta_arrow="off")
    else:
        c3.metric("Grid from", pred["source"].capitalize(), f"model: {pred['model']}",
                  delta_color="off", delta_arrow="off")

    chart_df = out.assign(pct=out["win_probability"] * 100)
    chart = (
        alt.Chart(chart_df)
        .mark_bar(color=BAR_COLOR, cornerRadiusEnd=4)
        .encode(
            x=alt.X("win_probability:Q", title="Predicted win probability",
                    axis=alt.Axis(format="%", grid=True, tickCount=5)),
            y=alt.Y("driver_name:N", sort="-x", title=None,
                    scale=alt.Scale(paddingInner=0.35),
                    axis=alt.Axis(labelLimit=240, ticks=False, domain=False)),
            tooltip=[alt.Tooltip("driver_name:N", title="Driver"),
                     alt.Tooltip("team:N", title="Team"),
                     alt.Tooltip("grid:Q", title="Grid", format=".0f"),
                     alt.Tooltip("win_probability:Q", title="Win probability", format=".1%")],
        )
        .properties(height=alt.Step(26))
    )
    st.altair_chart(chart, width="stretch")
    st.caption(f"Model: {pred['model']} · grid from {pred['source']} · "
               "probabilities across the field add up to 100%")

    with st.expander("Full table"):
        st.dataframe(
            chart_df[["predicted_rank", "driver_name", "team", "grid", "pct", "actual_position"]],
            hide_index=True,
            width="stretch",
            column_config={
                "predicted_rank": st.column_config.NumberColumn("Rank", format="%d"),
                "driver_name": "Driver",
                "team": "Team",
                "grid": st.column_config.NumberColumn("Grid", format="%d"),
                "pct": st.column_config.NumberColumn("Win probability", format="%.1f%%"),
                "actual_position": st.column_config.NumberColumn("Finished", format="%d"),
            },
        )

    cols = st.columns(len(pred["files"]))
    mime = {".csv": "text/csv", ".png": "image/png",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
    for col, (name, path) in zip(cols, pred["files"].items()):
        col.download_button(f"Download {name}", data=path.read_bytes(), file_name=path.name,
                            mime=mime[path.suffix], width="stretch")


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
        prediction_tab(df, year, round_no, labels[round_no])


main()
