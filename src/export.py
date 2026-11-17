"""Export a prediction CSV to a formatted Excel table and a bar chart image.

Usage:
    python src/export.py --year 2026 --round 20
    python src/export.py --year 2026 --round 20 --from-qualifying   # the _quali file

Reads   outputs/prediction_<year>_<round>[_quali].csv  (made by predict.py)
Writes  outputs/prediction_<year>_<round>[_quali].xlsx
        outputs/prediction_<year>_<round>[_quali].png
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"

BAR_COLOR = "#2a78d6"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
HEADER_FILL = "1F2A44"

TABLE_COLUMNS = {
    "predicted_rank": "Rank",
    "driver": "Code",
    "driver_name": "Driver",
    "team": "Team",
    "grid": "Grid",
    "win_probability": "Win probability",
    "actual_position": "Actual position",
}


def load_prediction(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"{path.relative_to(ROOT)} not found. Run predict.py for this race first.")
    return pd.read_csv(path).sort_values("win_probability", ascending=False)


def write_excel(df: pd.DataFrame, path: Path, title: str) -> None:
    table = df[list(TABLE_COLUMNS)].rename(columns=TABLE_COLUMNS)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="Prediction", index=False, startrow=2)
        ws = writer.sheets["Prediction"]

        ws["A1"] = title
        ws["A1"].font = Font(bold=True, size=14)
        header_row = 3
        for cell in ws[header_row]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor=HEADER_FILL)
            cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.freeze_panes = ws.cell(row=header_row + 1, column=1)

        cols = list(table.columns)
        prob_col = cols.index("Win probability") + 1
        first, last = header_row + 1, header_row + len(table)
        for r in range(first, last + 1):
            ws.cell(row=r, column=prob_col).number_format = "0.0%"
            for name in ("Rank", "Grid", "Actual position"):
                c = ws.cell(row=r, column=cols.index(name) + 1)
                c.number_format = "0"
                c.alignment = Alignment(horizontal="center")
            if ws.cell(row=r, column=cols.index("Actual position") + 1).value == 1:
                for c in ws[r]:
                    c.font = Font(bold=True)

        for i, name in enumerate(cols, start=1):
            longest = max([len(str(name))] + [len(str(v)) for v in table[name]])
            ws.column_dimensions[get_column_letter(i)].width = min(longest + 3, 40)
        ws.column_dimensions[get_column_letter(prob_col)].width = 16


def write_chart(df: pd.DataFrame, path: Path, title: str, subtitle: str) -> None:
    df = df.sort_values("win_probability")  # largest bar on top
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(df) + 1.6), dpi=150)
    bars = ax.barh(df["driver_name"], df["win_probability"], color=BAR_COLOR, height=0.6)

    # Label only bars big enough to matter, so the chart stays readable.
    for bar, p in zip(bars, df["win_probability"]):
        if p >= 0.01:
            ax.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height() / 2,
                    f"{p:.1%}", va="center", fontsize=8, color=TEXT_MUTED)

    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, max(df["win_probability"].max() * 1.15, 0.05))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0, labelsize=9, colors=TEXT)
    ax.tick_params(axis="x", labelsize=8, colors=TEXT_MUTED)
    ax.set_xlabel("Predicted win probability", fontsize=9, color=TEXT_MUTED)

    fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold", color=TEXT)
    ax.set_title(subtitle, loc="left", fontsize=9, color=TEXT_MUTED, pad=10)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def export_prediction(df: pd.DataFrame, year: int, stem: str,
                      from_qualifying: bool) -> tuple[Path, Path]:
    """Write outputs/<stem>.xlsx and outputs/<stem>.png; return both paths."""
    df = df.sort_values("win_probability", ascending=False)
    event = df["event_name"].iloc[0]
    title = f"Who wins the {year} {event}?"
    winner = df[df["actual_position"] == 1]
    subtitle = "Model win probability per driver"
    if from_qualifying:
        subtitle += ", grid from qualifying"
    if not winner.empty:
        subtitle += f"  ·  actual winner: {winner['driver_name'].iloc[0]}"

    OUT_DIR.mkdir(exist_ok=True)
    xlsx, png = OUT_DIR / f"{stem}.xlsx", OUT_DIR / f"{stem}.png"
    write_excel(df, xlsx, f"{year} {event}: predicted win probability")
    write_chart(df, png, title, subtitle)
    return xlsx, png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_no")
    parser.add_argument("--from-qualifying", action="store_true",
                        help="export the _quali prediction file")
    args = parser.parse_args()

    stem = f"prediction_{args.year}_{args.round_no}" + ("_quali" if args.from_qualifying else "")
    df = load_prediction(OUT_DIR / f"{stem}.csv")
    xlsx, png = export_prediction(df, args.year, stem, args.from_qualifying)

    print(f"saved {xlsx.relative_to(ROOT)}")
    print(f"saved {png.relative_to(ROOT)}")
    print("\noutputs/:")
    for f in sorted(OUT_DIR.iterdir()):
        if f.name != ".gitkeep":
            print(f"  {f.name:<34}{f.stat().st_size / 1024:>7.1f} KB")


if __name__ == "__main__":
    main()
