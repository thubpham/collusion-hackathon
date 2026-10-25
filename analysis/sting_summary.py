import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ARMS = ["no_report", "leniency_start", "leniency_mid", "reward_mid"]
REPORT_ARMS = ["leniency_start", "leniency_mid", "reward_mid"]
PAIRS = [
    ("no_report", "leniency_start"),
    ("no_report", "leniency_mid"),
    ("no_report", "reward_mid"),
    ("leniency_start", "leniency_mid"),
    ("leniency_mid", "reward_mid"),
]
COLORS = {"reported": "#2a6f97", "discovered": "#c8553d", "none": "#b8b8b8"}


def load(run_dir):
    df = pd.read_csv(Path(run_dir) / "results.csv")
    for col in ["offered", "accepted", "reported", "first_reporter", "game_crashed"]:
        df[col] = df[col].astype(str).str.lower() == "true"
    return df


def individual_counts(df):
    offered = df[df["offered"]]
    eligible = offered[offered["report_available_rounds"].fillna(0) > 0]
    return offered, eligible


def game_table(df):
    games = df.groupby(["arm", "game_id"], as_index=False).agg(
        exposed_how=("exposed_how", "first"),
        any_acceptor=("accepted", "any"),
    )
    games["exposed_how"] = games["exposed_how"].fillna("none")
    return games


def count_rows(df):
    offered, eligible = individual_counts(df)
    games = game_table(df)
    rows = []
    for arm in ARMS:
        o = offered[offered["arm"] == arm]
        e = eligible[eligible["arm"] == arm]
        g = games[games["arm"] == arm]
        rows.append(("accepted_offer", "agent", arm, int(o["accepted"].sum()), len(o)))
        rows.append(("game_has_acceptor", "game", arm, int(g["any_acceptor"].sum()), len(g)))
        rows.append(("game_exposed_by_report", "game", arm, int((g["exposed_how"] == "reported").sum()), len(g)))
        rows.append(("game_caught_by_referee", "game", arm, int((g["exposed_how"] == "discovered").sum()), len(g)))
        rows.append(("game_exposed_any", "game", arm, int((g["exposed_how"] != "none").sum()), len(g)))
        if arm in REPORT_ARMS:
            ea = e[e["accepted"]]
            er = e[~e["accepted"]]
            rows.append(("reported_all_eligible", "agent", arm, int(e["reported"].sum()), len(e)))
            rows.append(("reported_acceptors", "agent", arm, int(ea["reported"].sum()), len(ea)))
            rows.append(("reported_refusers", "agent", arm, int(er["reported"].sum()), len(er)))
    return pd.DataFrame(rows, columns=["metric", "unit", "arm", "numerator", "denominator"])


def mean_rows(df):
    offered, _ = individual_counts(df)
    acceptors = offered[offered["accepted"]]
    specs = [
        ("pact_kept_rounds_acceptors", acceptors, "pact_kept_rounds"),
        ("penalty_points_acceptors", acceptors, "penalty_points"),
        ("final_score_acceptors", acceptors, "final_score"),
        ("final_score_refusers", offered[~offered["accepted"]], "final_score"),
        ("retaliation_reporters_victims", offered, "retaliation"),
    ]
    rows = []
    for metric, frame, col in specs:
        for arm in ARMS:
            vals = frame[frame["arm"] == arm][col].dropna()
            rows.append(
                {
                    "metric": metric,
                    "unit": "agent",
                    "arm": arm,
                    "mean": float(vals.mean()) if len(vals) else np.nan,
                    "sd": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                    "n": len(vals),
                }
            )
    return pd.DataFrame(rows)


def build_summary(df):
    counts = count_rows(df)
    counts["rate"] = counts["numerator"] / counts["denominator"].where(counts["denominator"] > 0)
    counts["rate_text"] = counts.apply(lambda r: f"{r.numerator}/{r.denominator}", axis=1)
    means = mean_rows(df)
    return pd.concat([counts, means], ignore_index=True, sort=False)


def fisher_tests(counts):
    rows = []
    for metric, sub in counts.groupby("metric"):
        by_arm = sub.set_index("arm")
        for a, b in PAIRS:
            if a not in by_arm.index or b not in by_arm.index:
                continue
            a_num, a_den = int(by_arm.loc[a, "numerator"]), int(by_arm.loc[a, "denominator"])
            b_num, b_den = int(by_arm.loc[b, "numerator"]), int(by_arm.loc[b, "denominator"])
            if a_den == 0 or b_den == 0:
                continue
            odds, p = fisher_exact([[a_num, a_den - a_num], [b_num, b_den - b_num]])
            rows.append(
                {
                    "metric": metric,
                    "unit": by_arm.loc[a, "unit"],
                    "arm_a": a,
                    "arm_b": b,
                    "a_count": f"{a_num}/{a_den}",
                    "b_count": f"{b_num}/{b_den}",
                    "odds_ratio": odds,
                    "p_value": p,
                }
            )
    return pd.DataFrame(rows)


def label_bars(ax, bars, texts):
    for bar, text in zip(bars, texts):
        ax.annotate(
            text,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 2),
            textcoords="offset points",
        )


def plot_exposure(df, out):
    games = game_table(df)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bottom = np.zeros(len(ARMS))
    for how, label in [("reported", "Reported by a member"), ("discovered", "Caught by referee"), ("none", "Not exposed")]:
        vals = np.array([int((games[(games["arm"] == a)]["exposed_how"] == how).sum()) for a in ARMS])
        bars = ax.bar(ARMS, vals, bottom=bottom, color=COLORS[how], label=label)
        for bar, v, b in zip(bars, vals, bottom):
            if v:
                ax.text(bar.get_x() + bar.get_width() / 2, b + v / 2, str(v), ha="center", va="center", color="white", fontsize=9)
        bottom += vals
    ax.set_ylabel("Games (of 10)")
    ax.set_title("How the pact ended up exposed")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_report_rate(df, out):
    _, eligible = individual_counts(df)
    groups = [("Acceptors", True, "#2a6f97"), ("Refusers", False, "#e0a458")]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    width = 0.36
    x = np.arange(len(REPORT_ARMS))
    for i, (name, flag, color) in enumerate(groups):
        rates, texts = [], []
        for arm in REPORT_ARMS:
            sub = eligible[(eligible["arm"] == arm) & (eligible["accepted"] == flag)]
            n = len(sub)
            k = int(sub["reported"].sum())
            rates.append(k / n if n else 0)
            texts.append(f"{k}/{n}")
        bars = ax.bar(x + (i - 0.5) * width, rates, width, color=color, label=name)
        label_bars(ax, bars, texts)
    ax.set_xticks(x)
    ax.set_xticklabels(REPORT_ARMS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share who reported")
    ax.set_title("Report rate among eligible agents")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_pact_length(df, out):
    offered, _ = individual_counts(df)
    acceptors = offered[offered["accepted"]]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    rng = np.random.default_rng(0)
    for i, arm in enumerate(ARMS):
        vals = acceptors[acceptors["arm"] == arm]["pact_kept_rounds"].dropna().to_numpy()
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(vals)), vals, color="#2a6f97", alpha=0.45, s=18)
        if len(vals):
            ax.hlines(vals.mean(), i - 0.25, i + 0.25, color="#c8553d", linewidth=2)
            ax.text(i, vals.max() + 0.6, f"mean {vals.mean():.1f}, n={len(vals)}", ha="center", fontsize=8)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels(ARMS)
    ax.set_ylabel("Rounds the acceptor kept the pact")
    ax.set_title("How long acceptors kept the pact")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_report_timing(df, out):
    reports = df[df["reported"]]
    timings = [("before_use", "Before ever using the pact", "#2a6f97"), ("during", "While using the pact", "#c8553d")]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bottom = np.zeros(len(REPORT_ARMS))
    for key, label, color in timings:
        vals = np.array([int(((reports["arm"] == a) & (reports["report_timing"] == key)).sum()) for a in REPORT_ARMS])
        bars = ax.bar(REPORT_ARMS, vals, bottom=bottom, color=color, label=label)
        for bar, v, b in zip(bars, vals, bottom):
            if v:
                ax.text(bar.get_x() + bar.get_width() / 2, b + v / 2, str(v), ha="center", va="center", color="white", fontsize=9)
        bottom += vals
    ax.set_ylabel("Reports filed")
    ax.set_title("When members reported")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run)
    df = load(run_dir)
    crashed = int(df.loc[df["game_crashed"], "game_id"].nunique())
    print(f"crashed games kept out of the analysis: {crashed}")
    df = df[~df["game_crashed"]]

    summary = build_summary(df)
    summary.to_csv(run_dir / "summary.csv", index=False)
    tests = fisher_tests(summary[summary["numerator"].notna()])
    tests.to_csv(run_dir / "fisher_tests.csv", index=False)

    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)
    plot_exposure(df, plots / "exposure_by_arm.png")
    plot_report_rate(df, plots / "report_rate_by_arm.png")
    plot_pact_length(df, plots / "pact_length_by_arm.png")
    plot_report_timing(df, plots / "report_timing_by_arm.png")

    print(f"summary.csv: {len(summary)} rows; fisher_tests.csv: {len(tests)} rows; plots: 4")


if __name__ == "__main__":
    main()
