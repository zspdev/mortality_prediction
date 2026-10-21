"""Overview figure: load multiplier curves by country (exp. 03 and 04) and m(1.0) summary, in Hungarian and English.

Outputs reports/figures/overview_multiplier.png (hu) and overview_multiplier_en.png. Run:
``UV_PROJECT_ENVIRONMENT=~/.venvs/hcw_ml uv run python scripts/plot_overview.py``
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP, FIG = ROOT / "reports" / "experiments", ROOT / "reports" / "figures"
INK, MUTED, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_TIME, C_ERA = "#2a78d6", "#eb6834"  # categorical slots 1 and 2

t3 = pd.read_csv(EXP / "03_multiplier_by_panel.csv")
t4 = pd.read_csv(EXP / "04_multiplier_stock.csv")
# (key, data, group value, trend model, era model)
PANELS = [
    ("IT", t4, "IT-NUTS2", "load_time", "load_era"),
    ("BE", t3, "BE-NUTS1", "nb_time", "nb_era"),
    ("ES", t3, "ES-NUTS3", "nb_time", "nb_era"),
    ("RIO", t3, "BR-Rio", "nb_time", "nb_era"),
    ("HU", t4, "HU-country", "load_time", "load_era"),
]
WEAK = {"RIO", "HU"}

TEXT = {
    "hu": {
        "name": {"IT": "Olaszország", "BE": "Belgium", "ES": "Spanyolország", "RIO": "Rio de Janeiro", "HU": "Magyarország"},
        "note": {"IT": "erős hatás", "BE": "hatás, de csak 3 régió", "ES": "gyenge hatás",
                 "RIO": "1 város, bizonytalan", "HU": "nem informatív"},
        "trend": "időtrenddel", "era": "korszak-korrekcióval",
        "xlabel": "relatív kórházi terhelés (1 = a hely saját csúcsterhelése)", "ylabel": "halálozási szorzó",
        "s_xlabel": "szorzó teljes terhelésnél\n(pont: becslés, vonal: 90%-os sáv)",
        "s_title": "Összehasonlítás", "s_note": "az 1 azt jelenti: nincs hatás",
        "title": "Mennyivel nő a halálozási esély, ha megtelnek a kórházak?",
        "sub": "Halálozási szorzó a kórházi terhelés függvényében, országonként (1 = a halálozás nem nő).\n"
               "Kék: rugalmas időtrenddel, narancs: durva korszak-korrekcióval. A két módszer eltérése mutatja a bizonytalanságot.",
        "foot": "Forrás: saját modell (negatív binomiális, késleltetett). Olaszország és Magyarország a betegszám-alapú változat. "
                "A szorzó az egyes országokra külön becsült érték, nem oksági hatás.",
        "side_title": "A két módszer",
        "h_time": "Rugalmas időtrend",
        "b_time": "Minden régió saját, folyamatosan\nváltozó görbét kap az időre.\n"
                  "Ez leveszi a variánsok, az oltás\nés a kezelés hatását, ezért\n"
                  "csak a terhelés hatása marad.\n"
                  "Óvatosabb: a hatás kisebb lehet,\nde túlzottan el is nyelheti.",
        "h_era": "Durva korszak-korrekció",
        "b_era": "Három időszakra (2021. márc.\nelőtt, 2021. dec.-ig, utána) külön\nszint.\n"
                 "Nem követi a fokozatos változást,\nígy a korai, halálosabb hullámok\n"
                 "hatása a terhelésre kerülhet.\nA hatást felülbecsülheti.",
        "why": "Miért mutatjuk mindkettőt?\nAz igazság valahol a kettő között\n"
               "lehet. Ahol a két vonal közel van\n(Olaszország, Belgium), a hatás\nstabilabb; ahol eltér (Rio,\nMagyarország), nem megbízható.",
        "out": "overview_multiplier.png",
    },
    "en": {
        "name": {"IT": "Italy", "BE": "Belgium", "ES": "Spain", "RIO": "Rio de Janeiro", "HU": "Hungary"},
        "note": {"IT": "strong effect", "BE": "effect, but only 3 regions", "ES": "weak effect",
                 "RIO": "1 city, uncertain", "HU": "uninformative"},
        "trend": "time trend", "era": "period dummies",
        "xlabel": "relative hospital load (1 = the unit's own peak load)", "ylabel": "mortality multiplier",
        "s_xlabel": "multiplier at full load\n(dot: estimate, line: 90% band)",
        "s_title": "Comparison", "s_note": "1 means no effect",
        "title": "How much does mortality rise when hospitals fill up?",
        "sub": "Mortality multiplier as a function of hospital load, by country (1 = mortality does not rise).\n"
               "Blue: flexible time trend, orange: coarse period dummies. The gap between the two methods shows the uncertainty.",
        "foot": "Source: own model (negative binomial, lagged). Italy and Hungary use the census-based version. "
                "The multiplier is estimated separately for each country and is not a causal effect.",
        "side_title": "The two methods",
        "h_time": "Flexible time trend",
        "b_time": "Each region gets its own smoothly\nvarying curve over time.\n"
                  "This removes the effect of variants,\nvaccination and treatment, so\n"
                  "only the effect of load remains.\n"
                  "More cautious: the effect may be\nsmaller, but it can also be absorbed.",
        "h_era": "Coarse period dummies",
        "b_era": "A separate level for three periods\n(before Mar 2021, until Dec 2021,\nafter).\n"
                 "It does not follow gradual change,\nso the effect of the early, deadlier\n"
                 "waves can end up on load.\nIt may overestimate the effect.",
        "why": "Why show both?\nThe truth may lie somewhere between\n"
               "the two. Where the lines are close\n(Italy, Belgium) the effect is more\nstable; where they differ (Rio,\nHungary) it is not reliable.",
        "out": "overview_multiplier_en.png",
    },
}


def sel(df, key, model):
    col = "group" if "group" in df else "panel"
    return df[(df[col] == key) & (df["model"] == model)].sort_values("rel_load")


def draw(lang):
    T = TEXT[lang]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK, "figure.facecolor": BG, "axes.facecolor": BG})
    fig = plt.figure(figsize=(17, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.32, left=0.05, right=0.77, top=0.80, bottom=0.12)
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(5)]
    ax_s = fig.add_subplot(gs[1, 2])
    summary = []
    for ax, (k, df, key, m_t, m_e) in zip(axes, PANELS):
        weak = k in WEAK
        a, b = sel(df, key, m_t), sel(df, key, m_e)
        ax.axhline(1, color=MUTED, lw=0.8, ls=":")
        ax.fill_between(a.rel_load, a.lo90, a.hi90, color=C_TIME, alpha=0.10 if weak else 0.18, lw=0)
        ax.plot(a.rel_load, a.multiplier, color=C_TIME, lw=2.2, alpha=0.45 if weak else 1)
        ax.plot(b.rel_load, b.multiplier, color=C_ERA, lw=1.8, ls="--", alpha=0.45 if weak else 1)
        ax.set_ylim(0.5, 3.0); ax.set_xlim(0.05, 1.2)
        ax.set_title(T["name"][k], loc="left", fontsize=12, fontweight="bold", color=MUTED if weak else INK, pad=16)
        ax.text(0, 1.03, T["note"][k], transform=ax.transAxes, fontsize=9, color=MUTED)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel(T["xlabel"])
        for m, nm in ((a, T["trend"]), (b, T["era"])):
            at = lambda c: float(np.interp(1.0, m.rel_load, m[c]))
            summary.append((T["name"][k], nm, at("multiplier"), at("lo90"), at("hi90")))
        ax.annotate(f"{summary[-2][2]:.1f}×", (1.0, summary[-2][2]), xytext=(0.62, 2.55), fontsize=11, color=C_TIME,
                    alpha=0.6 if weak else 1, arrowprops=dict(arrowstyle="-", color=C_TIME, alpha=0.4))
    axes[0].set_ylabel(T["ylabel"]); axes[3].set_ylabel(T["ylabel"])
    for ax in (axes[1], axes[2], axes[4]):
        ax.tick_params(labelleft=False)

    # summary: m(1.0) with bands, per country
    S = pd.DataFrame(summary, columns=["country", "spec", "m", "lo", "hi"])
    order = [T["name"][p[0]] for p in PANELS][::-1]
    for spec, color, off in ((T["trend"], C_TIME, 0.12), (T["era"], C_ERA, -0.12)):
        d = S[S.spec == spec].set_index("country").loc[order]
        y = np.arange(len(order)) + off
        ax_s.hlines(y, d.lo, d.hi, color=color, lw=3, alpha=0.5)
        ax_s.plot(d.m, y, "o", color=color, ms=7, mec=BG, mew=1.5, label=spec)
    ax_s.axvline(1, color=MUTED, lw=0.8, ls=":")
    ax_s.set_yticks(range(len(order))); ax_s.set_yticklabels(order)
    ax_s.set_xlim(0.4, 2.4); ax_s.set_ylim(-1.0, len(order) - 0.4); ax_s.set_xlabel(T["s_xlabel"])
    ax_s.set_title(T["s_title"], loc="left", fontsize=12, fontweight="bold", pad=16)
    ax_s.text(0, 1.03, T["s_note"], transform=ax_s.transAxes, fontsize=9, color=MUTED)
    ax_s.grid(axis="x", color=GRID, lw=0.6); ax_s.set_axisbelow(True)
    for s in ("top", "right"):
        ax_s.spines[s].set_visible(False)
    ax_s.legend(frameon=False, fontsize=9, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(T["title"], x=0.05, y=0.965, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.05, 0.885, T["sub"], fontsize=10, color=MUTED, va="top", linespacing=1.5)
    fig.text(0.05, 0.02, T["foot"], fontsize=8.5, color=MUTED)

    # side panel: short description of the two methods
    side = fig.add_axes([0.79, 0.12, 0.2, 0.68]); side.axis("off")

    def block(y, color, head, lines):
        side.plot([0, 0.04], [y, y], color=color, lw=3, ls="-" if color == C_TIME else "--",
                  transform=side.transAxes, clip_on=False)
        side.text(0.07, y, head, color=INK, fontsize=11.5, fontweight="bold", va="center", transform=side.transAxes)
        side.text(0, y - 0.04, lines, color=MUTED, fontsize=9.5, va="top", linespacing=1.5, transform=side.transAxes)

    side.text(0, 1.0, T["side_title"], fontsize=13, fontweight="bold", va="top", transform=side.transAxes)
    block(0.90, C_TIME, T["h_time"], T["b_time"])
    block(0.52, C_ERA, T["h_era"], T["b_era"])
    side.text(0, 0.12, T["why"], fontsize=9.5, color=INK, va="top", linespacing=1.5, transform=side.transAxes,
              bbox=dict(boxstyle="round,pad=0.6", fc="#f1f0ec", ec="none"))
    fig.savefig(FIG / T["out"], dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for lang in ("hu", "en"):
        draw(lang)
