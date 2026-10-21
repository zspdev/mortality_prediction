# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
# ---

# %% [markdown]
# # 01 – Exploratory data analysis
# See `PLAN.md` §3 for the questions this notebook answers.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from hcw_ml.data import load_catalogue, load_long, to_wide

pd.set_option("display.width", 160)

# %% [markdown]
# ## 1. Structure and coverage

# %%
df = load_long()
cat = load_catalogue()
print(df.shape)
df.head()

# %%
coverage = (
    df.groupby(["country_iso2", "geo_level", "indicator"], observed=True)
    .agg(units=("geo_code", "nunique"), start=("date", "min"), end=("date", "max"), n=("value", "size"),
         missing=("value", lambda s: s.isna().mean()), flagged=("value_flag", lambda s: s.notna().sum()))
)
coverage

# %% [markdown]
# ## 2. Target candidates: deaths per 100k / per admission, by country

# %%
deaths = df[(df.indicator == "deaths_new") & (df.series_role == "primary")]
weekly = (deaths.set_index("date").groupby(["country_iso2", "geo_level"], observed=True)["value"]
          .resample("W").sum().reset_index())
fig, ax = plt.subplots(figsize=(11, 4))
for (c, g), s in weekly.groupby(["country_iso2", "geo_level"], observed=True):
    ax.plot(s["date"], s["value"], label=f"{c}-{g}")
ax.set_yscale("log"); ax.legend(ncol=3, fontsize=7); ax.set_title("Weekly deaths (all units summed)")
plt.show()

# %% [markdown]
# ## 3. Weekly panel and targets (`hcw_ml.features`)
# Target B (`cfr_flow`: deaths wk+1..+2 / admissions wk-1..0) for ES, BE, Rio; stock version B' (`cfr_stock`:
# deaths wk+1 / census) for IT and HU too. Relative load = census / the unit's own p95.

# %%
from hcw_ml.features import add_relative_load, build_panel

panel = add_relative_load(build_panel())
panel.groupby("panel").agg(units=("unit", "nunique"), weeks=("week", "size"), start=("week", "min"),
                           end=("week", "max"), flow=("cfr_flow", "count"), stock=("cfr_stock", "count"))

# %% [markdown]
# ## 4. Crude load gradient: pooled deaths per admission by relative-load quintile
# Ratio of sums (not mean of ratios), so small units do not dominate. Descriptive only: no unit, period or
# case-mix control, so this is the starting point for the Poisson model, not a result.

# %%
def gradient(d, num, den, load="rel_load", q=5):
    d = d.dropna(subset=[num, den, load]).query(f"{den} > 0").copy()
    d["bin"] = pd.qcut(d[load], q, labels=False, duplicates="drop") + 1
    return d.groupby("bin").agg(weeks=(num, "size"), deaths=(num, "sum"), denom=(den, "sum"), load=(load, "median")).assign(
        cfr=lambda x: x.deaths / x.denom)

for name, g in panel.groupby("panel"):
    if g["cfr_flow"].notna().any():
        print(name, "- deaths per admission"); print(gradient(g, "deaths_lead12", "adm_lag01").round(3), "\n")

# %%
for name, g in panel.groupby("panel"):
    print(name, "- deaths per patient-week"); print(gradient(g.assign(pw=g["census"] * 7), "deaths_lead1", "census").round(3), "\n")

# %% [markdown]
# ## 5. TODO
# - Negative / flagged values; reporting-day artefacts (HU); wave structure and period shifts
# - Lead-lag cross-correlation of admissions/census vs deaths per unit
# - Poisson/NB model with log-admissions offset, unit and period effects (PLAN.md §6)
