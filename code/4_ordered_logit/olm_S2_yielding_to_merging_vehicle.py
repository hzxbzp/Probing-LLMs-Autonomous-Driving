"""Ordered logit model of the 'yielding to a merging vehicle' scenario (Table 3 and Fig. 4 of the paper).
Estimates the final OLM specification on the standardised predictors with statsmodels' OrderedModel (logit link, BFGS),
prints the estimation summary (coefficients, standard errors, z, p, 95% CI, log-likelihood, AIC, BIC) and computes
average marginal effects by central finite differences (epsilon = 1e-4 on the standardised scale, Eq. 4).
Outputs in results/: S2_yielding_to_merging_vehicle_olm_summary.txt, S2_yielding_to_merging_vehicle_marginal_effects.csv and S2_yielding_to_merging_vehicle_marginal_effects.png."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # replication-package root
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# === 1. Data preparation ===
df = pd.read_csv(ROOT / "data/3_encoded_datasets/S2_yielding_to_merging_vehicle_encoded.csv")
print("Rows in the input data:", df.shape[0])
print("Missing llm_behavior_code values:", df["llm_behavior_code"].isna().sum())


X = df.drop(columns=[
    "id", "llm_behavior_code", 
    # "weather_rainy",
    "weather_sunny",
    "time_day",
    # "time_night",	
    "road_condition_dry",	
    # "road_condition_wet",
    
    # "merging_vehicle_speed",	
    "merging_gap",
    # "mainstream_front_space",	
    # "occupant_requirement",
    "ego_distance",
    "ego_velocity",
    "delta_t_diff_code",

    "ego_state_approaching",	
    # "merging_vehicle_type_emergency vehicle",	
    "merging_vehicle_type_normal car",
    "merging_vehicle_type_truck",
    # "ego_role_emergency vehicle",
    "ego_role_normal car",
    "ego_role_truck",
    "following_vehicle_emergency vehicle",	
    # "following_vehicle_none",
    "following_vehicle_normal car",

    # "is_english",
    # "is_chinese",
    "is_french",

    # "is_gpt",
    "is_llama",
    "is_deepseek",
    ])
print("Missing values in X:", X.isna().sum().sum())

y = df["llm_behavior_code"]
print("Rows in y:", y.shape[0])

# Standardise the predictors (effects are reported per one standard deviation)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print("Rows after standardisation:", X_scaled.shape[0])

# Use the full sample for estimation
X_train = X_scaled
y_train = y
print("Estimation sample size:", X_train.shape[0])

# === 2. Estimate the ordered logit model ===
model = OrderedModel(
    y_train,
    X_train,
    distr="logit"  # or "probit"
)

res = model.fit(method='bfgs')
print(res.summary())
with open(RESULTS / "S2_yielding_to_merging_vehicle_olm_summary.txt", "w", encoding="utf-8") as f:
    f.write(res.summary().as_text())

# === 3. Marginal effects (central finite differences, Eq. 4) ===
epsilon = 1e-4
probs = res.model.predict(res.params, exog=X_train)

effects = {0: [], 1: [], 2: []}

for i, col in enumerate(X_train.columns):
    X_plus = X_train.copy()
    X_minus = X_train.copy()
    X_plus[col] += epsilon
    X_minus[col] -= epsilon
    
    probs_plus = res.model.predict(res.params, exog=X_plus)
    probs_minus = res.model.predict(res.params, exog=X_minus)

    marginal_effect = (probs_plus - probs_minus) / (2 * epsilon)
    for k in range(3):  # classes 0, 1, 2
        effects[k].append(marginal_effect[:, k].mean())

# Collect the marginal effects in a DataFrame
mfx_df = pd.DataFrame({
    "variable": X_train.columns,
    "Assertive (y=0)": effects[0],
    "Cooperative (y=1)": effects[1],
    "Conservative (y=2)": effects[2],
})
mfx_df.to_csv(RESULTS / "S2_yielding_to_merging_vehicle_marginal_effects.csv", index=False)
print(mfx_df.to_string())

# Wide to long format for plotting
mfx_long = pd.melt(
    mfx_df,
    id_vars="variable",
    var_name="Category",
    value_name="Marginal Effect"
)

# === 4. Plot: four panels with a common y-axis range ===
palette = {
    "Assertive (y=0)": "#b6a8d3",
    "Cooperative (y=1)": "#f4b183",
    "Conservative (y=2)": "#b6e0d4"
}
hatch_patterns = {
    "Assertive (y=0)": "//",
    "Cooperative (y=1)": "xx",
    "Conservative (y=2)": ".."
}

# Common y-axis range for all panels
ymin = mfx_long["Marginal Effect"].min()
ymax = mfx_long["Marginal Effect"].max()

# Split the variables into four groups (one panel each)
n = len(mfx_df)
chunk_size = int(np.ceil(n / 4))
groups = [mfx_df.iloc[i:i+chunk_size] for i in range(0, n, chunk_size)]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
bar_width = 0.25

for idx, group_df in enumerate(groups):
    ax = axes[idx]
    subset_long = mfx_long[mfx_long["variable"].isin(group_df["variable"])]
    categories = subset_long["Category"].unique()
    x = np.arange(len(group_df))
    
    # Grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for i, cat in enumerate(categories):
        subdata = subset_long[subset_long["Category"] == cat]
        ax.bar(
            x + i * bar_width,
            subdata["Marginal Effect"],
            width=bar_width,
            label=cat if idx == 0 else "",
            color=palette[cat],
            hatch=hatch_patterns[cat],
            edgecolor="black",
            linewidth=0.8
        )
        # Value labels with three decimals
        offset = max(0.01, 0.07 * (ymax - ymin))  # automatic offset (works for mixed positive/negative values)

        last_label_y = {}

        for xi, val in zip(x + i * bar_width, subdata["Marginal Effect"]):
            # Base offset just above/below the bar
            base_offset = 0.006 * (ymax - ymin)
            text_y = val + base_offset if val >= 0 else val - base_offset

            # Avoid overlapping labels at the same x position
            if xi in last_label_y:
                prev_y = last_label_y[xi]
                if abs(prev_y - text_y) < 0.03:  # increase the spacing if labels are closer than 0.03
                    adjust = 0.02 * (ymax - ymin)
                    text_y = text_y + adjust if val >= 0 else text_y - adjust

            last_label_y[xi] = text_y

            ax.text(
                xi,
                text_y,
                f"{val:.3f}",
                ha='center',
                va='bottom' if val >= 0 else 'top',
                fontsize=16,
                fontweight="bold"
            )
    
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([""] * len(group_df))
    ax.tick_params(axis='y', labelsize=16)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(ymin - 0.002, ymax + 0.002)

# Shared axis labels
# fig.text(0.5, 0.04, 'Predictors', ha='center', fontsize=16)
fig.text(0.04, 0.5, 'Marginal effects (ΔP per unit increase)', va='center', rotation='vertical', fontsize=16)
axes[0].legend(fontsize=16, title_fontsize=16)

plt.tight_layout(rect=[0.05, 0.05, 1, 1])
plt.savefig(RESULTS / "S2_yielding_to_merging_vehicle_marginal_effects.png", dpi=300, bbox_inches="tight")
plt.show()

# === 5. Pearson correlation matrix of the predictors (optional) ===
# plt.figure(figsize=(12, 10))

# # Pearson correlation matrix of the standardised predictors
# corr_matrix = X_train.corr(method='pearson')

# # Draw the heat map
# sns.heatmap(
#     corr_matrix,
#     annot=True,           # show the coefficients
#     fmt=".2f",            # two decimals
#     cmap="coolwarm",      # diverging colour map
#     center=0,             # centred at zero
#     square=True,          # square cells
#     linewidths=0.5,       # cell borders
#     cbar_kws={"shrink": .75}
# )

# plt.title("Pearson Correlation Matrix of Predictors", fontsize=15, fontweight="bold")
# plt.xticks(rotation=45, ha='right', fontsize=11)
# plt.yticks(rotation=0, fontsize=11)
# plt.tight_layout()
# plt.show()
