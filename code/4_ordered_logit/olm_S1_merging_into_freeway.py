"""Ordered logit model of the 'merging into freeway' scenario (Table 2 and Fig. 2 of the paper).
Estimates the final OLM specification on the standardised predictors with statsmodels' OrderedModel (logit link, BFGS),
prints the estimation summary (coefficients, standard errors, z, p, 95% CI, log-likelihood, AIC, BIC) and computes
average marginal effects by central finite differences (epsilon = 1e-4 on the standardised scale, Eq. 4).
Outputs in results/: S1_merging_into_freeway_olm_summary.txt, S1_merging_into_freeway_marginal_effects.csv and S1_merging_into_freeway_marginal_effects.png."""
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
df = pd.read_csv(ROOT / "data/3_encoded_datasets/S1_merging_into_freeway_encoded.csv")
print("Rows in the input data:", df.shape[0])
print("Missing llm_behavior_code values:", df["llm_behavior_code"].isna().sum())

# if want to add more features, comment the following lines
X = df.drop(columns=[
    "id", "llm_behavior_code", 
    "ego_state_stopping",
    "ego_state_approaching",
    # "weather_rainy",
    # "time_night",
    # "road_condition_wet",
    "weather_sunny",
    "time_day",
    "road_condition_dry",
    # "mainstream_front_space", 

    # "following_vehicle_none",
    "following_vehicle_emergency vehicle",
    "following_vehicle_normal car",
    "mainstream_vehicle_type_normal car",
    "mainstream_vehicle_type_truck",
    # "mainstream_vehicle_type_emergency vehicle",
    "ego_role_normal car",
    "ego_role_truck",
    # "ego_role_emergency vehicle",

    # "delta_t_diff_code", 
    # "occupant_requirement",

    "is_english",
    # "is_chinese",
    # "is_french",

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
with open(RESULTS / "S1_merging_into_freeway_olm_summary.txt", "w", encoding="utf-8") as f:
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
    "dy/dx (y=0(Aggressive))": effects[0],
    "dy/dx (y=1(Neutral))": effects[1],
    "dy/dx (y=2(Conservative))": effects[2],
})
mfx_df.to_csv(RESULTS / "S1_merging_into_freeway_marginal_effects.csv", index=False)
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
    "dy/dx (y=0(Aggressive))": "#b6a8d3",
    "dy/dx (y=1(Neutral))": "#f4b183",
    "dy/dx (y=2(Conservative))": "#b6e0d4"
}
hatch_patterns = {
    "dy/dx (y=0(Aggressive))": "//",
    "dy/dx (y=1(Neutral))": "xx",
    "dy/dx (y=2(Conservative))": ".."
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
        for xi, val in zip(x + i * bar_width, subdata["Marginal Effect"]):
            ax.text(
                xi,
                val + np.sign(val)*0.002,
                f"{val:.3f}",
                ha='center',
                va='bottom' if val >= 0 else 'top',
                fontsize=16,
                fontweight="bold"
            )
    
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(group_df["variable"], rotation=30, ha="right", fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(ymin - 0.002, ymax + 0.002)

# Shared axis labels
fig.text(0.5, 0.04, 'Predictors', ha='center', fontsize=16)
fig.text(0.04, 0.5, 'Marginal effects (ΔP per unit increase)', va='center', rotation='vertical', fontsize=16)
axes[0].legend(title="Predicted class", fontsize=16, title_fontsize=16)

plt.tight_layout(rect=[0.05, 0.05, 1, 1])
plt.savefig(RESULTS / "S1_merging_into_freeway_marginal_effects.png", dpi=300, bbox_inches="tight")
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
