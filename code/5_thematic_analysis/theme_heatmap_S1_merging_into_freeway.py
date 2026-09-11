"""Theme x selected-option heat map of the English justifications in the 'merging into freeway' scenario
(Fig. 3 of the paper, one panel per model).
Options are harmonised to the three ordered classes, every class is down-sampled to the minimum class count
(random_state=42), each justification is matched against the theme keyword dictionary below, and the theme counts are
column-normalised to percentages (each option column sums to 100%).
Usage: python theme_heatmap_<scenario>.py [gpt4o|deepseek|llama]   Output: results/<scenario>_theme_heatmap_<model>.{png,csv}"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]  # replication-package root
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===== 1. Read the data =====
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt4o"   # gpt4o | deepseek | llama
INPUT_FILES = {
    "gpt4o":    ROOT / "data/2_llm_responses/S1_merging_into_freeway/gpt4o_en.csv",
    "deepseek": ROOT / "data/2_llm_responses/S1_merging_into_freeway/deepseek_v3_en.csv",
    "llama":    ROOT / "data/2_llm_responses/S1_merging_into_freeway/llama31_405b_en.csv",
}
df = pd.read_csv(INPUT_FILES[MODEL])
# ========== 2. Clean and merge the option labels ==========
df["selected_option"] = df["selected_option"].str.replace(r"\*\*", "", regex=True).str.strip()
df["selected_option"] = df["selected_option"].replace({
    "Cautious Merge": "Cooperative Merge (observe first, act later)"
})

# ===== 3. Theme keyword dictionary =====
theme_keywords = {
    "Environmental conditions": [
        "rainy", "wet", "slippery", "dry", "sunny", "night", "day", "visibility", "bad weather",
        "road surface", "weather", "dark", "poor condition", "foggy", "lighting","time", "wet road", "rainy weather", "road condition","nighttime conditions","nighttime visibility","weather wet","night conditions"
    ],
    "Surrounding traffic information": [
        "gap", "space", "distance", "merging", "bicycle", "cyclist", "pedestrian", "truck", "emergency vehicle",
        "main lane", "vehicle ahead", "vehicle behind", "crosswalk", "intersection", "speed", "yield", "right of way","road users", "speed km", "high speed","accident","lane traffic","following","traffic rules","traffic flow","35km h","30km h","60km h","100m", "ramp","lane vehicles'", "main","50km h","100km h","current conditions","surrounding","way","approaching","80km h","role","speed difference","free","dry road","50m","higher speed","situation","large gap","sufficiently large","relatively slow",
        "relatively low","clear","truck travelling"
    ],
    "Ego vehicle information": [
        "ego", "my vehicle", "speed", "position", "distance to ramp", "stopped", "brake", "signal", 
        "act first", "slow down", "accelerate", "wait", "stop", "go", "prepare", "adjust", "appraoch ensures","reach"," stopping distance","current speed"
    ],
    "Occupant requirement": [
        "urgent", "urgency", "hurry", "need to be fast", "do not delay", "time-sensitive", "balances urgency",
        "rushing", "medical", "high priority", "quick", "occupant", "passenger", "delivery", "deadline","hospital","passenger requirement","passenger urgency","balances urgency","require","passenger urgent","urgent requirement"
    ],
    "Safety": [
        "safety", "safe", "risk", "danger", "hazard", "collision", "avoid", "prevent", "secure",
        "protect", "caution", "alert", "careful", "avoid unnecessary", "merge safely","balances safety","unsafe maneuvers","emergency",
        "safe gap", "approach avoids","safety remains","risky","brake suddenly","safely","conditions safety","unnecessary risk","ensuring safety","priority especially", "priority","later approach","cooperative merge","observe","clear yielding",
        "merge proactively","proactive approach","waiting","yielding action","observing","intent early","avoids forcing","stopping distance","signaling intent","act later","safer","cautious approach","appropriate","causing abrupt","prioritize safety","balances assertiveness","suitable","cautious","brake abruptly","sufficient time","potential risk","lane safety","prioritize caution","safety takes","slow speed"
    ],
    "Mobility":[
        "merge efficiently", "km h","delay","approach minimizes","efficiency","unnecessary delay","efficiently","merging proactively",
        "observe later","proactively","act","assertive","forcing","disrupt","force","hiher speed","assertive approach","smooth","timely", "immediately","minimizing"
    ]
}

class_counts = df["selected_option"].value_counts()
print("Raw class counts:")
print(class_counts)

min_count = class_counts.min()
print(f"\nDown-sampling to the smallest class: {min_count} samples per option")

# Down-sample every option to the minimum class count
df_balanced = (
    df.groupby("selected_option", group_keys=False)
      .apply(lambda x: x.sample(min_count, random_state=42))
      .reset_index(drop=True)
)

print("\nClass counts after balancing:")
print(df_balanced["selected_option"].value_counts())

# ========== 4. Build the theme-option association matrix ==========
themes = list(theme_keywords.keys())
options = df_balanced["selected_option"].unique()
association_matrix = pd.DataFrame(0, index=themes, columns=options)

for theme, keywords in theme_keywords.items():
    # Build the regular-expression pattern
    pattern = '|'.join([r"\b" + kw + r"\b" for kw in keywords])
    for option in options:
        option_text = df_balanced[df_balanced["selected_option"] == option]["justification"]
        count = option_text.str.contains(pattern, regex=True).sum()
        association_matrix.loc[theme, option] = count

# ========== 5. Column-normalise (theme share within each option) ==========
association_pct_col = association_matrix.div(association_matrix.sum(axis=0), axis=1) * 100

# ========== 6. Draw the heat map ==========
plt.figure(figsize=(12, 6))

ax = sns.heatmap(
    association_pct_col,
    annot=True, fmt=".1f",
    cmap="coolwarm",
    cbar_kws={'label': 'Percentage (%)'},
    annot_kws={"size": 15, "weight": "bold","color": "white"}  # bold cell values
)

# Axis label fonts
plt.xlabel("Selected Option", fontsize=15, fontweight="bold")
plt.ylabel("Theme", fontsize=15, fontweight="bold")

# Tick label fonts
plt.xticks(rotation=45, ha='right', fontsize=15, fontweight="bold")
plt.yticks(fontsize=15, fontweight="bold")

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)        # tick label size
cbar.set_label('Percentage (%)', fontsize=15, fontweight='bold')

# Layout
plt.tight_layout()
association_pct_col.round(2).to_csv(RESULTS / f"S1_merging_into_freeway_theme_heatmap_{MODEL}.csv")
plt.savefig(RESULTS / f"S1_merging_into_freeway_theme_heatmap_{MODEL}.png", dpi=300, bbox_inches="tight")
plt.show()
