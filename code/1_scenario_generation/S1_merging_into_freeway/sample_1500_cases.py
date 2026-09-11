"""Draw the 1,500 cases of the 'merging into freeway' scenario.
1,500 ids are sampled uniformly at random without replacement (random_state=42) from the English
full-factorial table, and the same ids are selected from the Chinese and French tables so that the three
language versions describe identical cases. Outputs: sample_1500_{en,cn,fr}.csv."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # replication-package root

from pathlib import Path
import pandas as pd

# ----------------------------------------------------------------------
# Input and output paths
# ----------------------------------------------------------------------
SRC_CSV = Path(ROOT / "data/1_scenario_configurations/S1_merging_into_freeway/all_scenarios_en.csv")             # English version
SRC_CSV2 = Path(ROOT / "data/1_scenario_configurations/S1_merging_into_freeway/all_scenarios_cn.csv")     # Chinese version
SRC_CSV3 = Path(ROOT / "data/1_scenario_configurations/S1_merging_into_freeway/all_scenarios_fr.csv")     # French version
DST_CSV1 = Path(ROOT / "data/1_scenario_configurations/S1_merging_into_freeway/sample_1500_en.csv")              # English sample output
DST_CSV2 = Path(ROOT / "data/1_scenario_configurations/S1_merging_into_freeway/sample_1500_cn.csv")              # Chinese sample output
DST_CSV3 = Path(ROOT / "data/1_scenario_configurations/S1_merging_into_freeway/sample_1500_fr.csv")              # French sample output


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
SAMPLE_SIZE = 1500
RANDOM_SEED = 42  # set to an int for reproducibility

# ----------------------------------------------------------------------
# 1) Load the three scenario tables
# ----------------------------------------------------------------------
df_en = pd.read_csv(SRC_CSV)
df_cn = pd.read_csv(SRC_CSV2)
df_fr = pd.read_csv(SRC_CSV3)

# ----------------------------------------------------------------------
# 2) Sample 1500 matching IDs from the English version
# ----------------------------------------------------------------------
id_sample = (
    df_en.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)["id"]
)

# ----------------------------------------------------------------------
# 3) Filter the three tables by the sampled ids
# ----------------------------------------------------------------------
df_sample_en = df_en[df_en["id"].isin(id_sample)].reset_index(drop=True)
df_sample_cn = df_cn[df_cn["id"].isin(id_sample)].reset_index(drop=True)
df_sample_fr = df_fr[df_fr["id"].isin(id_sample)].reset_index(drop=True)

# ----------------------------------------------------------------------
# 4) Save the three samples
# ----------------------------------------------------------------------
df_sample_en.to_csv(DST_CSV1, index=False)
df_sample_cn.to_csv(DST_CSV2, index=False, encoding='utf-8-sig')
df_sample_fr.to_csv(DST_CSV3, index=False, encoding='utf-8-sig')

print(f"Saved {len(df_sample_en)} English samples to '{DST_CSV1}'.")
print(f"Saved {len(df_sample_cn)} Chinese samples to '{DST_CSV2}'.")
print(f"Saved {len(df_sample_fr)} French samples to '{DST_CSV3}'.")
print("All samples saved successfully!")
