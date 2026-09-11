"""Build the estimation dataset of the 'merging into freeway' scenario.
The 1,500 cases are split into nine subsets (3 LLMs x 3 prompt languages, ~167 cases each); the contextual factors
of every case are encoded (numeric levels, ordinal occupant urgency, one-hot categorical attributes) and merged with the
ordered behaviour code of the option selected by the assigned model (1 = aggressive, 2 = neutral, 3 = conservative)
plus model / language indicators. Output: data/3_encoded_datasets/S1_merging_into_freeway_encoded.csv."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # replication-package root

import pandas as pd
import numpy as np

# 1. Read the English scenario configuration (1,500 cases)
df = pd.read_csv(ROOT / "data/1_scenario_configurations/S1_merging_into_freeway/sample_1500_en.csv")

# 2. Read the LLM responses (9 model x language cells, ~167 cases each = 1,500 cases) and tag language and model
gpt_en = pd.read_csv(ROOT / "data/2_llm_responses/S1_merging_into_freeway/gpt4o_en.csv", encoding='utf-8-sig').iloc[:167]
gpt_cn = pd.read_csv(ROOT / "data/2_llm_responses/S1_merging_into_freeway/gpt4o_cn.csv", encoding='utf-8-sig').iloc[167:334]
gpt_fr = pd.read_csv(ROOT / "data/2_llm_responses/S1_merging_into_freeway/gpt4o_fr.csv", encoding='utf-8-sig').iloc[334:500]

dp_en = pd.read_csv(ROOT / "data/2_llm_responses/S1_merging_into_freeway/deepseek_v3_en.csv", encoding='utf-8-sig').iloc[500:667]
dp_cn = pd.read_csv(ROOT / "data/2_llm_responses/S1_merging_into_freeway/deepseek_v3_cn.csv", encoding='utf-8-sig').iloc[667:834]
dp_fr = pd.read_csv(ROOT / "data/2_llm_responses/S1_merging_into_freeway/deepseek_v3_fr.csv", encoding='utf-8-sig').iloc[834:1000]

lla_en = pd.read_csv(ROOT / "data/2_llm_responses/S1_merging_into_freeway/llama31_405b_en.csv", encoding='utf-8-sig').iloc[1000:1167]
lla_cn = pd.read_csv(ROOT / "data/2_llm_responses/S1_merging_into_freeway/llama31_405b_cn.csv", encoding='utf-8-sig').iloc[1167:1334]
lla_fr = pd.read_csv(ROOT / "data/2_llm_responses/S1_merging_into_freeway/llama31_405b_fr.csv", encoding='utf-8-sig').iloc[1334:1500]

# Add language and model indicator columns
def add_flags(df, model, lang):
    df = df.copy()
    df['is_gpt'] = model == 'gpt'
    df['is_deepseek'] = model == 'deepseek'
    df['is_llama'] = model == 'llama'
    df['is_english'] = lang == 'en'
    df['is_chinese'] = lang == 'cn'
    df['is_french'] = lang == 'fr'
    return df

gpt_en = add_flags(gpt_en, 'gpt', 'en')
gpt_cn = add_flags(gpt_cn, 'gpt', 'cn')
gpt_fr = add_flags(gpt_fr, 'gpt', 'fr')
dp_en = add_flags(dp_en, 'deepseek', 'en')
dp_cn = add_flags(dp_cn, 'deepseek', 'cn')
dp_fr = add_flags(dp_fr, 'deepseek', 'fr')
lla_en = add_flags(lla_en, 'llama', 'en')
lla_cn = add_flags(lla_cn, 'llama', 'cn')
lla_fr = add_flags(lla_fr, 'llama', 'fr')

# Concatenate the responses and align them with the scenario table by id
llm_all = pd.concat([gpt_en, gpt_cn, gpt_fr, dp_en, dp_cn, dp_fr, lla_en, lla_cn, lla_fr], ignore_index=True)
llm_all['id'] = llm_all['id'].astype(int)
df['id'] = df['id'].astype(int)
llm_all = llm_all.set_index('id').loc[df['id']].reset_index()

# ---------- Encode the scenario configuration ----------
df['mainstream_front_space'] = df['mainstream_front_space'].replace({
    'free': 300, '20m': 20, '50m': 50, '100m': 100
})

def extract_numeric(series, unit):
    return series.str.replace(unit, '', regex=False).astype(int)

df['mainstream_vehicle_speed_val'] = extract_numeric(df['mainstream_vehicle_speed'], 'km/h')
df['mainstream_gap_val'] = extract_numeric(df['mainstream_gap'], 'm')
df['ego_distance_val'] = extract_numeric(df['ego_distance'], 'm')
df['ego_velocity_val'] = extract_numeric(df['ego_velocity'], 'km/h')

def kmh_to_mps(kmh): return kmh * 1000 / 3600

ego_v_mps = kmh_to_mps(df['ego_velocity_val'])
mainstream_v_mps = kmh_to_mps(df['mainstream_vehicle_speed_val'])

ego_delta_t = np.where(ego_v_mps == 0, 999, df['ego_distance_val'] / ego_v_mps)
mainstream_delta_t = df['mainstream_front_space'] / mainstream_v_mps

df['delta_t_diff'] = (mainstream_delta_t - ego_delta_t).round().astype(int)

def encode_t_diff(diff):
    if diff <= -2:
        return 1
    elif -2 < diff <= 2:
        return 2
    else:
        return 3

df['delta_t_diff_code'] = df['delta_t_diff'].apply(encode_t_diff)

df = df.drop(columns=['delta_t_diff'])
df = df.drop(columns=[
    'mainstream_vehicle_speed', 'mainstream_gap',
    'ego_distance', 'ego_velocity',
    'mainstream_vehicle_speed_val', 'mainstream_gap_val',
    'ego_distance_val', 'ego_velocity_val',
])

df['occupant_requirement'] = df['occupant_requirement'].map({
    'none': 0,
    'hurry (commute or flight)': 1,
    'urgent (to hospital)': 2
})
df['following_vehicle'] = df['following_vehicle'].fillna('none')

categorical_cols = [
    'weather', 'time', 'road_condition', 'mainstream_vehicle_type',
    'ego_role', 'following_vehicle', 'ego_state'
]
df_encoded = pd.get_dummies(df.drop(columns=['id']), columns=categorical_cols)
df_encoded.insert(0, 'id', df['id'])

# ---------- Normalise the selected option ----------
def normalize_behavior(text):
    text = str(text).lower()
    
    if "果断" in text or "assertive" in text or "insertion assertive" in text:
        return "Assertive Merge"
    elif ("先观察" in text or "observe first" in text or
          "insertion coopérative (observer d’abord" in text):
        return "Cooperative Merge (observe first)"
    elif ("先行动" in text or "act first" in text or
          "insertion coopérative (agir d’abord" in text):
        return "Cooperative Merge (act first)"
    elif "谨慎" in text or "cautious" in text or "insertion prudente" in text:
        return "Cautious Merge"
    else:
        return "Unknown"

llm_all['selected_option_cleaned'] = llm_all['selected_option'].str.replace('*', '', regex=False).str.strip()
llm_all['llm_behavior'] = llm_all['selected_option_cleaned'].apply(normalize_behavior)

behavior_map = {
    'Assertive Merge': 1,
    'Cooperative Merge (act first)': 2,
    'Cooperative Merge (observe first)': 3,
    'Cautious Merge': 3
}
llm_all['llm_behavior_code'] = llm_all['llm_behavior'].map(behavior_map)

# Merge the final result
final_data = df_encoded.merge(
    llm_all[['id', 'llm_behavior_code',
             'is_gpt', 'is_deepseek', 'is_llama',
             'is_english', 'is_chinese', 'is_french']],
    on='id', how='left'
)

# Save the result
output_path = ROOT / "data/3_encoded_datasets/S1_merging_into_freeway_encoded.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
final_data.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"Encoding finished, saved to {output_path}")
