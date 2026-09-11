"""Build the estimation dataset of the 'yielding to a merging vehicle' scenario.
The 1,500 cases are split into nine subsets (3 LLMs x 3 prompt languages, ~167 cases each); the contextual factors
of every case are encoded (numeric levels, ordinal occupant urgency, one-hot categorical attributes) and merged with the
ordered behaviour code of the option selected by the assigned model (1 = aggressive, 2 = neutral, 3 = conservative)
plus model / language indicators. Output: data/3_encoded_datasets/S2_yielding_to_merging_vehicle_encoded.csv."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # replication-package root

import pandas as pd
import numpy as np

# 1. Read the English scenario configuration (1,500 cases)
df = pd.read_csv(ROOT / "data/1_scenario_configurations/S2_yielding_to_merging_vehicle/sample_1500_en.csv")

# 2. Read the LLM responses (9 model x language cells, ~167 cases each = 1,500 cases) and tag language and model
gpt_en = pd.read_csv(ROOT / "data/2_llm_responses/S2_yielding_to_merging_vehicle/gpt4o_en.csv", encoding='utf-8-sig').iloc[:167]
gpt_cn = pd.read_csv(ROOT / "data/2_llm_responses/S2_yielding_to_merging_vehicle/gpt4o_cn.csv", encoding='utf-8-sig').iloc[167:334]
gpt_fr = pd.read_csv(ROOT / "data/2_llm_responses/S2_yielding_to_merging_vehicle/gpt4o_fr.csv", encoding='utf-8-sig').iloc[334:500]

dp_en = pd.read_csv(ROOT / "data/2_llm_responses/S2_yielding_to_merging_vehicle/deepseek_v3_en.csv", encoding='utf-8-sig').iloc[500:667]
dp_cn = pd.read_csv(ROOT / "data/2_llm_responses/S2_yielding_to_merging_vehicle/deepseek_v3_cn.csv", encoding='utf-8-sig').iloc[667:834]
dp_fr = pd.read_csv(ROOT / "data/2_llm_responses/S2_yielding_to_merging_vehicle/deepseek_v3_fr.csv", encoding='utf-8-sig').iloc[834:1000]

lla_en = pd.read_csv(ROOT / "data/2_llm_responses/S2_yielding_to_merging_vehicle/llama31_405b_en.csv", encoding='utf-8-sig').iloc[1000:1167]
lla_cn = pd.read_csv(ROOT / "data/2_llm_responses/S2_yielding_to_merging_vehicle/llama31_405b_cn.csv", encoding='utf-8-sig').iloc[1167:1334]
lla_fr = pd.read_csv(ROOT / "data/2_llm_responses/S2_yielding_to_merging_vehicle/llama31_405b_fr.csv", encoding='utf-8-sig').iloc[1334:1500]

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

llm_df = pd.concat([gpt_en, gpt_cn, gpt_fr, dp_en, dp_cn, dp_fr, lla_en, lla_cn, lla_fr], ignore_index=True)
llm_df['id'] = llm_df['id'].astype(int)
df['id'] = df['id'].astype(int)

# ========================
# Scenario configuration encoding
# ========================
df['mainstream_front_space'] = df['mainstream_front_space'].replace('free', '300m')

def strip_unit(series, unit):
    return series.str.replace(unit, '', regex=False).astype(int)

df['merging_vehicle_speed'] = strip_unit(df['merging_vehicle_speed'], 'km/h')
df['merging_gap'] = strip_unit(df['merging_gap'], 'm')
df['mainstream_front_space'] = strip_unit(df['mainstream_front_space'], 'm')
df['ego_distance'] = strip_unit(df['ego_distance'], 'm')
df['ego_velocity'] = strip_unit(df['ego_velocity'], 'km/h')

df['occupant_requirement'] = df['occupant_requirement'].map({
    'none': 0,
    'hurry (commute or flight)': 1,
    'urgent (to hospital)': 2
})

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
onehot_cols = ['weather', 'time', 'road_condition', 'ego_state',
               'merging_vehicle_type', 'ego_role', 'following_vehicle']
df_onehot = pd.get_dummies(df, columns=onehot_cols)

# ========================
# LLM behaviour encoding
# ========================
llm_df['selected_option_cleaned'] = llm_df['selected_option'].astype(str).str.replace('*', '', regex=False).str.strip()

def normalize_behavior(text):
    text = str(text).lower()
    
    if "强势" in text or "assertive" in text:
        return "Assertive Interaction"
    elif ("先观察" in text or "observe first" in text or
          "agir ensuite" in text):
        return "Cooperative Interaction (Observe first)"
    elif ("先行动" in text or "act first" in text or
          "observer ensuite" in text):
        return "Cooperative Interaction (Act first)"
    elif "谨慎" in text or "cautious" in text or "prudente" in text:
        return "Cautious Interaction"
    else:
        return "Unknown"

llm_df['llm_behavior'] = llm_df['selected_option_cleaned'].apply(normalize_behavior)
llm_df['llm_behavior_code'] = llm_df['llm_behavior'].map({
    'Assertive Interaction': 1,
    'Cooperative Interaction (Act first)': 2,
    'Cooperative Interaction (Observe first)': 3,
    'Cautious Interaction': 3
})

# ===== 6. Arrival-time difference code (delta_t_diff_code) =====
def kmh_to_mps(kmh): return kmh * 1000 / 3600

ego_v = kmh_to_mps(df['ego_velocity'])
merge_v = kmh_to_mps(df['merging_vehicle_speed'])

ego_dt = np.where(ego_v == 0, 999, df['ego_distance'] / ego_v)
merge_dt = np.where(merge_v == 0, 999, df['merging_gap'] / merge_v)

df_onehot['delta_t_diff'] = (merge_dt - ego_dt).round().astype(int)

def encode_t(diff):
    if diff <= -2: return 1
    elif -2 < diff <= 2: return 2
    else: return 3

df_onehot['delta_t_diff_code'] = df_onehot['delta_t_diff'].apply(encode_t)
df_onehot = df_onehot.drop(columns=['delta_t_diff'])

llm_behavior = llm_df[['id', 'llm_behavior_code', 'is_gpt', 'is_deepseek', 'is_llama', 'is_english', 'is_chinese', 'is_french']]
df_final = df_onehot.merge(llm_behavior, on='id', how='inner')

# Save the final result
output_path = ROOT / "data/3_encoded_datasets/S2_yielding_to_merging_vehicle_encoded.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df_final.to_csv(output_path, index=False)
