"""Build the estimation dataset of the 'following a cyclist' scenario.
The 1,500 cases are split into nine subsets (3 LLMs x 3 prompt languages, ~167 cases each); the contextual factors
of every case are encoded (numeric levels, ordinal occupant urgency, one-hot categorical attributes) and merged with the
ordered behaviour code of the option selected by the assigned model (1 = aggressive, 2 = neutral, 3 = conservative)
plus model / language indicators. Output: data/3_encoded_datasets/S3_following_a_cyclist_encoded.csv."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # replication-package root

import pandas as pd

# 1. Read the English scenario configuration (1,500 cases)
df = pd.read_csv(ROOT / "data/1_scenario_configurations/S3_following_a_cyclist/sample_1500_en.csv")

# 2. Read the LLM responses (9 model x language cells, ~167 cases each = 1,500 cases) and tag language and model
gpt_en = pd.read_csv(ROOT / "data/2_llm_responses/S3_following_a_cyclist/gpt4o_en.csv", encoding='utf-8-sig').iloc[:167]
gpt_cn = pd.read_csv(ROOT / "data/2_llm_responses/S3_following_a_cyclist/gpt4o_cn.csv", encoding='utf-8-sig').iloc[167:334]
gpt_fr = pd.read_csv(ROOT / "data/2_llm_responses/S3_following_a_cyclist/gpt4o_fr.csv", encoding='utf-8-sig').iloc[334:500]

dp_en = pd.read_csv(ROOT / "data/2_llm_responses/S3_following_a_cyclist/deepseek_v3_en.csv", encoding='utf-8-sig').iloc[500:667]
dp_cn = pd.read_csv(ROOT / "data/2_llm_responses/S3_following_a_cyclist/deepseek_v3_cn.csv", encoding='utf-8-sig').iloc[667:834]
dp_fr = pd.read_csv(ROOT / "data/2_llm_responses/S3_following_a_cyclist/deepseek_v3_fr.csv", encoding='utf-8-sig').iloc[834:1000]

lla_en = pd.read_csv(ROOT / "data/2_llm_responses/S3_following_a_cyclist/llama31_405b_en.csv", encoding='utf-8-sig').iloc[1000:1167]
lla_cn = pd.read_csv(ROOT / "data/2_llm_responses/S3_following_a_cyclist/llama31_405b_cn.csv", encoding='utf-8-sig').iloc[1167:1334]
lla_fr = pd.read_csv(ROOT / "data/2_llm_responses/S3_following_a_cyclist/llama31_405b_fr.csv", encoding='utf-8-sig').iloc[1334:1500]

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

# Strip units and convert to integers
def strip_unit(series, unit):
    return series.str.replace(unit, '', regex=False).astype(int)

df['bicycle_front_space'] = df['bicycle_front_space'].replace('free', '300m')
df['dist_bicycle'] = df['dist_bicycle'].replace('free', '300m')

df['bicycle_front_space'] = strip_unit(df['bicycle_front_space'], 'm')
df['dist_bicycle'] = strip_unit(df['dist_bicycle'], 'm')
df['ego_vehicle_speed'] = strip_unit(df['ego_vehicle_speed'], 'km/h')

df['occupant_requirement'] = df['occupant_requirement'].map({
    'none': 0,
    'hurry (commute or flight)': 1,
    'urgent (to hospital)': 2,
})

df['following_vehicle'] = df['following_vehicle'].fillna('none')

# Clean the column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# One-hot encoding
categorical_cols = [
    'weather', 'time', 'road_condition',
    'bicycle_position', 'bicycle_rider',
    'ego_role', 'following_vehicle'
]
df_onehot = pd.get_dummies(df, columns=categorical_cols)

# ========== 5. Encode the selected option ==========
llm_df['selected_option_cleaned'] = llm_df['selected_option'].astype(str).str.replace('*', '', regex=False).str.strip()

def normalize_behavior(text):
    text = text.lower()
    if "honk" in text or "鸣笛" in text or "signaler" in text:
        return "Honk or Signal and Pass"
    elif "assertive" in text or "果断" in text or "ferme" in text:
        return "Assertive Pass with Minimum Distance"
    elif "cautious" in text or "谨慎" in text or "prudent" in text:
        return "Cautious Pass with Wide Margin"
    elif "follow" in text or "跟随" in text or "patiemment" in text:
        return "Follow Patiently"
    else:
        return "Unknown"

llm_df['llm_behavior'] = llm_df['selected_option_cleaned'].apply(normalize_behavior)
llm_df['llm_behavior_code'] = llm_df['llm_behavior'].map({
    'Honk or Signal and Pass': 1,
    'Assertive Pass with Minimum Distance': 2,
    'Cautious Pass with Wide Margin': 3,
    'Follow Patiently': 3
})

# ========== 6. Merge into the final table ==========
llm_behavior = llm_df[['id', 'llm_behavior_code', 'is_gpt', 'is_deepseek', 'is_llama', 'is_english', 'is_chinese', 'is_french']]
df_final = df_onehot.merge(llm_behavior, on='id', how='inner')


# Save the result
output_path = ROOT / "data/3_encoded_datasets/S3_following_a_cyclist_encoded.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df_final.to_csv(output_path, index=False)
