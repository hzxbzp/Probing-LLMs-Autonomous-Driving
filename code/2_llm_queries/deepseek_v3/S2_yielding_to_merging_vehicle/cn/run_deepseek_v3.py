"""Query DeepSeek-V3 sequentially for the 1,500 cases of the 'yielding to a merging vehicle' scenario (Chinese prompts)
through its OpenAI-compatible chat-completions endpoint (api.deepseek.com, model deepseek-chat).
Requires DEEPSEEK_KEY in config.yaml. Output: data/2_llm_responses/<scenario>/deepseek_v3_cn.csv."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]  # replication-package root

from tqdm import tqdm
import json
import time
import csv
import pandas as pd
from openai import OpenAI
import yaml
from prompt import SYSTEM_PROMPT, OPTIONS

# set OpenAI API key and model
OPENAI_CONFIG = yaml.load(open(ROOT / "config.yaml"), Loader=yaml.FullLoader)
api_key = OPENAI_CONFIG['DEEPSEEK_KEY']
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com",
)

# Read the input scenario CSV
df = pd.read_csv(ROOT / "data/1_scenario_configurations/S2_yielding_to_merging_vehicle/sample_1500_cn.csv")

# Initialise the output CSV
output_path = ROOT / "data/2_llm_responses/S2_yielding_to_merging_vehicle/deepseek_v3_cn.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, mode="w", newline='', encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "selected_option", "justification"])

    # Everything below stays inside the 'with' block so that every row is written to the file
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        user_prompt = f"""感知信息：
- 天气：{row['天气']}
- 时间：{row['时间']}
- 路况：{row['路况']}
- 主车道前方空隙：{row['主车道前方空隙']}
- 并车道车辆速度：{row['并车车速']}
- 并车车辆与匝道距离：{row['并车与本车距']}
- 并车车辆类型：{row['并车车辆类型']}

本车状态：
- 本车角色：{row['本车角色']}
- 当前状态：{row['本车状态']}
- 距主车道距离：{row['本车距合流点距离']}
- 本车速度：{row['本车速度']}
- 后车类型：{row['后车类型']}

乘客信息：
- 乘客需求：{row['乘客需求']}

驾驶行为选项：
{OPTIONS}""".strip()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # print(f"Processing id: {row['id']}")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=1.0,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            # print(f"Model response (id={row['id']}): {content}")  # DEBUG
            parsed = json.loads(content)
            selected_option = parsed.get("选择的选项", "")
            justification = parsed.get("选择理由", "")

        except Exception as e:
            print(f"Error (id={row['id']}): {e}")
            selected_option = "ERROR"
            justification = str(e)

        writer.writerow([row["id"], selected_option, justification])
        time.sleep(0.5)
        

print("All results saved to:", output_path)
