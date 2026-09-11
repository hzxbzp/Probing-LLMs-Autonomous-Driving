"""Build the 1,500 GPT-4o requests of the 'yielding to a merging vehicle' scenario (Chinese prompts) and submit them
to the OpenAI Batch API. Requires OPENAI_KEY in config.yaml (copy config.example.yaml).
Usage: python submit_batch.py [--dry-run]   (--dry-run only writes the JSONL request file)
The request file is written to data/2_llm_responses/<scenario>/gpt4o_requests_<lang>.jsonl."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]  # replication-package root

import os
import json
import time
import csv
import pandas as pd
from openai import OpenAI
import yaml
from prompt import SYSTEM_PROMPT, OPTIONS

# set OpenAI API key and model
OPENAI_CONFIG = yaml.load(open(ROOT / "config.yaml"), Loader=yaml.FullLoader)
if OPENAI_CONFIG['OPENAI_API_TYPE'] == 'openai':
    api_key = OPENAI_CONFIG['OPENAI_KEY']
    client = OpenAI(api_key = api_key)

df = pd.read_csv(ROOT / "data/1_scenario_configurations/S2_yielding_to_merging_vehicle/sample_1500_cn.csv")

TEMPERATURE = 1.0  # sampling temperature

tasks = []

for index, row in df.iterrows():
    # Dynamically generate user content from row
    user_content = f"""感知信息：
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

    task = {
        "custom_id": str(row["id"]),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-2024-11-20",
            "temperature": TEMPERATURE,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
        }
    }
    tasks.append(task)

# Creating the jsonl file
file_name = ROOT / "data/2_llm_responses/S2_yielding_to_merging_vehicle/gpt4o_requests_cn.jsonl"

file_name.parent.mkdir(parents=True, exist_ok=True)
with open(file_name, 'w') as file:
    for obj in tasks:
        file.write(json.dumps(obj) + '\n')

if "--dry-run" in sys.argv:
    print(f"Dry run: wrote {len(tasks)} requests to {file_name}; nothing was uploaded.")
    sys.exit(0)

batch_file = client.files.create(
    file=open(file_name, "rb"),
    purpose="batch"
)

# Start batch job
batch_job = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

# Print the batch job ID
print(f"Batch job ID: {batch_job.id}")
print(f"Batch job status: {batch_job.status}")

