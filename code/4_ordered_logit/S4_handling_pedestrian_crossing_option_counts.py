"""Option frequencies of the 'handling pedestrian crossing' scenario (Section 3.5 of the paper).

No ordered logit model is estimated for this scenario because the neutral option ('Slow Down and Monitor')
is selected almost exclusively. This script tabulates the selected options per model and prompt language from
the raw response files and from the encoded nine-subset dataset.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # replication-package root
RESP = ROOT / "data/2_llm_responses/S4_handling_pedestrian_crossing"


def normalise(text):
    text = str(text).lower().replace("*", "")
    if "alertness" in text or "警觉" in text or "maintenir" in text:
        return "Maintain Speed with Alertness (aggressive)"
    if "monitor" in text or "观察" in text or "observer" in text:
        return "Slow Down and Monitor (neutral)"
    if "yield" in text or "让行" in text or "proactive" in text:
        return "Yield Preemptively (conservative)"
    return "unparsed"


rows = []
for model in ("gpt4o", "deepseek_v3", "llama31_405b"):
    for lang in ("en", "cn", "fr"):
        df = pd.read_csv(RESP / f"{model}_{lang}.csv", encoding="utf-8-sig")
        counts = df["selected_option"].map(normalise).value_counts()
        for option, n in counts.items():
            rows.append({"model": model, "language": lang, "option": option, "n": n})
table = pd.DataFrame(rows).pivot_table(index=["model", "language"], columns="option", values="n", fill_value=0)
print("Selected options per model and prompt language (1,500 cases each):")
print(table.to_string())

enc = pd.read_csv(ROOT / "data/3_encoded_datasets/S4_handling_pedestrian_crossing_encoded.csv")
print("\nBehaviour codes in the nine-subset estimation dataset (1 = aggressive, 2 = neutral, 3 = conservative):")
for flag in ("is_gpt", "is_deepseek", "is_llama"):
    print(flag, enc.loc[enc[flag] == True, "llm_behavior_code"].value_counts().sort_index().to_dict())
out = ROOT / "results" / "S4_handling_pedestrian_crossing_option_counts.csv"
out.parent.mkdir(exist_ok=True)
table.to_csv(out)
print(f"\nSaved {out}")
