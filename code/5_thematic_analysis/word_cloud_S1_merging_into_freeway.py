"""Theme-coloured word cloud of the English justifications in the 'merging into freeway' scenario
(Fig. 3 of the paper, one panel per model).
Usage: python word_cloud_<scenario>.py [gpt4o|deepseek|llama]   Output: results/<scenario>_word_cloud_<model>.png"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]  # replication-package root
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import matplotlib.patches as mpatches


# ===== 1. Theme keywords and colours =====
theme_keywords = {
    "Environmental conditions": [
        "rainy", "wet", "slippery", "dry", "sunny", "night", "day", "visibility", "bad weather",
        "road surface", "weather", "dark", "poor condition", "foggy", "lighting","time", "wet road", "rainy weather", "road condition","nighttime conditions","nighttime visibility","weather wet","night conditions"
    ],
    "Surrounding traffic": [
        "gap", "space", "distance", "merging", "bicycle", "cyclist", "pedestrian", "truck", "emergency vehicle",
        "main lane", "vehicle ahead", "vehicle behind", "crosswalk", "intersection", "speed", "yield", "right of way","road users", "speed km", "high speed","accident","lane traffic","following","traffic rules","traffic flow","35km h","30km h","60km h","100m", "ramp","lane vehicles'", "main","50km h","100km h","current conditions","surrounding","way","approaching","80km h","role","speed difference","free","dry road","50m","higher speed","situation","large gap","sufficiently large","relatively slow",
        "relatively low","clear","truck travelling"
    ],
    "Ego vehicle state": [
        "ego", "my vehicle", "speed", "position", "distance to ramp", "stopped", "brake", "signal", 
        "act first", "slow down", "accelerate", "wait", "stop", "go", "prepare", "adjust", "appraoch ensures","reach"," stopping distance","current speed"
    ],
    "Occupant urgency": [
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
    ],
    # "cooperation": [
    #     "signaling intent", "merging proactively","later approach","cooperative merge","observe", "clear yielding","merge proactively","proactive approach",
    #     "cooperative","waiting","yielding action","observe later","proactively","act","assertive","intent early","avoids forcing","forcing","disrupt","observing"
    # ],
    # "traffic rules": [
    #     "rule", "regulation", "law", "policy", "guideline", "protocol", "standard", "requirement",
    #     "compliance", "obligation", "mandate", "directive", "procedure","traffic rules","traffic law","traffic regulation","role","priority especially",
    #     "priority"
    # ]
}

# Stop words
custom_stopwords = {
    "vehicle", "vehicles", "given", "option", "this", "that", "it", "to", "and", "for", "in", "on", "with", "at",
    "is", "are", "was", "were", "be", "been", "being", "will", "can", "should", "must", "may", "might","do", "does", "did", "has", "have", "had",
    "if", "but", "so", "or", "not", "which", "ensure","abort","necessary require","maneuver","need","necessary","prompt","create", "maintaining", "additionally","allowing",
    "approach ensrues","ensures","merge","merge act","allow","balance","available"

}

# Theme colours
theme_colors = {
    "Environmental conditions": "#ff5877",  # deep rose
    "Surrounding traffic": "#377ec8",           # deep lake blue
    "Ego vehicle state": "#2e8b74",   # deep mint green
    "Occupant urgency": "#d2691e",      # smoky orange-brown
    "Safety": "#7aa24a",                    # deep grass green
    "Mobility": "#7949f2",                # deep purple
}


# ===== 2. Word -> colour lookup =====
word_theme_map = {}
for theme, words in theme_keywords.items():
    color = theme_colors[theme]
    for word in words:
        word_theme_map[word.lower()] = color

# ===== 3. Colour function =====
def theme_color_func(word, **kwargs):
    word_lc = word.lower()
    return word_theme_map.get(word_lc, "gray")  # theme words are coloured, all other words are grey

# ===== 4. Load the justification texts =====
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt4o"   # gpt4o | deepseek | llama
INPUT_FILES = {
    "gpt4o":    ROOT / "data/2_llm_responses/S1_merging_into_freeway/gpt4o_en.csv",
    "deepseek": ROOT / "data/2_llm_responses/S1_merging_into_freeway/deepseek_v3_en.csv",
    "llama":    ROOT / "data/2_llm_responses/S1_merging_into_freeway/llama31_405b_en.csv",
}
df = pd.read_csv(INPUT_FILES[MODEL])
text = " ".join(df["justification"].dropna().astype(str)).lower()


final_stopwords = ENGLISH_STOP_WORDS.union(custom_stopwords)

# ===== 5. Generate the word cloud =====
font_path = None  # None = default WordCloud font; the manuscript figures used Microsoft YaHei (C:/Windows/Fonts/msyh.ttc)

wordcloud = WordCloud(
    font_path=font_path,
    width=1200,
    height=600,
    background_color="white",
    stopwords=final_stopwords,
    random_state=42,
    max_words=100
).generate(text)

# ===== 6. Draw the word cloud with theme colours =====
# plt.figure(figsize=(14, 7))
# plt.imshow(wordcloud.recolor(color_func=theme_color_func), interpolation="bilinear")
# plt.axis("off")
# plt.title("Justification Word Cloud Colored by Theme", fontsize=16)
# plt.show()

plt.figure(figsize=(16, 8))  # larger figure to leave room for the legend
plt.imshow(wordcloud.recolor(color_func=theme_color_func), interpolation="bilinear")
plt.axis("off")
# plt.title("Justification Word Cloud Colored by Theme", fontsize=18)

# 2. Legend entries
legend_patches = []
for theme, color in theme_colors.items():
    patch = mpatches.Patch(color=color, label=theme)
    legend_patches.append(patch)

# 3. Add the legend
legend_patches.append(mpatches.Patch(color="gray", label="Other"))

# 3. Add the legend
plt.legend(
    handles=legend_patches,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),  # position: below the image
    ncol=len(legend_patches),     # legend entries in one row
    fontsize=14,                  # legend font size
    title="Themes",
    title_fontsize=15,
    frameon=False                 # optional: no frame
)

# 4. Save / show the figure
plt.tight_layout()
plt.savefig(RESULTS / f"S1_merging_into_freeway_word_cloud_{MODEL}.png", dpi=300, bbox_inches="tight")
plt.show()
