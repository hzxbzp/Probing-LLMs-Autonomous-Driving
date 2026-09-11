"""Radar chart of the 'handling pedestrian crossing' scenario (Fig. 7b of the paper): relative thematic emphasis of the
English justifications of the three models (keyword matching, counts column-normalised to 100% per model).
Output: results/S4_handling_pedestrian_crossing_radar.png"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # replication-package root
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re, os

BASE = str(ROOT)
OUT  = str(RESULTS)

MODEL_CSV = {
    "GPT-4o":   f"{BASE}/data/2_llm_responses/S4_handling_pedestrian_crossing/gpt4o_en.csv",
    "DeepSeek": f"{BASE}/data/2_llm_responses/S4_handling_pedestrian_crossing/deepseek_v3_en.csv",
    "Llama":    f"{BASE}/data/2_llm_responses/S4_handling_pedestrian_crossing/llama31_405b_en.csv",
}

MODEL_COLORS = {
    "GPT-4o":   "#e74c3c",
    "DeepSeek": "#27ae60",
    "Llama":    "#2980b9",
}
MODEL_MARKERS = {"GPT-4o": "o", "DeepSeek": "s", "Llama": "D"}

THEME_KEYWORDS = {
    "Environmental\nconditions": [
        "rainy", "wet", "slippery", "dry", "sunny", "night", "day", "visibility", "bad weather",
        "road surface", "weather", "dark", "poor condition", "foggy", "lighting", "time",
        "wet road", "rainy weather", "road condition", "nighttime conditions",
        "nighttime visibility", "weather wet", "night conditions", "rainy conditions",
        "wet conditions", "weather nighttime", "rainy nighttime", "conditions wet",
        "intent rainy", "conditions slowing", "weather presence", "conditions nighttime",
        "rainy night", "conditions reduced",
    ],
    "Surrounding\ntraffic": [
        "gap", "space", "distance", "merging", "bicycle", "cyclist", "pedestrian", "truck",
        "emergency vehicle", "main lane", "vehicle ahead", "vehicle behind", "crosswalk",
        "intersection", "speed", "right of way", "road users", "speed km", "high speed",
        "accident", "lane traffic", "following", "traffic rules", "traffic flow",
        "child", "children", "vulnerable road", "child extra", "lane", "walking",
        "pedestrian decides", "pedestiran behavior", "elderly status", "police", "roadside",
        "cyclist position", "slippery conditions", "ample space", "condition",
        "child cyclist", "wide clearance", "child s", "stopping traffic", "police officer",
        "pedestrian distracted", "elderly pedestrian", "cross", "wide margin",
        "child prioritizing", "night combined", "road user", "close proximity",
        "oncoming traffic", "destination", "behavior", "react", "emergency role",
        "status pregnant", "monitoring", "crosswalk reducing", "presence", "bicycle rider",
        "children riding", "bicycle extra", "situation transporting", "light signal",
        "required distance", "pedestrian behavior", "vulnerable status", "action",
        "distraction watching", "cyclist especially", "clear road", "cyclist space",
        "cyclist efficiently", "distance balances", "potential intent", "orientation",
        "cross balancing", "crossing occurs", "pedestrian begins", "pedestrian intent",
        "pedestrain unclear", "conditions reducing", "pedestrian unclear",
        "pedestrian pregnant", "phone reducing", "pedestrian action", "pedestrian position",
        "pregnant state", "pedestrian proximity", "pedestrian elderly", "near",
        "behavior provides", "speed allows", "position near", "pedestrian role",
        "behavior closely", "child near", "children near", "distraction", "phone slowing",
        "unclear intention", "pregnant status", "distracted", "phone", "intent pregnant",
        "intent elderly", "crosswalk slowing",
    ],
    "Ego vehicle\nstate": [
        "ego", "my vehicle", "speed", "position", "distance to ramp", "stopped", "brake",
        "signal", "act first", "slow down", "accelerate", "wait", "stop", "go", "prepare",
        "adjust", "appraoch ensures", "reach", "stopping distance", "current speed",
        "current state", "pedestrian potential", "cross unexpectedly",
    ],
    "Occupant\nurgency": [
        "urgent", "urgency", "hurry", "need to be fast", "do not delay", "time-sensitive",
        "balances urgency", "rushing", "medical", "high priority", "quick", "occupant",
        "passenger", "delivery", "deadline", "hospital", "passenger requirement",
        "passenger urgency", "require", "passenger urgent", "urgent requirement",
        "urgency requirement", "urgent journey", "hospital destination", "urgent hospital",
        "urgent passenger", "requirement", "occupants", "hospital reducing",
        "requirement reducing", "hospital completely",
    ],
    "Safety": [
        "safety", "safe", "risk", "danger", "hazard", "collision", "avoid", "prevent",
        "secure", "safely completed", "protect", "caution", "alert", "careful",
        "avoid unnecessary", "merge safely", "balances safety", "unsafe maneuvers",
        "emergency", "reduce speed", "timely stop", "continuously monitoring", "safe gap",
        "approach avoids", "safety remains", "risky", "brake suddenly", "safely",
        "conditions safety", "unnecessary risk", "ensuring safety", "priority especially",
        "priority", "later approach", "cooperative merge", "observe", "clear yielding",
        "merge proactively", "proactive approach", "waiting", "yielding action", "observing",
        "intent early", "avoids forcing", "stopping distance", "signaling intent",
        "act later", "safer", "cautious approach", "appropriate", "causing abrupt",
        "prioritize safety", "balances assertiveness", "suitable", "cautious",
        "brake abruptly", "sufficient time", "potential risk", "lane safety",
        "prioritize caution", "safety takes", "slow speed", "crucial observing",
        "abrupt maneuvers", "unsafe action", "dangerous", "confirmed", "avoid forcing",
        "avoid abrupt", "paramount observing", "essential observing", "prioritizing safety",
        "yield ensuring", "safest", "slower acceleration", "lower speed",
        "confirmed ensuring", "signals intent", "emergency situation", "minimizing risk",
        "approriate action", "potential skidding", "reduced visibility",
        "critical observing", "yield", "minimizes risk", "clear communication",
        "especially considering", "sufficient distance", "reduced traction",
        "cautious observation", "distracted behavior", "cautious interaction",
        "hospital safety", "potential risks", "safe interaction", "potential collision",
        "s safety", "endangered", "vulnerable statues", "balances caution",
        "balancing safety", "reducing speed", "potential unpredictability", "cautious pass",
        "endangering", "cyclist safety", "extra caution", "accommodating",
        "reaction time", "cautious response", "sufficient reaction", "better reaction",
        "monitoring allows", "safety honking", "takes precedence", "precedence", "slowing",
        "follow patiently", "slow", "moderate speed", "safe distance", "minimize risk",
        "following emergency", "safe overtaking", "hurry safety", "legal clearance",
        "minimum legal", "pass safely", "legally required", "safety especially", "complies",
        "safety ensuring", "caution slowing", "cautious monitoring", "react appropriately",
        "unclear intent", "cautious driving", "safer response", "safe stop",
        "speed slowing", "safe reaction", "pedestrian safety", "completely stopping",
        "exercise caution", "suitable approach", "timely reaction",
        "yielding preemptively", "caution reducing",
    ],
    "Mobility": [
        "merge efficiently", "km h", "delay", "approach minimizes", "efficiency",
        "unnecessary delay", "efficiently", "merging proactively", "avoid unnecessary",
        "unnecessary stop", "prudent approach", "prudent choice", "observe later",
        "proactively", "act", "assertive", "forcing", "disrupt", "force", "hiher speed",
        "assertive approach", "smooth", "timely", "immediately", "minimizing", "abrupt",
        "maintain progress", "slightly increasing", "proceeding minimizing", "proceeding",
        "proceeding reducing", "ensuring smooth", "assertive interaction",
        "proactively signaling", "timely progress", "timely response", "continue",
        "minimizing unnecessary", "continue driving", "smooth traffic", "increasing speed",
        "interfere", "speed prioritizing", "progress", "efficient way", "timely passage",
        "efficient", "quick response", "urgency avoiding", "unnecessarily delaying",
        "prudent", "maintain", "proceed", "timely arrival", "prioritize", "maintain speed",
        "making progress", "efficient interaction", "passage", "prioritize efficiency",
        "proceed efficiently", "pass promptly", "overtaking", "pass", "overtake", "hurry",
        "efficient approach", "avoiding unnecessary", "unnecessarily stopping",
        "delays unless", "pass using", "assertive pass", "undue delay", "horn",
        "balances efficiency", "passing", "adhering", "minimum legally",
    ],
}

THEMES = list(THEME_KEYWORDS.keys())


def load_model(name, path):
    df = pd.read_csv(path)
    df["selected_option"] = (
        df["selected_option"]
        .str.replace(r"\*\*", "", regex=True)
        .str.strip()
    )
    merge_map = {
        "Yield Preemptively": "Slow Down and Monitor",
        "Maintain Speed with Alertness": "Slow Down and Monitor",
    }
    df["selected_option"] = df["selected_option"].replace(merge_map)
    df["model"] = name
    return df


def theme_col_normalized(df):
    """
    Replicate generate_all_radar.py / correlation_theme_option2.py pipeline:
    count keyword hits per theme -> column-normalize so themes sum to 100%.
    """
    counts = {}
    for theme, keywords in THEME_KEYWORDS.items():
        pattern = "|".join([r"\b" + re.escape(kw) + r"\b" for kw in keywords])
        counts[theme] = df["justification"].str.contains(pattern, regex=True, na=False).sum()
    total = sum(counts.values())
    if total == 0:
        return {t: 0.0 for t in counts}
    return {t: c / total * 100 for t, c in counts.items()}


def plot_radar(model_rates):
    N = len(THEMES)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    theta_off = np.pi / 2 + np.pi / N
    ax.set_theta_offset(theta_off)
    ax.set_theta_direction(-1)

    # ---- grid setup (adaptive to data range) ----
    max_val = max(max(r.values()) for r in model_rates.values())
    r_max = max(max_val * 1.15, 25)
    ax.set_ylim(0, r_max)

    ring_theta = np.linspace(0, 2 * np.pi, 300)
    step = 5 if r_max <= 35 else 10
    grid_vals = list(np.arange(step, r_max, step))

    # Solid concentric circles (reference style)
    for rv in grid_vals:
        ax.plot(ring_theta, [rv] * len(ring_theta),
                color='#888', linewidth=0.8, alpha=0.4, zorder=1)
    # Outer circle (thicker)
    ax.plot(ring_theta, [r_max] * len(ring_theta),
            color='#555', linewidth=1.5, zorder=1)

    # Dotted spoke lines (reference style)
    for a in angles[:-1]:
        ax.plot([a, a], [0, r_max], color='#888',
                linewidth=0.8, linestyle=':', alpha=0.7, zorder=1)

    # Hide all default grids
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.spines['polar'].set_visible(False)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.set_yticks(grid_vals)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.tick_params(axis='y', labelsize=12, labelcolor='#444')
    ax.set_rlabel_position(15)

    # ---- data lines ----
    for model, rates in model_rates.items():
        vals = [rates[t] for t in THEMES] + [rates[THEMES[0]]]
        color = MODEL_COLORS[model]
        marker = MODEL_MARKERS[model]
        ax.plot(angles, vals, marker + "-", linewidth=2.2, label=model,
                color=color, markersize=6, zorder=5)
        ax.fill(angles, vals, alpha=0.10, color=color)

    def _mono_center_stack(theme_key):
        """Pad lines to equal character width so columns center (DejaVu Sans Mono)."""
        lines = [s.strip() for s in theme_key.split("\n") if s.strip()]
        if not lines:
            return ""
        w = max(len(s) for s in lines)
        return "\n".join(s.center(w) for s in lines)

    # ---- Labels: monospace center-stack + tangent rotation (perpendicular to radial) ----
    label_r0 = r_max + 1.2

    for i, theme in enumerate(THEMES):
        angle_rad = angles[i]
        disp_deg = (np.degrees(theta_off - angle_rad)) % 360
        rot = disp_deg - 90
        if 90 < rot < 270:
            rot += 180

        ax.text(
            angle_rad,
            label_r0,
            _mono_center_stack(theme),
            ha="center",
            va="center",
            fontsize=14,
            color="black",
            fontfamily="DejaVu Sans Mono",
            fontweight="bold",
            rotation=rot,
            rotation_mode="anchor",
            linespacing=1.05,
        )

    # ---- legend: lower-left inside axes (corner wedge, away from filled polygon) ----
    leg = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.03, 0.12),
        bbox_transform=ax.transAxes,
        ncol=1,
        fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=False,
        facecolor="white",
        edgecolor="#bbb",
        framealpha=0.92,
        title="Model",
        title_fontsize=12,
    )
    leg.set_zorder(10)

    fig.subplots_adjust(left=0.08, right=0.92, top=0.94, bottom=0.08)

    out = f"{OUT}/S4_handling_pedestrian_crossing_radar.png"
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    model_rates = {}
    for name, path in MODEL_CSV.items():
        df = load_model(name, path)
        print(f"{name}: {len(df)} rows, options: {df['selected_option'].value_counts().to_dict()}")
        model_rates[name] = theme_col_normalized(df)
        for t, v in model_rates[name].items():
            print(f"  {t:30s} {v:6.1f}%")

    plot_radar(model_rates)
    print("Done.")
