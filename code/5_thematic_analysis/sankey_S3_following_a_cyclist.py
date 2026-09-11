"""Sankey diagram of the 'following a cyclist' scenario (Fig. 7a of the paper):
keyword groups -> themes -> models -> selected behaviours, built from the English justifications of the three models
(1,500 cases each). Output: results/S3_following_a_cyclist_sankey.png"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # replication-package root
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import re, os
import numpy as np
import pandas as pd

BASE = str(ROOT)
OUT  = str(RESULTS)
os.makedirs(OUT, exist_ok=True)

# ============================================================
# Keyword groups (merged from full correlation_theme_option2.py lists)
# ============================================================
KEYWORD_GROUPS = {
    # --- Environmental conditions ---
    "Rain / Wet / Slippery": {
        "theme": "Environmental\nconditions",
        "patterns": [
            "rainy", "wet", "slippery", "rain", "wet road", "rainy weather",
            "rainy conditions", "wet conditions", "weather wet", "rainy night",
            "slippery conditions",
        ],
    },
    "Night / Visibility": {
        "theme": "Environmental\nconditions",
        "patterns": [
            "night", "nighttime", "dark", "visibility", "foggy", "fog", "lighting",
            "nighttime conditions", "nighttime visibility", "night conditions",
            "conditions nighttime", "poor condition", "reduced visibility",
            "night combined", "weather nighttime",
        ],
    },
    "Road / Weather": {
        "theme": "Environmental\nconditions",
        "patterns": [
            "road surface", "road condition", "weather", "dry", "sunny", "day",
            "bad weather", "weather presence", "time",
        ],
    },
    # --- Surrounding traffic information ---
    "Child / Cyclist / Vulnerable": {
        "theme": "Surrounding\ntraffic",
        "patterns": [
            "child", "children", "cyclist", "bicycle", "bicycle rider",
            "vulnerable road", "child extra", "child cyclist", "children riding",
            "cyclist position", "cyclist especially", "child prioritizing",
            "childe prioritizing", "cyclist space", "cyclist efficiently",
            "child s", "bicycle extra", "road user", "road users", "pedestrian",
        ],
    },
    "Distance / Gap / Space": {
        "theme": "Surrounding\ntraffic",
        "patterns": [
            "gap", "space", "distance", "clearance", "margin", "wide margin",
            "wide clearance", "ample space", "close proximity", "required distance",
            "distance balances", "sufficiently large", "large gap", "sufficient distance",
        ],
    },
    "Speed / Position": {
        "theme": "Surrounding\ntraffic",
        "patterns": [
            "speed", "speed km", "high speed", "speed difference", "higher speed",
            "slower speed", "low speed", "relatively slow", "relatively low",
            "relatively high", "approaching", "situation",
        ],
    },
    "Vehicles / Traffic Rules": {
        "theme": "Surrounding\ntraffic",
        "patterns": [
            "truck", "emergency vehicle", "oncoming traffic", "following",
            "traffic rules", "traffic flow", "traffic norms", "vehicle ahead",
            "vehicle behind", "right of way", "lane", "lane traffic",
            "main lane", "crosswalk", "intersection", "light signal",
        ],
    },
    # --- Ego vehicle state ---
    "Ego Speed / State": {
        "theme": "Ego vehicle\nstate",
        "patterns": [
            "ego", "my vehicle", "speed", "position", "current speed",
            "current state", "distance to ramp",
        ],
    },
    "Brake / Slow / Signal": {
        "theme": "Ego vehicle\nstate",
        "patterns": [
            "brake", "slow down", "stop", "stopped", "accelerate", "signal",
            "act first", "wait", "go", "prepare", "adjust", "stopping distance",
            "decelerat",
        ],
    },
    # --- Occupant urgency ---
    "Urgency / Hospital": {
        "theme": "Occupant\nurgency",
        "patterns": [
            "urgent", "urgency", "hospital", "medical", "time-sensitive",
            "rushing", "hurry", "balances urgency", "passenger urgency",
            "urgent requirement", "urgency requirement", "urgent journey",
            "do not delay", "need to be fast", "high priority", "quick",
        ],
    },
    "Passenger / Occupant": {
        "theme": "Occupant\nurgency",
        "patterns": [
            "passenger", "occupant", "occupants", "delivery", "deadline",
            "passenger requirement", "passenger urgent", "requirement", "require",
        ],
    },
    # --- Safety ---
    "Risk / Collision / Danger": {
        "theme": "Safety",
        "patterns": [
            "risk", "collision", "danger", "hazard", "accident", "risky",
            "unsafe", "dangerous", "potential risk", "potential collision",
            "potential risks", "potential skidding", "endangered", "endangering",
            "unsafe action", "unsafe maneuvers", "emergency situation",
            "potential unpredictability", "injury",
        ],
    },
    "Caution / Careful": {
        "theme": "Safety",
        "patterns": [
            "caution", "cautious", "careful", "extra caution", "cautious approach",
            "cautious pass", "cautious interaction", "cautious observation",
            "prioritize safety", "prioritize caution", "prioritizing safety",
            "balances safety", "safety takes", "safety remains", "safety especially",
            "cyclist safety", "safety ensuring", "safety honking",
        ],
    },
    "Safe / Protect / Prevent": {
        "theme": "Safety",
        "patterns": [
            "safety", "safe", "safely", "protect", "secure", "prevent",
            "avoid", "safe gap", "safe distance", "safe interaction",
            "safe overtaking", "safely completed", "pass safely",
            "ensuring safety", "minimize risk", "minimizing risk", "minimizes risk",
            "sufficient time", "sufficient distance", "reduced traction",
        ],
    },
    "Yield / Observe / Slow": {
        "theme": "Safety",
        "patterns": [
            "yield", "observe", "observing", "yielding", "waiting",
            "slowing", "slow", "slower acceleration", "slow speed",
            "moderate speed", "lower speed", "takes precedence", "precedence",
            "paramount", "essential", "crucial", "accommodating",
            "follow patiently", "legal clearance", "legally required", "complies",
        ],
    },
    # --- Mobility ---
    "Efficiency / Progress": {
        "theme": "Mobility",
        "patterns": [
            "efficiency", "efficiently", "progress", "efficient",
            "making progress", "maintain progress", "timely", "timely progress",
            "timely passage", "timely arrival", "timely response", "smooth",
            "ensuring smooth", "smooth traffic", "efficient way",
            "efficient approach", "efficient interaction",
            "prioritize efficiency", "proceed efficiently", "balances efficiency",
        ],
    },
    "Pass / Overtake": {
        "theme": "Mobility",
        "patterns": [
            "pass ", "passing", "overtake", "overtaking", "pass promptly",
            "assertive pass", "pass using", "horn",
        ],
    },
    "Assertive / Force / Speed": {
        "theme": "Mobility",
        "patterns": [
            "assertive", "assertive approach", "forcing", "force", "disrupt",
            "proactively", "maintain speed", "maintain", "proceed", "continue",
            "increasing speed", "immediately", "delay", "unnecessary delay",
            "undue delay", "hurry",
        ],
    },
}

THEME_ORDER = [
    "Environmental\nconditions", "Surrounding\ntraffic", "Ego vehicle\nstate",
    "Occupant\nurgency", "Safety", "Mobility",
]
MODEL_ORDER = ["GPT-4o", "DeepSeek", "Llama"]
BEHAVIOR_SHORT = {
    "Assertive Pass with Minimum Distance": "Overtake with Signal\n(Aggressive)",
    "Honk or Signal and Pass": "Negotiate the Overtake\n(Neutral)",
    "Cautious Pass with Wide Margin": "Follow without Overtaking\n(Conservative)",
}
BEHAVIOR_ORDER = [
    "Overtake with Signal\n(Aggressive)",
    "Negotiate the Overtake\n(Neutral)",
    "Follow without Overtaking\n(Conservative)",
]

THEME_COLORS = {
    "Environmental\nconditions": "#1abc9c",
    "Surrounding\ntraffic":     "#3498db",
    "Ego vehicle\nstate":        "#9b59b6",
    "Occupant\nurgency":    "#e67e22",
    "Safety":                   "#e74c3c",
    "Mobility":                 "#27ae60",
}
MODEL_COLORS = {"GPT-4o": "#e74c3c", "DeepSeek": "#27ae60", "Llama": "#2980b9"}
BEHAVIOR_COLORS = {
    "Overtake with Signal\n(Aggressive)": "#e74c3c",
    "Negotiate the Overtake\n(Neutral)": "#27ae60",
    "Follow without Overtaking\n(Conservative)": "#2980b9",
}


def text_has_pattern(text, patterns):
    t = str(text).lower()
    for p in patterns:
        pl = p.lower().strip()
        if not pl:
            continue
        if " " in pl:
            if pl in t:
                return True
        else:
            if re.search(re.escape(pl), t):
                return True
    return False


def load_all():
    frames = []
    for model, path in {
        "GPT-4o":   f"{BASE}/data/2_llm_responses/S3_following_a_cyclist/gpt4o_en.csv",
        "DeepSeek": f"{BASE}/data/2_llm_responses/S3_following_a_cyclist/deepseek_v3_en.csv",
        "Llama":    f"{BASE}/data/2_llm_responses/S3_following_a_cyclist/llama31_405b_en.csv",
    }.items():
        df = pd.read_csv(path)
        df["selected_option"] = df["selected_option"].str.replace(r"\*\*", "", regex=True).str.strip()
        df["selected_option"] = df["selected_option"].replace({"Follow Patiently": "Cautious Pass with Wide Margin"})
        df["model"] = model
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def compute_links(df):
    links_kw_th, links_th_mo, links_mo_bh = [], [], []

    for kw_name, info in KEYWORD_GROUPS.items():
        count = df["justification"].apply(lambda x: text_has_pattern(x, info["patterns"])).sum()
        if count > 0:
            links_kw_th.append((kw_name, info["theme"], count))

    for theme in THEME_ORDER:
        kgs = [k for k, v in KEYWORD_GROUPS.items() if v["theme"] == theme]
        pats = []
        for kg in kgs:
            pats.extend(KEYWORD_GROUPS[kg]["patterns"])
        for model in MODEL_ORDER:
            mask = df["model"] == model
            count = df.loc[mask, "justification"].apply(lambda x: text_has_pattern(x, pats)).sum()
            if count > 0:
                links_th_mo.append((theme, model, count))

    for model in MODEL_ORDER:
        for beh_full, beh_short in BEHAVIOR_SHORT.items():
            count = ((df["model"] == model) & (df["selected_option"] == beh_full)).sum()
            if count > 0:
                links_mo_bh.append((model, beh_short, count))

    return links_kw_th, links_th_mo, links_mo_bh


def draw_curved_band(ax, x0, y0_bot, y0_top, x1, y1_bot, y1_top, color, alpha=0.32):
    verts, codes = [], []
    n = 40
    for i in range(n + 1):
        t = i / n
        x = x0 + t * (x1 - x0)
        s = 3 * t**2 - 2 * t**3
        verts.append((x, y0_top + s * (y1_top - y0_top)))
        codes.append(Path.LINETO if i else Path.MOVETO)
    for i in range(n + 1):
        t = i / n
        x = x1 - t * (x1 - x0)
        s = 3 * t**2 - 2 * t**3
        verts.append((x, y1_bot + s * (y0_bot - y1_bot)))
        codes.append(Path.LINETO)
    codes.append(Path.CLOSEPOLY)
    verts.append(verts[0])
    ax.add_patch(mpatches.PathPatch(Path(verts, codes), facecolor=color, edgecolor='none', alpha=alpha))


def plot_sankey(links_kw_th, links_th_mo, links_mo_bh):
    kw_names = list(KEYWORD_GROUPS.keys())
    col_x = [0.08, 0.36, 0.64, 0.92]
    # Wider outer columns so labels fit inside colored bars
    bar_w_by_col = [0.058, 0.038, 0.038, 0.072]

    fig, ax = plt.subplots(figsize=(24, 16))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.axis('off')

    def layout_nodes(names, col_idx, flow_dict):
        x = col_x[col_idx]
        bw = bar_w_by_col[col_idx]
        total = sum(flow_dict.values())
        usable = 0.90
        pad = 0.006
        bar_total = usable - pad * (len(names) - 1)
        pos = {}
        y = 0.96
        for name in names:
            frac = flow_dict[name] / total if total else 1 / len(names)
            h = max(frac * bar_total, 0.014)
            pos[name] = (x, (y + y - h) / 2, y - h, y, h, bw)
            y -= h + pad
        return pos

    kw_flows = {kw: sum(v for s, t, v in links_kw_th if s == kw) for kw in kw_names}
    th_in = {th: sum(v for s, t, v in links_kw_th if t == th) for th in THEME_ORDER}
    th_out = {th: sum(v for s, t, v in links_th_mo if s == th) for th in THEME_ORDER}
    th_flows = {th: max(th_in.get(th, 0), th_out.get(th, 0)) for th in THEME_ORDER}
    mo_in = {m: sum(v for s, t, v in links_th_mo if t == m) for m in MODEL_ORDER}
    mo_out = {m: sum(v for s, t, v in links_mo_bh if s == m) for m in MODEL_ORDER}
    mo_flows = {m: max(mo_in.get(m, 0), mo_out.get(m, 0)) for m in MODEL_ORDER}
    bh_flows = {b: sum(v for s, t, v in links_mo_bh if t == b) for b in BEHAVIOR_ORDER}

    kw_pos = layout_nodes(kw_names, 0, kw_flows)
    th_pos = layout_nodes(THEME_ORDER, 1, th_flows)
    mo_pos = layout_nodes(MODEL_ORDER, 2, mo_flows)
    bh_pos = layout_nodes(BEHAVIOR_ORDER, 3, bh_flows)

    # Draw nodes
    for kw in kw_names:
        c = THEME_COLORS[KEYWORD_GROUPS[kw]["theme"]]
        x, yc, yb, yt, h, bw = kw_pos[kw]
        ax.add_patch(plt.Rectangle((x - bw / 2, yb), bw, h, fc=c, ec='white', lw=1.2, zorder=3))
        fs = 11 if len(kw) > 22 else 13
        ax.text(
            x, yc, kw, ha='center', va='center', fontsize=fs, fontweight='bold',
            color='black', zorder=4, linespacing=0.95,
        )

    for th in THEME_ORDER:
        x, yc, yb, yt, h, bw = th_pos[th]
        c = THEME_COLORS[th]
        ax.add_patch(plt.Rectangle((x - bw / 2, yb), bw, h, fc=c, ec='white', lw=1.5, zorder=3))
        ax.text(x, yc, th, ha='center', va='center', fontsize=13, fontweight='bold',
                color='black', zorder=4, linespacing=0.95)

    for mo in MODEL_ORDER:
        x, yc, yb, yt, h, bw = mo_pos[mo]
        c = MODEL_COLORS[mo]
        ax.add_patch(plt.Rectangle((x - bw / 2, yb), bw, h, fc=c, ec='white', lw=1.5, zorder=3))
        ax.text(x, yc, mo, ha='center', va='center', fontsize=14, fontweight='bold',
                color='black', zorder=4)

    for bh in BEHAVIOR_ORDER:
        x, yc, yb, yt, h, bw = bh_pos[bh]
        c = BEHAVIOR_COLORS[bh]
        ax.add_patch(plt.Rectangle((x - bw / 2, yb), bw, h, fc=c, ec='white', lw=1.5, zorder=3))
        ax.text(
            x, yc, bh, ha='center', va='center', fontsize=13, fontweight='bold',
            color='black', zorder=4, linespacing=0.92,
        )

    # Draw links
    def draw_links(links, src_pos, tgt_pos, color_fn):
        total_out = {}
        total_in = {}
        for s, t, v in links:
            total_out[s] = total_out.get(s, 0) + v
            total_in[t] = total_in.get(t, 0) + v
        src_cur = {k: v[3] for k, v in src_pos.items()}
        tgt_cur = {k: v[3] for k, v in tgt_pos.items()}

        for s, t, v in links:
            sx, syc, syb, syt, sh, sw = src_pos[s]
            tx, tyc, tyb, tyt, th_, tw = tgt_pos[t]
            sb = (v / total_out[s]) * sh if total_out[s] else 0
            tb = (v / total_in[t]) * th_ if total_in[t] else 0
            s_top = src_cur[s]; s_bot = s_top - sb; src_cur[s] = s_bot
            t_top = tgt_cur[t]; t_bot = t_top - tb; tgt_cur[t] = t_bot
            draw_curved_band(ax, sx + sw / 2, s_bot, s_top, tx - tw / 2, t_bot, t_top, color_fn(s, t))

    draw_links(links_kw_th, kw_pos, th_pos,
               lambda s, t: THEME_COLORS[KEYWORD_GROUPS[s]["theme"]])
    draw_links(links_th_mo, th_pos, mo_pos, lambda s, t: MODEL_COLORS[t])
    draw_links(links_mo_bh, mo_pos, bh_pos, lambda s, t: BEHAVIOR_COLORS[t])

    for i, title in enumerate(["Keywords", "Themes", "Models", "Behaviors"]):
        ax.text(col_x[i], 0.994, title, ha='center', va='top', fontsize=17, fontweight='bold', color='black')

    out = f"{OUT}/S3_following_a_cyclist_sankey.png"
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    print("Loading data...")
    df = load_all()
    print(f"Total: {len(df)} rows")
    print("Computing links...")
    l1, l2, l3 = compute_links(df)
    print("\n--- Keywords -> Themes ---")
    for s, t, v in l1:
        print(f"  {s:>30s} -> {t:<25s}  {v:>5d}")
    print("\n--- Themes -> Models ---")
    for s, t, v in l2:
        print(f"  {s:>30s} -> {t:<25s}  {v:>5d}")
    print("\n--- Models -> Behaviors ---")
    for s, t, v in l3:
        print(f"  {s:>30s} -> {t:<25s}  {v:>5d}")
    print("\nDrawing...")
    plot_sankey(l1, l2, l3)
    print("Done.")
