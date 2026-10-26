"""Possible-emotion interpretation of a reaction event, updated by what the cat does next.

This is NOT emotion recognition. Each event starts from a prior over a few
candidate interpretations, set by the reaction label (e.g. "ears back" is often
linked to tension, but also to attentive listening). Cues observed in the
seconds after the change (moving away, approaching, pouncing, settling back,
staying still, a robot action just before) then update it with naive Bayes:

    P(state | cues) ∝ P(state | reaction) · Π P(cue | state)

The priors and likelihoods are hand-set hypotheses from the literature
directions documented in docs/反应识别与事件报告.md. After a test they can be
re-estimated from human-labelled events with `learn()` (Dirichlet / Beta counts
on top of the hand-set values, so a handful of labels nudges rather than
overwrites them).

The output is shown to the owner as "looks like X (a guess, confidence ...)",
always with the runner-up and the evidence trail; a low-confidence result shows
no emotion at all.
"""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Optional

STATES = ["tense", "curious", "playful", "attentive", "calm"]
FACE_BASED = {"ears_back", "freeze", "alert"}  # reactions whose evidence is the face

DEFAULT_CONFIG = {
    "version": "hand-set v1 (hypotheses, not validated)",
    "state_text": {
        "tense": "紧张 / 害怕", "curious": "警惕 / 好奇", "playful": "想玩 / 兴奋",
        "attentive": "专注地听或看", "calm": "放松",
    },
    # P(state | reaction label)
    "prior": {
        "ears_back": {"tense": .40, "curious": .15, "playful": .05, "attentive": .30, "calm": .10},
        "freeze":    {"tense": .45, "curious": .05, "playful": .20, "attentive": .25, "calm": .05},
        "alert":     {"tense": .15, "curious": .50, "playful": .05, "attentive": .25, "calm": .05},
        "active":    {"tense": .15, "curious": .30, "playful": .40, "attentive": .05, "calm": .10},
    },
    # P(cue observed | state)
    "likelihood": {
        "moves_away":   {"tense": .60, "curious": .20, "playful": .10, "attentive": .10, "calm": .05},
        "flees":        {"tense": .55, "curious": .08, "playful": .10, "attentive": .04, "calm": .02},
        "approaches":   {"tense": .05, "curious": .50, "playful": .50, "attentive": .20, "calm": .10},
        "plays":        {"tense": .05, "curious": .20, "playful": .70, "attentive": .05, "calm": .05},
        "settles":      {"tense": .15, "curious": .30, "playful": .20, "attentive": .50, "calm": .60},
        "stays_still":  {"tense": .50, "curious": .20, "playful": .30, "attentive": .40, "calm": .30},
        "robot_before": {"tense": .30, "curious": .50, "playful": .60, "attentive": .40, "calm": .20},
    },
    "cue_text": {
        "moves_away": "随后离开或躲开", "flees": "随后快速跑开", "approaches": "随后靠近", "plays": "随后扑玩、追逐",
        "settles": "随后很快平静下来", "stays_still": "随后保持不动", "robot_before": "之前小车刚有动作",
    },
    # what to look at in the video to tell the top two apart
    "tell_apart": {
        "tense|attentive": "随后躲起来更像紧张；随后去查看或继续原来的事，更像只是在专注听",
        "tense|playful": "随后逃开更像紧张；随后扑出去更像准备玩",
        "curious|tense": "随后靠近、嗅闻更像好奇；随后后退更像警惕或紧张",
        "playful|curious": "有扑、追的动作更像想玩；只是靠近看看更像好奇",
        "attentive|calm": "耳朵和头一直朝向某处更像专注；很快躺回去更像放松",
    },
    "sources": "Feline Grimace Scale (Evangelista 2019); Cats Protection body-language guide; "
               "predatory sequence (Leyhausen 1979); Bennett et al. 2017",
    "follow_s": 5.0,       # seconds after the change in which cues are collected
    "pseudo_count": 10.0,  # weight of the hand-set values when learning from labels
}


def load_config(path: Optional[str]) -> dict:
    cfg = deepcopy(DEFAULT_CONFIG)
    if path and Path(path).exists():
        cfg.update(json.loads(Path(path).read_text()))
    return cfg


def interpret(reaction: str, cues: list[str], face_visible: float, cfg: dict) -> Optional[dict]:
    """Posterior over STATES plus the update trail and a confidence band; None for reactions with no prior."""
    prior = cfg["prior"].get(reaction)
    if prior is None:
        return None
    post = {s: prior[s] for s in STATES}
    norm = lambda d: {k: v / sum(d.values()) for k, v in d.items()}
    post = norm(post)
    trail = [("起点：" + _top_text(post, cfg), None)]
    for cue in cues:
        lik = cfg["likelihood"].get(cue)
        if lik is None:
            continue
        before_top = max(post, key=post.get)
        before_p = post[before_top]
        post = norm({s: post[s] * lik[s] for s in STATES})
        top = max(post, key=post.get)
        if top != before_top:
            arrow = f"→ 变为 {cfg['state_text'][top]}"
        else:
            arrow = "↑" if post[top] > before_p + 0.005 else ("↓" if post[top] < before_p - 0.005 else "")
        trail.append((f"{cfg['cue_text'][cue]}：{_top_text(post, cfg)}", arrow))
    ranked = sorted(post.items(), key=lambda kv: -kv[1])
    (s1, p1), (s2, p2) = ranked[0], ranked[1]
    if reaction in FACE_BASED and face_visible < 0.3:
        band = "低"  # the label rests on ears and eyes that were mostly not visible
    elif p1 >= 0.6 and p1 - p2 >= 0.3:
        band = "高"
    elif p1 >= 0.45 and p1 - p2 >= 0.15:
        band = "中"
    else:
        band = "低"
    if band == "高" and face_visible < 0.3:
        band = "中"  # body movement alone never earns high confidence
    key = f"{s1}|{s2}" if f"{s1}|{s2}" in cfg["tell_apart"] else f"{s2}|{s1}"
    return {
        "top": s1, "top_p": round(p1, 2), "second": s2, "second_p": round(p2, 2), "band": band,
        "show": band != "低", "posterior": {k: round(v, 3) for k, v in post.items()},
        "cues": cues, "trail": trail, "tell_apart": cfg["tell_apart"].get(key, ""),
    }


def _top_text(post: dict, cfg: dict) -> str:
    s = max(post, key=post.get)
    return f"{cfg['state_text'][s]} {post[s]:.0%}"


def learn(events_json: str, labels_csv: str, cfg: dict) -> dict:
    """Re-estimate priors and likelihoods from labelled events (Dirichlet / Beta counts on the hand-set values).

    labels_csv columns: event_id, your_label (one of STATES, or 'unsure' to skip).
    """
    events = {str(i): e for i, e in enumerate(json.loads(Path(events_json).read_text()))}
    k = cfg["pseudo_count"]
    prior_counts = {r: {s: p * k for s, p in d.items()} for r, d in cfg["prior"].items()}
    lik_counts = {c: {s: [p * k, (1 - p) * k] for s, p in d.items()} for c, d in cfg["likelihood"].items()}
    used = 0
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ev, lab = events.get(row["event_id"].strip()), row["your_label"].strip()
            if ev is None or lab not in STATES or ev["reaction"] not in prior_counts:
                continue
            used += 1
            prior_counts[ev["reaction"]][lab] += 1
            seen = set((ev.get("interp") or {}).get("cues", []))
            for c in lik_counts:
                lik_counts[c][lab][0 if c in seen else 1] += 1
    new = deepcopy(cfg)
    new["prior"] = {r: {s: round(v / sum(d.values()), 3) for s, v in d.items()} for r, d in prior_counts.items()}
    new["likelihood"] = {c: {s: round(a / (a + b), 3) for s, (a, b) in d.items()} for c, d in lik_counts.items()}
    new["version"] = f"learned from {used} labelled events on top of hand-set values"
    return new


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Update interpretation priors from labelled events")
    p.add_argument("--events", required=True, help="events.json from a --report-dir")
    p.add_argument("--labels", required=True, help="labels CSV (event_id, your_label)")
    p.add_argument("--config", default=None, help="current config JSON (default: hand-set values)")
    p.add_argument("--out", required=True, help="where to write the updated config JSON")
    args = p.parse_args()
    new = learn(args.events, args.labels, load_config(args.config))
    Path(args.out).write_text(json.dumps(new, ensure_ascii=False, indent=1))
    print(new["version"], "->", args.out)


if __name__ == "__main__":
    main()
