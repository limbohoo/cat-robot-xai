"""Owner-facing report page: summary first, moments as cards, details on demand.

Three levels of reading, each answering the owner's next question:

1. Summary  — how was the time alone? One sentence, a time-share bar, robot rounds.
2. Moments  — which moments are worth a look? Cards with a thumbnail and the guess.
3. Details  — is the guess right? Video first, then the guess, the runner-up, how to
              tell them apart, why it was guessed, before/after frames.

Low-confidence moments are folded away, technical numbers (SD, lift, rule version)
sit in a collapsed section for researchers, and every moment has a "is this guess
right?" control whose answers can be exported as CSV for the user test.
The page is one self-contained HTML file next to its clips/ folder; no network.
"""

from __future__ import annotations

import base64
import html
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

STATE_COLOR = {"tense": "#c2410c", "curious": "#2563eb", "playful": "#7c3aed", "attentive": "#0f766e",
               "calm": "#64748b"}
BUDGET_COLOR = ["#94a3b8", "#2563eb", "#c2410c", "#f59e0b", "#7c3aed", "#cbd5e1", "#e2e8f0"]
FEATURE_PLAIN = {  # (part, went down, went up)
    "ear_height": ("耳朵", "比刚才低了", "比刚才竖起来了"),
    "ear_angle": ("耳朵", "更朝前了", "向外转了"),
    "eye_aperture": ("眼睛", "眯起来了", "睁大了"),
    "muzzle_ratio": ("口鼻", "收紧了", "放松了"),
    "head_drop": ("头", "抬高了", "放低了"),
    "activity": ("身体", "静下来了", "动起来了"),
}
ACTION_PLAIN = {"slow_prey": "慢速逗猫", "light": "灯光", "sound": "声音", "approach": "慢慢靠近", "retreat": "后退离开"}
STATE_SHORT = {"tense": "紧张", "curious": "好奇", "playful": "想玩", "attentive": "专注", "calm": "放松"}
SHORT_RESPONSE = {"参与（靠近或扑玩）": "参与", "回避或离开": "回避", "没有明显反应": "没反应"}
REACTION_PLAIN = {
    "ears_back": "耳朵压低", "freeze": "突然不动", "alert": "注意到了什么", "active": "动起来了",
    "resting": "没有明显变化", "unknown": "看不清",
}


def _fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def _img(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode() if path.exists() else ""


def _plain_change(k: str, dz: float) -> str:
    part, down, up = FEATURE_PLAIN[k]
    amount = "一点" if abs(dz) < 2 else ("明显" if abs(dz) < 3 else "很明显")
    return f"{part}{down if dz < 0 else up}（{amount}）"


def _summary_sentence(budget: dict, moments: list, rounds: list) -> str:
    if not budget:
        return "这段时间没有看到猫。"
    top, share = next(iter(budget.items()))
    s = (f"这段时间它大部分时候很平静（{share:.0%}）。" if top == "平静"
         else f"这段时间它花得最多的是“{top}”（{share:.0%}）。")
    if moments:
        s += f"有 {len(moments)} 个时刻值得看一看。"
    if rounds:
        engaged = sum(1 for _, _, r in rounds if r.startswith("参与"))
        s += f"小车邀请了 {len(rounds)} 次，它参与了 {engaged} 次。"
    return s


TRIGGER_PLAIN = {"sound": "之前突然出现较大的声音", "motion": "之前画面里别的地方有东西在动"}
DRIVER_PLAIN = {  # (part, below its usual, above its usual): the label's basis, always "vs its usual"
    "ear_height": ("耳朵", "比它平时低", "比它平时竖得高"),
    "ear_angle": ("耳朵", "比它平时更朝前", "比它平时更向外转"),
    "eye_aperture": ("眼睛", "比它平时眯", "比它平时睁得大"),
    "activity": ("身体", "比它平时安静", "比它平时动得多"),
}
USUAL_PHRASE = {"ear_height": "耳朵回到它平时的高度", "ear_angle": "耳朵回到它平时的角度",
                "eye_aperture": "眼睛回到它平时的样子", "activity": "身体的活动回到平时的水平"}
FAITH_FAIL = {"reproduces": "用记录下来的数值重算，得不到同样的标签",
              "names_cause": "把写出的依据改回平时的样子，标签没有改变",
              "nothing_hidden": "还有没写出来的特征影响了判断",
              "guess_reproduces": "推测和规则重算的结果不一致"}


def _amount(z: float) -> str:
    a = abs(z)
    return "有些" if a < 2 else ("明显" if a < 3 else "很明显")


def explain_basis(ev) -> list[dict]:
    """Why the event got its label: one plain line per feature the rules used, main reasons first.

    A feature is "main" when putting it back to the cat's usual value alone changes the label
    (catpose.faithfulness), else "support". Reports made before this existed have no basis.
    """
    roles = (ev.faith or {}).get("roles", {})
    out, seen = [], set()
    for d in getattr(ev, "drivers", []) or []:
        k = d["feature"]
        if k in seen or k not in DRIVER_PLAIN:
            continue
        seen.add(k)
        part, down, up = DRIVER_PLAIN[k]
        out.append({"feature": k, "z": d["z"], "role": roles.get(k, "main"),
                    "text": f"{part}{down if d['z'] < 0 else up}（{_amount(d['z'])}）"})
    out.sort(key=lambda b: b["role"] != "main")
    return out


def label_counterfactuals(ev) -> list[str]:
    """'If X were back to usual, it would not be labelled Y but Z' for each main reason."""
    f = ev.faith or {}
    if not f.get("checked"):
        return []
    now = REACTION_PLAIN.get(ev.reaction, ev.reaction)
    out = [f"如果{USUAL_PHRASE.get(k, k)}，就不会标为「{now}」，而是「{REACTION_PLAIN.get(new, new)}」。"
           for k, new in f.get("if_usual", {}).items() if new != ev.reaction]
    if not out and f.get("if_all_usual") not in (None, ev.reaction):  # only the reasons together decide
        parts = "、".join(USUAL_PHRASE.get(k, k) for k in f.get("cited", []))
        out.append(f"如果{parts}，就不会标为「{now}」。")
    return out


def guess_counterfactuals(ev, cfg: dict) -> list[str]:
    """'The key step: X. Without it, ...' for each follow-up cue that changed the guess or its confidence."""
    g, f = ev.interp or {}, ev.faith or {}
    if not (g.get("show") and f.get("checked")) or f.get("band_checked") != g.get("band"):
        return []  # no guess shown, or another camera view changed the confidence afterwards
    out = []
    for c in f.get("decisive_cues", []):
        alt = f["cue_effects"][c]
        if not alt["show"]:
            tail = "把握就不够，不会给出推测"
        elif alt["top"] != g["top"]:
            tail = f"看起来更像「{cfg['state_text'][alt['top']]}」"
        else:
            tail = f"把握会从「{g['band']}」变为「{alt['band']}」"
        out.append(f"关键的一步：{cfg['cue_text'].get(c, c)}。如果没有这一步，{tail}。")
    return out


def faith_line(ev) -> str:
    f = ev.faith or {}
    if not f.get("checked"):
        return ""
    if f.get("passed"):
        return "已自动检验：把上面的依据改回它平时的样子，标签就会改变；其他特征改回平时的样子，标签不变。"
    failed = [FAITH_FAIL[k] for k in FAITH_FAIL if f.get(k) is False]
    return "自动检验没有通过：" + "；".join(failed) + "。这条解释请谨慎对待。"


def faith_summary(events) -> dict:
    checked = [ev.faith for ev in events if (ev.faith or {}).get("checked")]
    return {"events": len(events), "checked": len(checked), "passed": sum(1 for f in checked if f["passed"]),
            "old_card_missed_main": sum(1 for f in checked if f.get("old_card_named_main") is False)}


def build_report_data(events, source: str, duration_s: float, budget: dict, rounds: list, cfg: dict,
                      clock0: Optional[datetime] = None, session_info: Optional[list] = None,
                      stops: Optional[list] = None, alerts: Optional[list] = None) -> dict:
    """Everything the owner app (catpose/app.py) shows, as plain-language text; saved as report.json.

    The HTML report and the app read the same wording, so they never disagree.
    """
    def fmt(t: float) -> str:
        return (clock0 + timedelta(seconds=t)).strftime("%H:%M:%S") if clock0 else _fmt(t)
    st = cfg["state_text"]
    shown = [ev for ev in events if ev.interp and ev.interp["show"]]
    moments = []
    for i, ev in enumerate(events):
        g = ev.interp or {}
        show = bool(g.get("show"))
        moments.append({
            "id": i, "time": fmt(ev.start_s), "start_s": round(ev.start_s, 2),
            "duration_s": round((ev.end_s or ev.start_s) - ev.start_s, 1),
            "camera": getattr(ev, "camera", ""), "reaction": ev.reaction,
            "reaction_text": REACTION_PLAIN.get(ev.reaction, ev.reaction),
            "show": show, "guess": g.get("top", ""), "guess_text": st.get(g.get("top"), "") if show else "",
            "band": g.get("band", ""), "color": STATE_COLOR.get(g.get("top"), "#64748b") if show else "#64748b",
            "second_text": st.get(g.get("second"), "") if show else "",
            "tell_apart": g.get("tell_apart", "") if show else "",
            "trail": [t + ((" " + a) if a else "") for t, a in g.get("trail", [])] if show else [],
            "basis": explain_basis(ev),
            "basis_if": label_counterfactuals(ev),
            "guess_if": guess_counterfactuals(ev, cfg),
            "faith_ok": (ev.faith or {}).get("passed") if (ev.faith or {}).get("checked") else None,
            "faith_text": faith_line(ev),
            "basis_marked": any(b["feature"] in ("ear_height", "ear_angle", "eye_aperture") for b in explain_basis(ev)),
            "changes": [_plain_change(k, dz) for k, dz in ev.changes],
            "before": sorted({TRIGGER_PLAIN[k] for k, _, _ in ev.triggers if k in TRIGGER_PLAIN}),
            "notes": list(ev.notes),
            "clip": ev.clip, "before_img": ev.before_img, "after_img": ev.after_img,
            "alt_views": [{"name": n, "clip": c} for n, c in (getattr(ev, "alt_views", []) or [])],
        })
    return {
        "title": Path(source).stem, "source": Path(source).name,
        "period": f"{fmt(0)}–{fmt(duration_s)}" if clock0 else f"时长 {_fmt(duration_s)}",
        "duration_s": round(duration_s, 1),
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "summary": _summary_sentence(budget, shown, rounds),
        "alerts": list(alerts or []),
        "budget": [{"label": k, "share": round(v, 4), "color": BUDGET_COLOR[j % len(BUDGET_COLOR)]}
                   for j, (k, v) in enumerate(budget.items())],
        "rounds": [{"time": fmt(t), "action": ACTION_PLAIN.get(a, a), "response": SHORT_RESPONSE.get(r, r),
                    "kind": "ok" if r.startswith("参与") else ("no" if r.startswith("回避") else "na")}
                   for t, a, r in rounds],
        "stops": [{"time": fmt(t), "detail": d} for t, d in (stops or [])],
        "session_info": list(session_info or []),
        "states": {k: {"text": v, "short": STATE_SHORT.get(k, v), "color": STATE_COLOR.get(k, "#64748b")}
                   for k, v in st.items()},
        "rule_version": cfg.get("version", ""),
        "faithfulness": faith_summary(events),
        "moments": moments,
    }


def render_owner_report(events, source: str, duration_s: float, root: Path, budget: dict, rounds: list,
                        cfg: dict, rates: dict, n_brief: int, has_audio: bool, clock0: Optional[datetime] = None,
                        session_info: Optional[list] = None, stops: Optional[list] = None,
                        alerts: Optional[list] = None) -> str:
    """`clock0` (session start) shows wall-clock times; `session_info` lists cameras and sync;
    `stops` lists robot safety stops (t, detail). Events may carry `camera` and `alt_views`.
    Also writes report.json (the owner app's data) next to the report."""
    data = build_report_data(events, source, duration_s, budget, rounds, cfg, clock0, session_info, stops, alerts)
    (Path(root) / "report.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    e = html.escape

    def fmt(t: float) -> str:
        return (clock0 + timedelta(seconds=t)).strftime("%H:%M:%S") if clock0 else _fmt(t)
    st = cfg["state_text"]
    shown = [(i, ev) for i, ev in enumerate(events) if ev.interp and ev.interp["show"]]
    unsure = [(i, ev) for i, ev in enumerate(events) if not (ev.interp and ev.interp["show"])]

    # ---------- level 1: summary ----------
    bar = "".join(f"<span style='flex:{v:.4f};background:{BUDGET_COLOR[j % len(BUDGET_COLOR)]}'></span>"
                  for j, v in enumerate(budget.values()))
    legend = "".join(f"<li><i style='background:{BUDGET_COLOR[j % len(BUDGET_COLOR)]}'></i>{e(k)} <b>{v:.0%}</b></li>"
                     for j, (k, v) in enumerate(budget.items()))
    robot = ""
    if rounds:
        chips = "".join(
            f"<li class='r {'ok' if r.startswith('参与') else ('no' if r.startswith('回避') else 'na')}'>"
            f"<span>{fmt(t)}</span><b>{e(SHORT_RESPONSE.get(r, r))}</b><small>{e(ACTION_PLAIN.get(a, a))}</small></li>" for t, a, r in rounds)
        robot = f"<section class='panel'><h2>小车邀请它玩的时候</h2><ul class='rounds'>{chips}</ul></section>"
    session = ""
    if session_info or stops:
        items = "".join(f"<li>{e(x)}</li>" for x in (session_info or []))
        items += "".join(f"<li>{fmt(t)} 小车按安全规则停止{('：' + e(d)) if d else ''}</li>" for t, d in (stops or []))
        session = f"<details class='fold'><summary>这次记录的情况（摄像头、时间对齐、小车停止）</summary><ul class='info'>{items}</ul></details>"

    # ---------- level 2 + 3: moments ----------
    def moment(i, ev, low=False):
        g = ev.interp or {}
        cam = getattr(ev, "camera", "")
        alts = "".join(f"<details class='alt'><summary>换个角度看（{e(name)}）</summary>"
                       f"<video controls playsinline preload='none' src='{e(clip)}'></video></details>"
                       for name, clip in getattr(ev, "alt_views", []) or [])
        color = STATE_COLOR.get(g.get("top"), "#64748b")
        title = f"看起来像：{st[g['top']]}" if not low else REACTION_PLAIN.get(ev.reaction, ev.reaction)
        badge = (f"<span class='badge' style='--c:{color}'>推测 · 把握{e(g['band'])}</span>" if not low
                 else "<span class='badge muted'>把握不够，不做推测</span>")
        changes = "".join(f"<li>{e(_plain_change(k, dz))}</li>" for k, dz in ev.changes) or "<li>变化不大</li>"
        notes = "".join(f"<p class='note'>⚠ {e(n)}</p>" for n in ev.notes)
        basis = explain_basis(ev)
        why_label = ""
        if basis:
            items = "".join(f"<li><span class='role {b['role']}'>{'主要' if b['role'] == 'main' else '补充'}</span>"
                            f"{e(b['text'])}</li>" for b in basis)
            ifs = "".join(f"<p class='if'>{e(x)}</p>" for x in label_counterfactuals(ev))
            check = f"<p class='fine'>{e(faith_line(ev))}</p>" if faith_line(ev) else ""
            why_label = (f"<div class='basis'><p class='sub'>为什么标为「{e(REACTION_PLAIN.get(ev.reaction, ev.reaction))}」</p>"
                         f"<ul>{items}</ul>{ifs}{check}</div>")
        marked = any(b["feature"] in ("ear_height", "ear_angle", "eye_aperture") for b in basis)
        detail = ""
        if not low:
            trail = "".join(f"<li>{e(t)}{(' <b>' + e(a) + '</b>') if a else ''}</li>" for t, a in g["trail"])
            tell = f"<p class='tell'><b>怎么分辨：</b>{e(g['tell_apart'])}</p>" if g.get("tell_apart") else ""
            key_steps = "".join(f"<p class='if'>{e(x)}</p>" for x in guess_counterfactuals(ev, cfg))
            detail = f"""
      <div class="guess" style="--c:{color}">
        <p>也可能是：<b>{e(st[g['second']])}</b></p>{tell}{key_steps}
      </div>
      <details class="why"><summary>为什么这么推测？</summary><ol>{trail}</ol>
        <p class="fine">推测会随猫随后的行为更新；它依据的是可见的动作和文献，不是诊断。</p></details>"""
        return f"""
<details class="moment" id="m{i}" style="--c:{color}">
  <summary>
    <img class="thumb" src="{_img(root / ev.after_img)}" alt="片段 {i} 的画面">
    <div class="meta"><p class="time">片段 {i} · {fmt(ev.start_s)}{(' · ' + e(cam)) if cam else ''}</p><p class="ttl">{e(title)}</p>{badge}</div>
    <span class="chev" aria-hidden="true">›</span>
  </summary>
  <div class="body">
    <video controls playsinline preload="none" poster="{_img(root / ev.after_img)}" src="{e(ev.clip)}"></video>{alts}{why_label}{detail}
    <div class="pair">
      <figure><img src="{_img(root / ev.before_img)}" alt="变化前"><figcaption>变化前</figcaption></figure>
      <figure><img src="{_img(root / ev.after_img)}" alt="变化后{'，红圈是判断依据的部位' if marked else ''}"><figcaption>变化后{'（红圈：判断依据的部位）' if marked else ''}</figcaption></figure>
    </div>
    {notes}<p class="sub">和 2 秒前相比</p><ul class="changes">{changes}</ul>
    <div class="fb" data-id="{i}" data-guess="{e(g.get('top', ''))}">
      <p class="sub">{'这个推测对吗？' if not low else '你觉得它当时怎么了？'}</p>
      <div class="btns"><button data-v="right">对</button><button data-v="wrong">不对</button><button data-v="unsure">说不准</button></div>
      <input type="text" placeholder="想补充什么？（可选）" aria-label="补充说明">
    </div>
  </div>
</details>"""

    moments = "".join(moment(i, ev) for i, ev in shown) or "<p class='empty'>这段时间没有把握足够高的推测。</p>"
    unsure_block = ""
    if unsure:
        unsure_block = (f"<details class='fold'><summary>另有 {len(unsure)} 个时刻看不太清（展开）</summary>"
                        + "".join(moment(i, ev, low=True) for i, ev in unsure) + "</details>")

    # ---------- researcher details ----------
    kind_zh = {"sound": "声音突然变大", "motion": "画面其他地方在动"}
    rate_rows = ""
    for kind, zh in kind_zh.items():
        if kind in rates:
            before = sum(any(k == kind for k, _, _ in ev.triggers) for ev in events) / max(len(events), 1)
            lift = before / rates[kind] if rates[kind] else float("inf")
            rate_rows += f"<tr><td>{zh}</td><td>{before:.0%}</td><td>{rates[kind]:.0%}</td><td>{lift:.1f}×</td></tr>"
    ev_rows = "".join(
        f"<tr><td>{i}</td><td>{fmt(ev.start_s)}</td><td>{e(ev.reaction)}</td>"
        f"<td>{e(', '.join(f'{k} {dz:+.1f}' for k, dz in ev.changes))}</td>"
        f"<td>{e(', '.join((ev.interp or {}).get('cues', [])))}</td>"
        f"<td>{e(json.dumps((ev.interp or {}).get('posterior', {}), ensure_ascii=False))}</td></tr>"
        for i, ev in enumerate(events))
    fs = faith_summary(events)
    faith_rows = ""
    for i, ev in enumerate(events):
        f = ev.faith or {}
        if not f.get("checked"):
            continue
        roles = "，".join(f"{k}:{'主' if r == 'main' else '辅'}" for k, r in f["roles"].items()) or "—"
        alone = "，".join(f"{k}→{v}" for k, v in f["if_usual"].items()) or "—"
        faith_rows += (f"<tr><td>{i}</td><td>{e(ev.reaction)}</td><td>{e(roles)}</td><td>{e(alone)}</td>"
                       f"<td>{e(f['if_all_usual'])}</td><td>{'是' if f['nothing_hidden'] else '否'}</td>"
                       f"<td>{e('、'.join(f.get('decisive_cues', [])) or '—')}</td><td>{'通过' if f['passed'] else '未通过'}</td></tr>")
    faith_block = (f"<p>解释忠实性检验（catpose/faithfulness.py）：{fs['checked']} 个事件中 {fs['passed']} 个通过。"
                   f"其中 {fs['old_card_missed_main']} 个事件，旧版卡片（只列“和 2 秒前相比”的变化）没有写出主要依据。</p>"
                   "<table><tr><th>#</th><th>标签</th><th>依据（主 / 辅）</th><th>单独改回平时 → 标签</th>"
                   f"<th>全部改回 → 标签</th><th>其他特征无影响</th><th>决定性线索</th><th>结果</th></tr>{faith_rows}</table>"
                   ) if fs["checked"] else ""
    tech = f"""
<details class="fold tech"><summary>技术细节（给研究者）</summary>
  <p>规则版本：{e(cfg['version'])}。依据：{e(cfg['sources'])}。另有 {n_brief} 次短于 0.5 秒的变化未列出。{'' if has_audio else '来源没有音频，未检查声音线索。'}</p>
  <table><tr><th>线索</th><th>事件前出现</th><th>平时出现</th><th>倍数</th></tr>{rate_rows}</table>
  <table><tr><th>#</th><th>时间</th><th>标签</th><th>特征变化 (SD)</th><th>后续线索</th><th>推测分布</th></tr>{ev_rows}</table>
  {faith_block}
</details>"""

    # One feedback store per generated report, so a regenerated report never inherits old answers.
    report_id = e(Path(source).stem)[:40] + ":" + str(int(time.time()))
    return (PAGE.replace("{{TITLE}}", "你不在的时候")
            .replace("{{SUB}}", f"{e(Path(source).name)} · "
                     + (f"{fmt(0)}–{fmt(duration_s)}" if clock0 else f"时长 {_fmt(duration_s)}"))
            .replace("{{SUMMARY}}", e(_summary_sentence(budget, shown, rounds)))
            .replace("{{ALERTS}}", "".join(f"<p class='alert'>⚠ {e(a)}</p>" for a in (alerts or [])))
            .replace("{{BAR}}", bar).replace("{{LEGEND}}", legend).replace("{{ROBOT}}", robot + session)
            .replace("{{MOMENTS}}", moments).replace("{{UNSURE}}", unsure_block).replace("{{TECH}}", tech)
            .replace("{{REPORT_ID}}", report_id))


PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>你不在的时候 · 猫咪报告</title>
<style>
:root{--bg:#f6f5f2;--card:#fff;--fg:#1f2430;--muted:#5f6675;--line:#e7e4dd;--soft:#f1efe9;--accent:#c2410c}
@media (prefers-color-scheme:dark){:root{--bg:#14161c;--card:#1d2028;--fg:#e9eaee;--muted:#a4a9b6;--line:#2e323d;--soft:#252934}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 -apple-system,"PingFang SC","Noto Sans SC",sans-serif}
main{max-width:720px;margin:0 auto;padding:28px 16px 64px}
header h1{font-size:30px;margin:0}
header .sub{color:var(--muted);margin:2px 0 16px;font-size:14px}
.lead{font-size:19px;font-weight:600;margin:0 0 16px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px 18px;margin:0 0 16px}
.panel h2{font-size:16px;margin:0 0 10px}
.bar{display:flex;height:16px;border-radius:8px;overflow:hidden;gap:2px}
.legend{list-style:none;padding:0;margin:10px 0 0;display:flex;flex-wrap:wrap;gap:6px 16px;font-size:14px;color:var(--muted)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px}
.legend b{color:var(--fg)}
.rounds{list-style:none;padding:0;margin:0;display:grid;grid-template-columns:repeat(auto-fit,minmax(96px,1fr));gap:8px}
.rounds li{border-radius:12px;padding:10px 12px;background:var(--soft);display:flex;flex-direction:column}
.rounds span{font-size:13px;color:var(--muted)} .rounds small{color:var(--muted)}
.rounds .ok b{color:#2563eb} .rounds .no b{color:#c2410c}
h3.sec{font-size:18px;margin:24px 0 10px}
.moment{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--c);border-radius:16px;margin:0 0 12px;overflow:hidden}
.moment summary{list-style:none;display:flex;gap:14px;align-items:center;padding:12px;cursor:pointer}
.moment summary::-webkit-details-marker{display:none}
.thumb{width:112px;height:72px;object-fit:cover;border-radius:10px;flex:none;background:var(--soft)}
.meta{flex:1;min-width:0} .meta p{margin:0}
.time{font-size:13px;color:var(--muted)} .ttl{font-size:18px;font-weight:700;color:var(--c)}
.badge{display:inline-block;margin-top:4px;font-size:12px;border:1.5px solid var(--c);color:var(--c);border-radius:999px;padding:0 9px}
.badge.muted{border-color:var(--line);color:var(--muted)}
.chev{font-size:26px;color:var(--muted);transition:transform .2s}
details[open]>summary .chev{transform:rotate(90deg)}
.body{padding:0 14px 16px}
video{width:100%;border-radius:12px;background:#000;display:block}
.guess{margin:14px 0 6px;padding:12px 14px;border-radius:12px;background:var(--soft)}
.guess .big{font-size:22px;font-weight:700;color:var(--c);margin:0 0 4px}
.guess p{margin:2px 0} .tell{color:var(--muted);font-size:15px}
.why{margin:6px 0} .why summary{cursor:pointer;color:var(--muted);font-size:15px}
.why ol{margin:6px 0;padding-left:22px;font-size:15px} .fine{color:var(--muted);font-size:13px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:12px 0 4px}
.pair img{width:100%;border-radius:10px;display:block} figure{margin:0}
figcaption{font-size:12px;color:var(--muted);margin-top:2px}
.sub{font-size:14px;font-weight:700;margin:12px 0 4px}
.changes{margin:0;padding-left:20px;font-size:15px}
.basis{margin:14px 0 4px;padding:12px 14px;border-radius:12px;border:1.5px solid var(--line)}
.basis ul{list-style:none;margin:4px 0 6px;padding:0;display:flex;flex-direction:column;gap:4px;font-size:15.5px}
.role{display:inline-block;font-size:12px;border-radius:6px;padding:0 6px;margin-right:8px;border:1.5px solid var(--line);color:var(--muted)}
.role.main{border-color:var(--fg);color:var(--fg);font-weight:700}
.if{font-size:14.5px;margin:6px 0 0}
.fb .btns{display:flex;gap:8px}
.fb button{flex:1;padding:9px 0;border-radius:10px;border:1.5px solid var(--line);background:var(--card);color:var(--fg);font:inherit;cursor:pointer}
.fb button.on{border-color:var(--c);background:var(--c);color:#fff}
.fb input{width:100%;margin-top:8px;padding:9px 12px;border-radius:10px;border:1.5px solid var(--line);background:var(--card);color:var(--fg);font:inherit}
.fold{margin:12px 0}
.alt{margin:8px 0} .alt summary{cursor:pointer;color:var(--muted);font-size:15px} .alt video{margin-top:6px}
.info{margin:4px 0 0;padding-left:20px;font-size:14px;color:var(--muted)}
.fold>summary{cursor:pointer;color:var(--muted);padding:8px 0}
.empty{color:var(--muted)}
.alert{background:#fef3c7;color:#78350f;border-radius:12px;padding:10px 14px;margin:0 0 10px;font-size:15px}
.note{font-size:14px;color:var(--muted);margin:10px 0 0}
.howto{font-size:14px;color:var(--muted);border-left:3px solid var(--line);padding:4px 12px;margin:20px 0}
.tech table{width:100%;border-collapse:collapse;font-size:12px;margin:8px 0;display:block;overflow-x:auto}
.tech td,.tech th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}
.export{width:100%;padding:12px;border-radius:12px;border:0;background:var(--fg);color:var(--bg);font:inherit;font-weight:700;cursor:pointer;margin-top:8px}
</style></head><body><main>
<header><h1>{{TITLE}}</h1><p class="sub">{{SUB}}</p></header>
{{ALERTS}}
<p class="lead">{{SUMMARY}}</p>
<section class="panel"><h2>它的时间都花在哪里</h2><div class="bar">{{BAR}}</div><ul class="legend">{{LEGEND}}</ul></section>
{{ROBOT}}
<h3 class="sec">值得看的时刻</h3>
{{MOMENTS}}
{{UNSURE}}
<p class="howto">“看起来像”是根据看得见的动作和文献做的<b>推测</b>，会随猫随后的行为更新，不是诊断。点开视频，用你对它的了解来判断。尾巴、胡须、瞳孔和叫声不在分析范围内。</p>
<button class="export" id="export">导出我的反馈（CSV）</button>
{{TECH}}
</main>
<script>
(function(){
  var KEY = "catreport:{{REPORT_ID}}";
  var store = {};
  try { store = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
  function save(){ try { localStorage.setItem(KEY, JSON.stringify(store)); } catch (e) {} }
  document.querySelectorAll(".fb").forEach(function(fb){
    var id = fb.dataset.id, rec = store[id] || {};
    var input = fb.querySelector("input");
    fb.querySelectorAll("button").forEach(function(b){
      if (rec.v === b.dataset.v) b.classList.add("on");
      b.addEventListener("click", function(){
        fb.querySelectorAll("button").forEach(function(x){ x.classList.remove("on"); });
        b.classList.add("on"); store[id] = Object.assign(store[id] || {}, {v: b.dataset.v, guess: fb.dataset.guess}); save();
      });
    });
    if (rec.note) input.value = rec.note;
    input.addEventListener("change", function(){ store[id] = Object.assign(store[id] || {}, {note: input.value, guess: fb.dataset.guess}); save(); });
  });
  document.querySelectorAll("details.moment").forEach(function(d){
    d.addEventListener("toggle", function(){ if (!d.open) { var v = d.querySelector("video"); if (v) v.pause(); } });
  });
  document.getElementById("export").addEventListener("click", function(){
    var rows = [["event_id","system_guess","owner_feedback","note"]];
    Object.keys(store).forEach(function(id){ var r = store[id]; rows.push([id, r.guess || "", r.v || "", (r.note || "").replace(/"/g, '""')]); });
    var csv = rows.map(function(r){ return r.map(function(c){ return '"' + c + '"'; }).join(","); }).join("\\n");
    var a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob(["\\ufeff" + csv], {type: "text/csv"}));
    a.download = "owner_feedback.csv"; a.click();
  });
})();
</script>
</body></html>"""
