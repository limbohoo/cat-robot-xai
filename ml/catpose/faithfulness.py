"""Is the explanation the real reason? Counterfactual checks on the deployed rules.

The course's second level of testing asks whether "what the system says it
responds to is what it actually responds to". Because the behaviour labels and
the possible-emotion guesses come from transparent rules, this can be checked
exactly, by re-running the same rules on the values each decision was made on
(`Event.z`, recorded by ReactionCoder) and changing one thing at a time:

1. reproduces      the recorded values give the recorded label again;
2. names_cause     putting the cited features back to this cat's usual value
                   (z = 0) changes the label: what the card names is what
                   decided it. Each cited feature is marked "main" if putting
                   it back alone changes the label, else "support";
3. nothing_hidden  putting every feature the card does NOT cite back to its
                   usual value leaves the label unchanged: nothing unnamed
                   decided it;
4. guess_reproduces the follow-up cues give the same guess and confidence
                   again (interpret is deterministic, so a mismatch means the
                   report and the rules have drifted apart);
5. cue effects     each follow-up cue is removed in turn; cues whose removal
                   changes the guess or its confidence are "decisive" and get a
                   "if this had not happened" sentence on the card.

An event passes when 1–4 hold. The checks say the explanation is faithful to
the rules; they say nothing about whether the rules are right about the cat
(that needs labelled clips, the course's first level).

    python -m catpose.faithfulness REPORT_DIR   # re-check a report folder's events.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Optional

from .interpret import DEFAULT_CONFIG, interpret, load_config
from .reactions import USED_FEATURES, as_features, classify

USUAL = 0.0  # z of this cat's usual value


def label_of(z: dict, prev: str) -> str:
    return classify(as_features(z), False, "", prev=prev).key


def check_event(reaction: str, z: dict, prev: str, drivers: list, interp: Optional[dict], cfg: dict,
                face_visible: Optional[float] = None) -> dict:
    """Run the checks for one event; see the module docstring."""
    if not z:
        return {"checked": False, "why": "no decision values recorded (report made before this check existed)"}
    cited = sorted({d["feature"] for d in drivers})
    res: dict = {"checked": True, "label": reaction, "prev": prev, "cited": cited}
    res["reproduces"] = label_of(z, prev) == reaction

    # 2. each cited feature back to usual, alone and all together
    alone = {}
    for f in cited:
        z2 = dict(z)
        z2[f] = USUAL
        alone[f] = label_of(z2, prev)
    res["if_usual"] = alone  # feature -> label it would get
    res["roles"] = {f: ("main" if new != reaction else "support") for f, new in alone.items()}
    z_all = {k: (USUAL if k in cited else v) for k, v in z.items()}
    res["if_all_usual"] = label_of(z_all, prev)
    res["names_cause"] = bool(cited) and res["if_all_usual"] != reaction

    # 3. everything not cited back to usual
    z_rest = {k: (v if k in cited else USUAL) for k, v in z.items()}
    res["nothing_hidden"] = label_of(z_rest, prev) == reaction
    res["not_used"] = sorted(k for k in z if k not in USED_FEATURES)  # features the rules never read

    # 4./5. the guess
    if interp:
        cues = list(interp.get("cues", []))
        face = interp.get("face_visible", 1.0) if face_visible is None else face_visible
        again = interpret(reaction, cues, face, cfg)
        res["guess_reproduces"] = bool(again) and again["top"] == interp["top"] and again["band"] == interp["band"]
        effects = {}
        for c in cues:
            alt = interpret(reaction, [x for x in cues if x != c], face, cfg)
            effects[c] = {"top": alt["top"], "band": alt["band"], "show": alt["show"]}
        res["cue_effects"] = effects
        res["decisive_cues"] = [c for c, a in effects.items()
                                if a["top"] != interp["top"] or a["band"] != interp["band"]]
        res["band_checked"] = interp["band"]
    else:
        res["guess_reproduces"] = True  # no guess to reproduce
    res["passed"] = bool(res["reproduces"] and res["names_cause"] and res["nothing_hidden"] and res["guess_reproduces"])
    return res


def check_events(events: list, cfg: dict) -> dict:
    """Check every event (objects with reaction, z, prev_reaction, drivers, interp, changes); attach
    the result as `event.faith` and return the summary."""
    n = passed = checked = old_missed = 0
    failures = []
    for i, ev in enumerate(events):
        r = check_event(ev.reaction, ev.z, ev.prev_reaction, ev.drivers, ev.interp, cfg)
        ev.faith = r
        n += 1
        if not r["checked"]:
            continue
        checked += 1
        passed += r["passed"]
        if not r["passed"]:
            failures.append({"event": i, "failed": [k for k in ("reproduces", "names_cause", "nothing_hidden",
                                                                 "guess_reproduces") if not r[k]]})
        # Would the earlier card (which listed only "changes in the last 2 s") have named the main reason?
        main = {f for f, role in r["roles"].items() if role == "main"}
        shown_before = {k for k, _ in ev.changes}
        r["old_card_named_main"] = bool(main & shown_before) if main else True
        old_missed += not r["old_card_named_main"]
    return {"events": n, "checked": checked, "passed": passed, "failures": failures,
            "old_card_missed_main": old_missed}


def main() -> None:
    from .events import Event  # local import: events imports this module
    p = argparse.ArgumentParser(description="Counterfactual faithfulness checks for a report folder")
    p.add_argument("report", help="folder with events.json")
    p.add_argument("--interp-config", default=None)
    args = p.parse_args()
    folder = Path(args.report)
    allowed = {f.name for f in fields(Event)}
    events = [Event(**{k: v for k, v in raw.items() if k in allowed})
              for raw in json.loads((folder / "events.json").read_text())]
    summary = check_events(events, load_config(args.interp_config) if args.interp_config else DEFAULT_CONFIG)
    for i, ev in enumerate(events):
        r = ev.faith
        if not r["checked"]:
            print(f"event {i}: not checked ({r['why']})")
            continue
        roles = ", ".join(f"{f} {role}" for f, role in r["roles"].items()) or "none"
        print(f"event {i}: {ev.reaction:9s} {'PASS' if r['passed'] else 'FAIL'}  cited: {roles}; "
              f"all cited -> usual gives {r['if_all_usual']}; decisive cues: {r.get('decisive_cues', [])}")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
