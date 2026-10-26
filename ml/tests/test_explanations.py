"""Tests for the explanation layer: the label's basis, the faithfulness checks, the owner-facing wording.

Run from ml/ (no pytest needed; pytest also collects these):

    PYTHONPATH=. .venv/bin/python tests/test_explanations.py
"""

from __future__ import annotations

import math

from catpose.events import Event
from catpose.faithfulness import check_event, check_events
from catpose.interpret import DEFAULT_CONFIG, interpret
from catpose.reactions import ReactionCoder, as_features, classify
from catpose.report_ui import explain_basis, faith_line, guess_counterfactuals, label_counterfactuals


def label(z: dict, prev: str = "resting"):
    return classify(as_features(z), False, prev=prev)


def features_of(r):
    return {d["feature"] for d in r.drivers}


# ---------- the rules record why ----------
def test_ears_back_from_ear_tip():
    r = label({"ear_height": -2.0, "activity": 0.2})
    assert r.key == "ears_back" and features_of(r) == {"ear_height"}
    assert r.drivers[0]["test"] == "<=" and r.drivers[0]["bar"] == -1.5


def test_ears_back_from_ear_angle_needs_ear_tip_too():
    r = label({"ear_height": -0.8, "ear_angle": 2.5})
    assert r.key == "ears_back" and features_of(r) == {"ear_angle", "ear_height"}
    tip = next(d for d in r.drivers if d["feature"] == "ear_height")
    assert tip["bar"] == -0.5  # the co-condition, not the main ear-tip threshold


def test_freeze_cites_stillness():
    r = label({"ear_height": -2.0, "activity": -1.5})
    assert r.key == "freeze" and features_of(r) == {"ear_height", "activity"}


def test_alert_with_eyes_as_support():
    r = label({"ear_height": 2.0, "eye_aperture": 2.5, "activity": 0.0})
    assert r.key == "alert" and features_of(r) == {"ear_height", "eye_aperture"}


def test_resting_has_no_basis():
    r = label({"ear_height": 0.1, "activity": 0.0})
    assert r.key == "resting" and r.drivers == []


def test_hysteresis_changes_the_threshold_in_the_basis():
    r = label({"ear_height": -1.2}, prev="ears_back")  # stays: exit bar is -1.0
    assert r.key == "ears_back" and r.drivers[0]["bar"] == -1.0


def test_coder_records_what_the_switch_was_decided_on():
    coder = ReactionCoder(hold_s=0.3, majority=0.6)
    t = 0.0
    for i in range(15):  # half a second of a calm cat
        coder.update(as_features({"ear_height": 0.0, "activity": 0.0}), False, t=t)
        t += 1 / 30
    for i in range(15):  # then the ears go down
        r = coder.update(as_features({"ear_height": -2.0 + 0.05 * (i % 3), "activity": 0.1}), False, t=t)
        t += 1 / 30
        if r.key == "ears_back" and r.snapshot:
            break
    assert r.key == "ears_back", r.key
    assert r.prev == "resting" and r.consistent
    assert abs(r.snapshot["ear_height"] + 2.0) < 0.2 and features_of(r) == {"ear_height"}


# ---------- the checks ----------
def test_honest_explanation_passes():
    z = {"ear_height": -2.0, "ear_angle": 0.3, "activity": 0.2, "muzzle_ratio": 1.4}
    r = label(z)
    res = check_event(r.key, z, "resting", r.drivers, None, DEFAULT_CONFIG)
    assert res["passed"] and res["roles"] == {"ear_height": "main"}
    assert res["if_usual"]["ear_height"] == "resting"
    assert res["not_used"] == ["muzzle_ratio"]  # the rules never read it


def test_wrong_explanation_is_caught():
    """A card that cites body movement when the ears decided the label must fail."""
    z = {"ear_height": -2.0, "activity": 0.2}
    wrong = [{"feature": "activity", "z": 0.2, "test": ">=", "bar": 2.0}]
    res = check_event("ears_back", z, "resting", wrong, None, DEFAULT_CONFIG)
    assert not res["names_cause"]  # putting activity back to usual changes nothing
    assert not res["nothing_hidden"]  # putting the uncited ears back to usual changes the label
    assert not res["passed"]


def test_main_and_support():
    z = {"ear_height": -2.0, "ear_angle": 2.5}
    r = label(z)
    res = check_event(r.key, z, "resting", r.drivers, None, DEFAULT_CONFIG)
    assert res["roles"] == {"ear_angle": "support", "ear_height": "main"}


def test_freeze_counterfactual_names_the_other_label():
    z = {"ear_height": -2.0, "activity": -1.5}
    r = label(z)
    res = check_event(r.key, z, "resting", r.drivers, None, DEFAULT_CONFIG)
    assert res["passed"] and res["if_usual"]["activity"] == "ears_back"


def test_decisive_follow_up_cue():
    g = interpret("ears_back", ["moves_away"], 1.0, DEFAULT_CONFIG)
    g["face_visible"] = 1.0
    z = {"ear_height": -2.0}
    res = check_event("ears_back", z, "resting", label(z).drivers, g, DEFAULT_CONFIG)
    assert res["guess_reproduces"] and res["decisive_cues"] == ["moves_away"]
    assert res["cue_effects"]["moves_away"]["show"] is False  # without it: too unsure to guess


# ---------- what the owner reads ----------
def _event(z, changes=()):
    r = label(z)
    ev = Event(start_s=1.0, reaction=r.key, changes=list(changes), drivers=r.drivers, z=z, prev_reaction="resting")
    ev.interp = interpret(r.key, ["moves_away"], 1.0, DEFAULT_CONFIG)
    ev.interp["face_visible"] = 1.0
    return ev


def test_owner_wording():
    ev = _event({"ear_height": -2.2, "ear_angle": 2.5})
    summary = check_events([ev], DEFAULT_CONFIG)
    basis = explain_basis(ev)
    assert basis[0]["role"] == "main" and basis[0]["text"] == "耳朵比它平时低（明显）"
    assert basis[1]["role"] == "support"
    assert label_counterfactuals(ev) == ["如果耳朵回到它平时的高度，就不会标为「耳朵压低」，而是「没有明显变化」。"]
    assert guess_counterfactuals(ev, DEFAULT_CONFIG)[0].startswith("关键的一步：随后离开或躲开。")
    assert faith_line(ev).startswith("已自动检验") and summary["passed"] == 1


def test_old_card_would_have_missed_the_reason():
    """The earlier card listed only 'changes in the last 2 s'; here that was the body, not the ears."""
    ev = _event({"ear_height": -2.0, "activity": -0.3}, changes=[("activity", -3.0)])
    summary = check_events([ev], DEFAULT_CONFIG)
    assert summary["old_card_missed_main"] == 1


def test_old_reports_are_marked_unchecked():
    ev = Event(start_s=0.0, reaction="ears_back")
    res = check_events([ev], DEFAULT_CONFIG)
    assert res["checked"] == 0 and ev.faith["checked"] is False and faith_line(ev) == ""


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except AssertionError as err:
            failed += 1
            print(f"FAIL  {name}: {err}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
