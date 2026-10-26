**English** | [中文](README.zh.md)

# CatPose Robot: a cat-play robot that explains itself

Code for the TU/e course project *AI Interactive Robot for Cats*. The system finds the cat in video, marks 20 body points and 48 face points, compares ears, eyes and activity with this cat's own calm state, assigns a behaviour label and a possible emotion with rules and naive Bayes, and writes the evidence into a report an owner can read. On the robot side, a researcher drives the robot during tests and every action is logged with a timestamp, so the report can show what the cat did after each action.

## What is measured and what is assumed

| Part | Status |
| --- | --- |
| Cat detection and keypoints | Trained and evaluated on public datasets (numbers below) |
| Features, per-cat baseline, behaviour rules | Run on real video; thresholds are hand-set |
| Emotion guess (five states) | Hand-set priors and likelihoods; **not validated against human labels** |
| Explanations | Checked for faithfulness: each card states the conditions the rules actually used |
| Robot actions | Triggered by a researcher (Wizard-of-Oz); the emotion output is not connected to the robot |

The system does **not** recognise emotions. It reports what changed in the cat's face and body, and what that change might mean.

## Repository layout

| Folder | Contents |
| --- | --- |
| `ml/catpose/` | Detection, keypoints, features, rules, Bayesian inference, faithfulness checks, owner report, live analysis, multi-camera sessions, cat identity by coat colour, robot bridge |
| `ml/checkpoints/` | Trained keypoint weights `body.pt` and `face.pt` (see the licence notes) |
| `ml/tests/` | Tests for the explanation layer |
| `ml/docs/` | English docs: architecture, API, quickstart, training |
| `ml/colab/` | Colab theory tutorial and the script that builds it |
| `ml/scripts/` | Dataset checks, ONNX export, Colab data pack |
| `ml/logs/`, `ml/splits/` | Evaluation results, speed tests, the fixed validation split |
| `ml/*.md` | Chinese docs: training report, run guide, references |
| `hardware/car-bridge/` | Example robot firmware that accepts commands, and the integration guide |
| `hardware/unoq/` | Arduino UNO Q edge app |
| [`docs/analysis.md`](docs/analysis.md) | Analysis of the Colab tutorial, the XAI Fieldnotes site and the owner app |

Not included: training datasets (download them as described in [ml/docs/training.md](ml/docs/training.md)), test recordings (they show teammates' homes), dataset images, and frames of third-party videos.

## Quick start

```bash
python3 -m venv ml/.venv
ml/.venv/bin/pip install -r requirements.txt

# Explanation-layer tests
cd ml && PYTHONPATH=. .venv/bin/python tests/test_explanations.py && cd ..

# Live camera analysis; writes an owner report when you quit
PYTHONPATH=ml ml/.venv/bin/python -m catpose.realtime --source 0 --detect --report-dir sessions/live1

# Analyse a video file
PYTHONPATH=ml ml/.venv/bin/python -m catpose.realtime --source clip.mp4 --detect --report-dir reports/clip --no-display

# Two cats: learn their coat colours, then follow one cat per run
PYTHONPATH=ml ml/.venv/bin/python -m catpose.identity discover videos/ --out sessions/cats.json
PYTHONPATH=ml ml/.venv/bin/python -m catpose.realtime --source clip.mp4 --detect --cats sessions/cats.json --cat dark --report-dir reports/dark

# Multi-camera test + robot log -> owner report
PYTHONPATH=ml ml/.venv/bin/python -m catpose.session sessions/test1
```

The YOLOv8 detector weights are downloaded by Ultralytics on first use. For the UNO Q app, create `yolo.onnx` with `ml/scripts/export_onnx.py`. The Colab tutorial needs a data pack with dataset sample images, which is not published here.

## Results

Keypoints and detection, on a fixed validation set:

| Metric | v1 | v2 (current) |
| --- | --- | --- |
| Body PCK@0.1 (150 cat instances) | 28.7% | 79.4% |
| Face NME (207 faces) | 3.62% | 1.21% |
| Cat detection recall (132 images) | 85.6% | 93.9% |
| Live pipeline (Apple M5 Pro) | 4–5 FPS | about 30 FPS |

The same validation set was used to choose the best epoch, so these numbers are optimistic; there is no independent test set. The validation images are photographs, not home-camera footage.

On the 16 October 2026 test (37.9 minutes, a silver and a black cat, one report per cat), the explanations of all 62 events (46 and 16) passed the faithfulness check. This shows that each card states the rules' real evidence. It does not show that the emotion guesses are correct.

Full details: [ml/训练报告.md](ml/训练报告.md) (Chinese) and [ml/docs/training.md](ml/docs/training.md).

## Related

- XAI Fieldnotes learning site: https://xai-fieldnotes.vercel.app (source: https://github.com/limbohoo/xai_fieldnote). Its `/case` lab and `/model` 3D walkthrough explain this program step by step.

## Licence

Code and documentation: MIT (see [LICENSE](LICENSE)). The model weights are not MIT: `face.pt` is trained on CatFLW (CC BY-NC 4.0, non-commercial use only), `body.pt` on Animal Pose and PASCAL VOC, and the YOLOv8 detector is AGPL-3.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
