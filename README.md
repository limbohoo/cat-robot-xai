**English** | [中文](README.zh.md)

# Reading a cat's face and body, and saying why

From ordinary home video, this project marks 48 points on a cat's face and 20 on its body, compares its ears, eyes and movement with **this cat's own calm state**, and turns the change into a behaviour label and a guess at how the cat might feel. Every guess comes with the evidence behind it, a confidence level, and the other likely reading.

It started as the TU/e course project *AI Interactive Robot for Cats*. This repository holds the recognition and explanation part; the robot was driven by hand during tests and is not connected to it.

![Pipeline](docs/img/pipeline.png)

## How it works

**1. Points, not pixels.** A detector finds the cat; two small networks place 48 face landmarks and 20 body keypoints, each with a confidence. Points below 0.25 confidence are dropped, so an unclear frame gives "cannot see" instead of a made-up number.

![Face and body keypoints](docs/img/keypoints.png)

**2. Compared with the cat itself.** Six features come from the points: ear-tip height, ear angle, eye aperture, muzzle ratio, head drop and activity. A Scottish Fold's ears always sit low, so each feature is measured against this cat's calm baseline as a z-score: how many of its usual wobbles away from its usual value.

**3. Readable rules give the behaviour.** "Ears back", "alert", "freeze" and "active" come from a few if-then rules on those z-scores. The rule records exactly which features crossed which threshold, and that record is what the owner sees.

![Evidence for one label](docs/img/evidence.png)

**4. What happens next decides the feeling.** Flattened ears can mean fear or focus. The system starts from a prior set by the behaviour, then watches the next 5 seconds (moves away, flees, approaches, pounces, settles, stays still) and updates five possible states with naive Bayes. Low confidence means no feeling is shown, only the behaviour.

![Bayesian update](docs/img/bayes.png)

**5. The explanation is checked.** For every event, the rules are re-run with the cited features reset to "usual" (the label must change) and with the uncited ones reset (the label must not change). On a 37.9-minute test with a silver and a black cat, all 62 events passed. Two cats are told apart by coat colour.

## Results

![Perception accuracy](docs/img/perception.png)

| Metric | v1 | v2 (current) |
| --- | --- | --- |
| Body PCK@0.1 (150 cat instances) | 28.7% | 79.4% |
| Face NME (207 faces, lower is better) | 3.62% | 1.21% |
| Cat detection recall (132 images) | 85.6% | 93.9% |
| Live pipeline (Apple M5 Pro) | 4–5 FPS | about 30 FPS |

## What is measured and what is assumed

| Part | Status |
| --- | --- |
| Detection and keypoints | Trained and evaluated on public datasets |
| Features, baseline, behaviour rules | Run on real video; thresholds are hand-set |
| Feeling guess (five states) | Hand-set probabilities; **not validated against human labels** |
| Explanations | Faithfulness-checked: each card states the conditions the rules actually used |

The system does not recognise emotions. It reports what changed in the cat's face and body, and what that change might mean. The validation numbers come from the same set used to choose the model, and the images are photographs, not home-camera footage.

## Quick start

```bash
python3 -m venv ml/.venv
ml/.venv/bin/pip install -r requirements.txt

# Explanation-layer tests
cd ml && PYTHONPATH=. .venv/bin/python tests/test_explanations.py && cd ..

# Analyse a video; writes an owner report with clips and evidence
PYTHONPATH=ml ml/.venv/bin/python -m catpose.realtime --source clip.mp4 --detect --report-dir reports/clip --no-display

# Live camera
PYTHONPATH=ml ml/.venv/bin/python -m catpose.realtime --source 0 --detect --report-dir sessions/live1

# Two cats: learn their coat colours, then follow one cat per run
PYTHONPATH=ml ml/.venv/bin/python -m catpose.identity discover videos/ --out sessions/cats.json
PYTHONPATH=ml ml/.venv/bin/python -m catpose.realtime --source clip.mp4 --detect --cats sessions/cats.json --cat dark --report-dir reports/dark
```

The YOLOv8 detector weights are downloaded by Ultralytics on first use.

## Repository layout

| Folder | Contents |
| --- | --- |
| `ml/catpose/` | Detection, keypoints, features, rules, Bayesian inference, faithfulness checks, owner report, live and multi-camera analysis, cat identity |
| `ml/checkpoints/` | Trained keypoint weights `body.pt` and `face.pt` |
| `ml/tests/` | Explanation-layer tests |
| `ml/docs/` | Architecture, API, quickstart, training |
| `ml/colab/` | Colab theory tutorial (needs a separate data pack with dataset images) |
| `ml/*.md` | Chinese docs: training report, run guide, references |
| `hardware/` | Robot bridge firmware and an Arduino UNO Q edge app, kept for reference; not connected to the recognition |
| [`docs/analysis.md`](docs/analysis.md) | The Colab tutorial, the learning site and the owner app compared |

The figures above are drawn from keypoints and event numbers of a 36-second web video; no video frames are included. Training datasets, test recordings and dataset images are not in this repository.

## See it step by step

The [case lab](https://xai-fieldnotes.vercel.app/case) walks one cat video through every step, and the [model walkthrough](https://xai-fieldnotes.vercel.app/model) shows the whole model in 3D.

## Licence

Code and documentation: MIT ([LICENSE](LICENSE)). The weights are not MIT: `face.pt` is trained on CatFLW (CC BY-NC 4.0, non-commercial only), `body.pt` on Animal Pose and PASCAL VOC, and the YOLOv8 detector is AGPL-3.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
