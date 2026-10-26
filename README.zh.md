[English](README.md) | **中文**

# CatPose Robot：会解释自己的猫咪互动机器人

TU/e 课程项目 *AI Interactive Robot for Cats* 的代码仓库。系统从视频里找到猫，标出身体 20 个点和脸部 48 个点，把耳朵、眼睛、活动量和这只猫自己的平静状态比较，用规则和朴素贝叶斯给出行为标签和可能的情绪，再把依据写成主人看得懂的报告。机器人端由研究者在测试中遥控，每个动作带时间戳记下来，报告能显示猫在每个动作之后做了什么。

## 哪些是测过的，哪些是假设

| 部分 | 状态 |
| --- | --- |
| 猫检测与关键点 | 在公开数据集上训练和评估（数字见下） |
| 特征、个体基线、行为规则 | 在真实视频上运行；阈值是手设的 |
| 情绪推测（五种状态） | 先验和似然手设，**没有用人工标注验证** |
| 解释 | 做过忠实度检验：每张卡片写的就是规则实际用到的条件 |
| 机器人动作 | 由研究者触发（Wizard-of-Oz）；情绪结果没有接到机器人上 |

系统**不识别情绪**。它报告猫的脸和身体发生了什么变化，以及这个变化可能意味着什么。

## 仓库内容

| 目录 | 内容 |
| --- | --- |
| `ml/catpose/` | 检测、关键点、特征、规则、贝叶斯推断、忠实度检验、主人报告、实时分析、多机位会话、按毛色区分猫、机器人网桥 |
| `ml/checkpoints/` | 训练好的关键点权重 `body.pt`、`face.pt`（见许可说明） |
| `ml/tests/` | 解释层测试 |
| `ml/docs/` | 英文文档：架构、API、快速开始、训练 |
| `ml/colab/` | Colab 理论教程及其生成脚本 |
| `ml/scripts/` | 数据检查、ONNX 导出、Colab 数据包 |
| `ml/logs/`、`ml/splits/` | 评估结果、速度测试、固定的验证集划分 |
| `ml/*.md` | 中文文档：训练报告、运行说明、参考文献 |
| `hardware/car-bridge/` | 小车接收命令的示例固件与对接说明 |
| `hardware/unoq/` | Arduino UNO Q 板端应用 |
| [`docs/analysis.zh.md`](docs/analysis.zh.md) | Colab 教程、XAI Fieldnotes 网站和主人 App 的分析 |

没有放进仓库的内容：训练数据集（下载方法见 [ml/docs/training.md](ml/docs/training.md)）、实测录像（涉及组员的住处）、数据集图片、第三方视频的画面。

## 快速开始

```bash
python3 -m venv ml/.venv
ml/.venv/bin/pip install -r requirements.txt

# 解释层测试
cd ml && PYTHONPATH=. .venv/bin/python tests/test_explanations.py && cd ..

# 摄像头实时分析，结束时生成主人报告
PYTHONPATH=ml ml/.venv/bin/python -m catpose.realtime --source 0 --detect --report-dir sessions/live1

# 分析一段视频
PYTHONPATH=ml ml/.venv/bin/python -m catpose.realtime --source clip.mp4 --detect --report-dir reports/clip --no-display

# 两只猫：先学毛色，再每次只跟一只
PYTHONPATH=ml ml/.venv/bin/python -m catpose.identity discover videos/ --out sessions/cats.json
PYTHONPATH=ml ml/.venv/bin/python -m catpose.realtime --source clip.mp4 --detect --cats sessions/cats.json --cat dark --report-dir reports/dark

# 多机位测试 + 小车日志 → 主人报告
PYTHONPATH=ml ml/.venv/bin/python -m catpose.session sessions/test1
```

YOLOv8 检测器权重第一次运行时由 Ultralytics 自动下载。UNO Q 应用需要的 `yolo.onnx` 用 `ml/scripts/export_onnx.py` 生成。Colab 教程需要一个含数据集样本图片的数据包，没有在这里公开。

## 结果

关键点和检测，固定验证集：

| 指标 | v1 | v2（当前） |
| --- | --- | --- |
| 身体 PCK@0.1（150 个猫实例） | 28.7% | 79.4% |
| 脸 NME（207 张） | 3.62% | 1.21% |
| 猫检测召回率（132 张） | 85.6% | 93.9% |
| 实时流水线（Apple M5 Pro） | 4–5 FPS | 约 30 FPS |

最佳轮次也是在这个验证集上挑的，所以数字偏乐观；没有独立测试集。验证图是照片，不是家用摄像头的画面。

2026 年 10 月 16 日的测试（37.9 分钟，一只银猫和一只黑猫，每只猫一份报告）中，62 个事件（46 个和 16 个）的解释全部通过忠实度检验。这说明卡片写的就是规则的真实依据，不说明情绪推测是对的。

详见 [ml/训练报告.md](ml/训练报告.md) 和 [ml/docs/training.md](ml/docs/training.md)。

## 相关链接

- XAI Fieldnotes 学习网站：https://xai-fieldnotes.vercel.app （源码 https://github.com/limbohoo/xai_fieldnote ），其中 `/case` 案例实验室和 `/model` 3D 图解一步步讲解本仓库的程序。

## 许可

代码和文档使用 MIT 许可（见 [LICENSE](LICENSE)）。模型权重不属于 MIT：`face.pt` 用 CatFLW 训练（CC BY-NC 4.0，仅限非商业用途），`body.pt` 用 Animal Pose 和 PASCAL VOC 训练，YOLOv8 检测器是 AGPL-3.0。详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
