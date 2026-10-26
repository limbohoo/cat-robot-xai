"""Generate CatPose_XAI_Tutorial.ipynb (the Colab tutorial) from the cells below.

    python3 ml/colab/build_notebook.py

Edit the cells here, not the .ipynb, so the notebook stays reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


# =============================================================================
# 0. Front matter
# =============================================================================
md(r"""
# CatPose-XAI 理论教程：从像素到可解释的猫咪状态与互动

**项目**：室内独处猫咪的具身陪伴机器人（xAI embodied intelligent system, group 01）
**这本笔记本做什么**：把我们系统里用到的每一个 AI 组件，按照 **数学模型 → 算法逻辑 → 可运行的演算 → 验证案例 → 在系统中的位置** 逐步拆开，并如实披露每一步已经验证到什么程度、还缺什么。

### 如何使用
1. 在 Colab 里选择 **运行时 → 全部运行**。不需要 GPU，CPU 运行时即可，全程约 3–5 分钟。
2. 数据包 `catpose_colab_pack.zip`（约 41 MB）放在 Google Drive 的 `MyDrive/CatPose-XAI/` 下。第 0 节会自动挂载 Drive 并解压；找不到时会弹出上传对话框。
3. 图表里的文字用英文（Colab 默认没有中文字体），讲解用中文。

### 证据等级：每一节标题后都会标注
| 标记 | 含义 |
|---|---|
| ✅ **已实现并验证** | 代码已部署，并在固定验证集上测过指标 |
| 🧪 **真实数据原型** | 已经在真实视频上运行，能发现问题，但结果还没有用人工标签验证过 |
| 🧪 **合成数据 / 模拟** | 算法可以运行，但数据是合成的，或者效果只在模拟环境里验证过 |
| 📐 **设计假设** | 有文献支持的方向，但还没有在我们的猫和我们的系统上验证 |

> **为什么要标注？** 可解释性（XAI）本身也需要被解释。一个系统如果把 🧪 和 📐 的东西讲成 ✅，那它的“解释”只会制造虚假的信任。本教程希望让读者清楚知道，每个结论可以信到什么程度。

### 目录
0. 环境准备与系统全景
1. 目标检测：猫在哪里？（YOLO）✅
2. 关键点估计：猫的姿态是什么？（热力图 + soft-argmax）✅
3. 时序平滑：让关键点不抖（One-Euro 滤波）✅
4. 从关键点到可解释特征（Feline Grimace Scale 的几何化、个体基线、头部姿态门控）🧪 真实数据原型
5. 从特征到状态：**先描述反应，再推断情绪**
   - 第一层：行为反应分类（规则，已在真实视频上运行）🧪 真实数据原型
   - 第二层：情绪推断（规则 → 决策树 → 梯度提升 → 校准 → 时序）🧪 合成数据
6. 可解释性（XAI）：三层解释、**给主人的“变化报告”**、如何评估解释本身，以及一个现在就能做的用户研究 🧪
7. 机器人交互的理论依据与形式化 📐
8. 强化学习：让机器人学会这只猫喜欢怎么玩（模拟）🧪
9. 数据、标签与部署：标签从哪里来、如何形成闭环、如何部署到移动机器人 📐
10. 版本演进的完整披露与局限：现在能说什么，实地测试后能说什么
11. 参考文献

> **本教程的主线**：识别（✅）→ 可解释的特征（🧪）→ **可观察的行为反应和状态变化**（🧪，不需要猜测情绪）→ 情绪推断（需要真实标签才能成立）。越往后，需要的证据越多，我们能说的就越少。给主人的输出围绕**变化**来组织：猫身上变了什么（看得见的依据），变化之前发生了什么（可能的触发）。
""")

md(r"""
## 0. 环境准备与系统全景

### 0.1 系统全景

$$
\underbrace{I_t}_{\text{camera frame}}
\xrightarrow{\text{YOLO}} b_t
\xrightarrow{\text{keypoint nets}} (K_t, c_t)
\xrightarrow{\text{One-Euro}} \tilde K_t
\xrightarrow{\phi(\cdot)} f_t
\xrightarrow{g_\theta} p(s_t \mid f_{1:t})
\longrightarrow
\begin{cases}
a_t = \pi(p(s_t), \dots) & \text{robot action (edge)}\\
E_t = \text{explain}(g_\theta, f_t) & \text{owner app}
\end{cases}
$$

| 符号 | 含义 | 维度 |
|---|---|---|
| $I_t$ | 第 $t$ 帧图像 | $H\times W\times 3$ |
| $b_t$ | 猫的包围框 $(x_0,y_0,x_1,y_1)$ 及置信度 | $4+1$ |
| $K_t$ | 关键点：身体 20 个 + 脸部 48 个 | $68\times 2$ |
| $c_t$ | 每个关键点的置信度 | $68$ |
| $f_t$ | 几何特征（耳朵、眼睛、头部、身体、活动量） | $d\approx 6$ |
| $s_t$ | 猫的状态：relaxed / alert / playful / anxious | 4 类 |
| $a_t$ | 机器人动作 | 离散 |
| $E_t$ | 解释：图像高亮 + 特征贡献 + 文字 + 建议 | — |

**设计上最关键的一个决定**：我们**不**训练“图像 → 情绪”的端到端网络，而是在中间插入人能看懂的**关键点和几何特征**。原因有两个：
1. **数据**：带情绪标签的猫图像几乎没有公开数据，但关键点有（CatFLW、Animal Pose）。
2. **可解释性**：每个判断都能追溯到“耳朵低了多少、眼睛眯了多少”，而不是一张说不清的热力图。这种设计在 XAI 文献里叫 *interpretable-by-design* 或 *concept bottleneck*（先预测人能理解的概念，再由概念做决策）。

下面先准备运行环境。
""")

md(r"""
### 本版更新（v4，2026-10-13）

| 位置 | 更新 | 证据等级 |
|---|---|---|
| 5.0 | 标签的**进入 / 退出阈值**（迟滞），数值在阈值附近徘徊时不再来回跳 | 🧪 |
| **5.10（新）** | 实际部署的第二层：**用随后的行为更新情绪推测**（朴素贝叶斯），把握分高 / 中 / 低，低就不显示；**看不清脸时最多给“中”** | 🧪 概率是手设假设 |
| **5.11（新）** | 随后行为线索的定义：**区分逃跑和扑玩**，**蹲低不算离开**，小车上的摄像头不判断远近 | 🧪 |
| **5.12（新）** | **从主人的反馈学习**：Dirichlet / Beta 伪计数，几条标注只“推动”、不推翻原来的假设 | 🧪 模拟 |
| 6.11 | 事件合并、声音措辞、主人 app 与反馈闭环 | — |
| 10.1–10.2 | 本轮修复及改前改后的实测对比；**结果对校准长度很敏感** | 测试视频 |
| **10.5（新）** | **对本项目的四条批评与回应**：证据薄、评估只覆盖技术层、模型类型窄、受众窄 | — |
| **6.14（新，v4.1）** | **部署系统的解释是不是真正的依据**：直接导入程序本身的规则文件，在测试视频的 7 个真实事件上做反事实检验 | ✅ 真实视频，7/7 通过 |

第 5.0、5.10、5.11 节的代码与 `catpose/reactions.py`、`interpret.py`、`events.py` 的逻辑一致，但在这里用 numpy 单独写出，方便逐行阅读。
""")

code(r"""
# 0.2 安装依赖、找到数据包
import os, sys, json, math, time, zipfile, itertools, subprocess
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'onnxruntime'], check=False)

import numpy as np
import cv2
import matplotlib.pyplot as plt
import onnxruntime as ort

np.set_printoptions(precision=3, suppress=True)
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': False})
RNG = np.random.default_rng(0)


def find_pack():
    if IN_COLAB and not Path('/content/drive').exists():
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except Exception as e:  # 没有授权 Drive 也可以继续：下面会让你手动上传
            print('Drive 未挂载：', e)
    root = Path('/content') if IN_COLAB else Path.cwd()
    candidates = [
        os.environ.get('CATPOSE_PACK', ''),
        str(root / 'catpose_colab_pack'),
        '/content/drive/MyDrive/CatPose-XAI/catpose_colab_pack.zip',
        '/content/drive/MyDrive/catpose_colab_pack.zip',
        str(root / 'catpose_colab_pack.zip'),
        '../../output/colab/catpose_colab_pack',
    ]
    for c in candidates:
        p = Path(c) if c else None
        if p is None or not p.exists():
            continue
        if p.is_dir() and (p / 'models').exists():
            return p
        if p.suffix == '.zip':
            with zipfile.ZipFile(p) as z:
                z.extractall(root)
            return root / 'catpose_colab_pack'
    if IN_COLAB:
        print('没有找到数据包，请上传 catpose_colab_pack.zip')
        from google.colab import files
        for name in files.upload():
            with zipfile.ZipFile(name) as z:
                z.extractall(root)
        return root / 'catpose_colab_pack'
    return None


PACK = find_pack()
assert PACK is not None and (PACK / 'models' / 'body.onnx').exists(), '缺少数据包，后面的真实模型演算无法运行'
META = json.loads((PACK / 'models' / 'meta.json').read_text())
SAMPLES = json.loads((PACK / 'samples' / 'samples.json').read_text())
print('数据包位置:', PACK)
print('模型:', sorted(p.name for p in (PACK / 'models').iterdir()))
print('验证样本:', sum(s['kind'] == 'body' for s in SAMPLES), '张身体图,', sum(s['kind'] == 'face' for s in SAMPLES), '张脸部图')


def load_rgb(name):
    return cv2.cvtColor(cv2.imread(str(PACK / 'samples' / name)), cv2.COLOR_BGR2RGB)
""")

# =============================================================================
# 1. Detection
# =============================================================================
md(r"""
## 1. 目标检测：猫在哪里？ ✅

### 1.1 数学模型

我们使用 **YOLOv8**（Jocher et al., 2023），它在 COCO 数据集上预训练，第 15 类就是 cat。我们没有重新训练检测器。

**网格预测。** 输入图像先 *letterbox* 缩放成 $S\times S$（机器人上 $S=320$）：按比例缩放 $s = S/\max(h,w)$，空白处填灰色，因此猫不会被拉伸变形。网络在步长 8、16、32 三个尺度的网格上，每个格子都预测一个候选框：

$$N = (S/8)^2 + (S/16)^2 + (S/32)^2 = 40^2+20^2+10^2 = 2100 \text{ 个候选}$$

每个候选输出 $(c_x, c_y, w, h)$ 和 80 个类别分数 $\sigma(z_k)\in[0,1]$。YOLOv8 是 *anchor-free* 的：框的四条边到格子中心的距离 $(l,t,r,b)$ 被当作一个离散分布来预测（Distribution Focal Loss），取期望值作为距离。这和第 2 节 soft-argmax 的思想是一样的：**用分布的期望代替硬性的取最大值**。

**IoU（交并比）**，衡量两个框的重叠程度：
$$\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}$$

**非极大值抑制（NMS）**：按分数从高到低排序，保留当前最高分的框，删掉和它 IoU 超过阈值的其他框，然后重复。

**我们系统里的简化。** 机器人上只取 cat 类分数最高的**一个**框（`argmax`），因此不需要 NMS。这隐含了“画面里只有一只猫”的假设。多猫家庭需要改回 NMS 并加上多目标跟踪，这一点列入局限。
""")

code(r"""
# 1.2 演算：IoU 与 NMS（用手算得出的数字核对）
def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def nms(boxes, scores, thr=0.5):
    order, keep = list(np.argsort(scores)[::-1]), []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) < thr]
    return keep


boxes = np.array([[10, 10, 110, 110], [20, 20, 120, 120], [200, 50, 260, 120]], float)
scores = np.array([0.90, 0.75, 0.60])
print('IoU(box0, box1) =', round(iou(boxes[0], boxes[1]), 3), '  手算: 90*90 / (2*100*100 - 90*90) = 0.681')
print('NMS 保留的框:', [int(i) for i in nms(boxes, scores)], ' → box1 与 box0 重叠过多，被抑制')
""")

code(r"""
# 1.3 真实模型：机器人上运行的同一个 YOLO ONNX 文件
COCO_CAT = 15


def session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    return ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider'])


class CatDetector:
    # 与 hardware/unoq/catpose-edge/python/catpose_onnx.py 完全一致：输入 BGR，内部转成 RGB
    def __init__(self, imgsz=META['yolo']['imgsz']):
        self.sess, self.imgsz = session(PACK / 'models' / 'yolo.onnx'), imgsz

    def raw(self, bgr):
        h, w = bgr.shape[:2]
        s = self.imgsz / max(h, w)
        nh, nw = round(h * s), round(w * s)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)  # letterbox
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh))
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0][0], s

    def best_cat(self, bgr):
        out, s = self.raw(bgr)
        i = int(np.argmax(out[4 + COCO_CAT]))
        cx, cy, bw, bh = out[:4, i] / s
        return float(out[4 + COCO_CAT, i]), (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


det = CatDetector()
out, _ = det.raw(cv2.imread(str(PACK / 'samples' / 'body_00.jpg')))
print('网络原始输出形状:', out.shape, '= 4 个框坐标 + 80 个类别分数，共', out.shape[1], '个候选（上面推导的 N）')

body_samples = [s for s in SAMPLES if s['kind'] == 'body']
fig, axes = plt.subplots(2, 6, figsize=(16, 5.6))
hits = []
for ax, s in zip(axes.flat, body_samples):
    bgr = cv2.imread(str(PACK / 'samples' / s['file']))
    conf, box = det.best_cat(bgr)
    ok = iou(box, s['bbox']) > 0.5 and conf >= 0.25  # 与机器人端一致：低于 0.25 的检测会被丢弃
    hits.append(ok)
    if not ok:
        why = f'置信度 {conf:.2f} 低于 0.25，机器人会认为“没有猫”' if conf < 0.25 else f'框的位置偏差大（IoU {iou(box, s["bbox"]):.2f}），或者框到了画面里的另一只猫'
        print(f'{s["file"]}: 未命中，{why}')
    ax.imshow(bgr[:, :, ::-1])
    x0, y0, x1, y1 = s['bbox']
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='white', ls='--', lw=1.5))
    ax.add_patch(plt.Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False, ec='lime' if ok else 'red', lw=2))
    ax.set_title(f'cat {conf:.2f}  IoU {iou(box, s["bbox"]):.2f}', fontsize=9)
    ax.axis('off')
plt.suptitle('White dashed = ground truth, green/red = detector (hit = IoU > 0.5 and confidence >= 0.25)')
plt.tight_layout(); plt.show()
print(f'这 {len(hits)} 张验证图上的召回率: {np.mean(hits):.0%}')
print('对照：完整验证集 132 张、输入 640 时，YOLOv8n 为 90.2%，YOLOv8s 为 93.9%。')
print('机器人上为了速度用的是 YOLOv8n、输入 320，这个配置还没有在完整验证集上单独评估过（见第 10 节）；小目标和多猫画面会受影响。')
""")

md(r"""
### 1.4 验证案例：一个真实的 bug 是怎么被发现的

v1 版本在摄像头里经常“找不到猫”。建立正式评估之后才定位到原因：检测库默认把 numpy 输入当作 **BGR** 通道顺序，而我们传入的是 **RGB**。红蓝两个通道互换之后，猫的颜色分布变了，置信度大幅下降。一张测试帧上，猫的置信度从 0.85 掉到 0.14，被判定成了“沙发”。

下面在验证图上直接复现这个 bug。影响因图而异：有的图几乎不变，有的图会跌破阈值、直接“丢猫”。在完整验证集上，修复这个 bug 让召回率从 85.6% 升到 90.2%。这也说明了一件事：**没有定量评估，就没有可信的解释**。如果当时我们对这个检测器做“可解释性分析”，解释的其实是一个输入本身就错了的模型。
""")

code(r"""
# 1.5 复现 BGR/RGB bug：同一张图，只交换红蓝通道
right, wrong = [], []
for s in body_samples:
    bgr = cv2.imread(str(PACK / 'samples' / s['file']))
    right.append(det.best_cat(bgr)[0])
    wrong.append(det.best_cat(bgr[:, :, ::-1].copy())[0])
right, wrong = np.array(right), np.array(wrong)
print('正确通道顺序  平均 cat 置信度:', right.mean().round(3))
print('红蓝通道互换  平均 cat 置信度:', wrong.mean().round(3))
print('低于阈值 0.25 的图片数：正确', int((right < 0.25).sum()), '张；互换后', int((wrong < 0.25).sum()), '张')
plt.figure(figsize=(6, 3))
plt.bar(np.arange(len(right)) - 0.2, right, 0.4, label='correct (BGR)')
plt.bar(np.arange(len(wrong)) + 0.2, wrong, 0.4, label='swapped (RGB bug)')
plt.axhline(0.25, color='k', ls=':', lw=1); plt.xlabel('validation image'); plt.ylabel('cat confidence'); plt.legend(); plt.tight_layout(); plt.show()
""")

# =============================================================================
# 2. Keypoints
# =============================================================================
md(r"""
## 2. 关键点估计：猫的姿态是什么？ ✅

### 2.1 问题定义

给定猫的裁剪图 $x\in\mathbb R^{256\times256\times3}$，预测 $K$ 个关键点的坐标 $\hat u_k\in[0,1]^2$（身体 $K=20$，脸部 $K=48$）。训练数据：

| 模型 | 数据集 | 规模 | 关键点 |
|---|---|---|---|
| 身体 | Animal Pose（Cao et al., 2019）+ PASCAL VOC 图片 | 5963 个实例（猫 1316，另有狗、羊、马、牛） | 眼、鼻、耳、肘、膝、爪、喉、肩胛、尾根 |
| 脸部 | CatFLW（Martvel et al., 2023） | 2079 张（1872 训练 / 207 验证） | 48 个点，方案来自 Finka et al. (2019)，与猫的面部肌肉对应 |

### 2.2 v1 的做法和它为什么失败：全局回归

$$\hat u = \sigma\big(W\cdot \mathrm{GAP}(F(x)) + b\big),\qquad \mathrm{GAP}(F)_c = \frac{1}{HW}\sum_{i,j}F_{c,i,j}$$

$F(x)$ 是 MobileNetV3 输出的 $7\times7$ 特征图。**全局平均池化（GAP）把每个通道的所有位置求平均**，于是“特征出现在图的哪个位置”这一信息在这里被丢掉了，后面的 MLP 只能靠死记硬背去恢复坐标。下面用一个最小的例子证明这一点。
""")

code(r"""
# 2.3 演算：GAP 对位置“失明”
blob = lambda r, c: np.pad(np.ones((2, 2)), ((r, 6 - r), (c, 6 - c)))  # 8x8 特征图上一个 2x2 的激活
A, B = blob(1, 1), blob(5, 4)
print('猫耳在左上 → GAP =', A.mean(), ';  猫耳在右下 → GAP =', B.mean(), ' → 完全相同，位置信息丢失')
fig, ax = plt.subplots(1, 2, figsize=(5, 2.4))
for a, m, t in zip(ax, (A, B), ('ear top-left', 'ear bottom-right')):
    a.imshow(m, cmap='viridis'); a.set_title(f'{t}\nGAP={m.mean():.4f}', fontsize=9); a.axis('off')
plt.tight_layout(); plt.show()
""")

md(r"""
### 2.4 v2 的做法：热力图 + soft-argmax（integral regression）

**网络结构。** MobileNetV3-Large 骨干（Howard et al., 2019）在步长 8、16、32 上输出特征 $C_8, C_{16}, C_{32}$，用一个轻量的特征金字塔（FPN，Lin et al., 2017）融合：
$$P = L_8(C_8) + \mathrm{Up}\big(L_{16}(C_{16}) + \mathrm{Up}(L_{32}(C_{32}))\big)$$
其中 $L$ 是 $1\times1$ 卷积，Up 是最近邻上采样。$P$ 保留了 $32\times32$ 的**空间网格**，再为每个关键点 $k$ 输出一张 logit 图 $Z_k\in\mathbb R^{32\times32}$。

**从热力图到坐标。** 先对整张图做 softmax，得到概率分布：
$$P_k(i,j)=\frac{e^{Z_k(i,j)}}{\sum_{i',j'}e^{Z_k(i',j')}}$$
坐标取这个分布的**期望**（soft-argmax，Sun et al., 2018）：
$$\hat u_k = \sum_{i,j} P_k(i,j)\,\Big(\tfrac{j+0.5}{W},\ \tfrac{i+0.5}{H}\Big)$$
和 argmax 相比有两个好处：(1) 可以求导，因此能端到端训练；(2) 精度可以细到格子内部（sub-cell）。

**置信度。** 取峰值附近 $3\times3$ 窗口里的概率质量：$c_k=\max_{i,j}\sum_{|\delta|\le1}P_k(i+\delta_i,j+\delta_j)$。分布越集中，$c_k$ 越接近 1。

**损失函数**（DSNT 风格，Nibali et al., 2018），$v_k\in\{0,1\}$ 表示该点是否被标注（被遮挡的点不参与）：
$$\mathcal L=\frac{1}{\sum_k v_k}\sum_k v_k\Big(\underbrace{\|(\hat u_k-u_k)\odot(W,H)\|_1}_{\text{坐标误差（格子单位）}}+\underbrace{\mathrm{KL}\big(G_k\,\|\,P_k\big)}_{\text{让分布集中成高斯形}}\Big)$$
$G_k$ 是以真值为中心、$\sigma=1.5$ 格的归一化高斯。只用坐标损失的话，分布可能很分散，但期望值恰好对；KL 项迫使分布本身也集中在正确位置，这样置信度才有意义。
""")

code(r"""
# 2.5 演算：soft-argmax 的优点，以及它的失败模式
H = W = 32
yy, xx = np.mgrid[0:H, 0:W] + 0.5


def softmax2d(z):
    e = np.exp(z - z.max()); return e / e.sum()


def soft_argmax(p):
    return (p * xx).sum(), (p * yy).sum()


def conf3x3(p):
    return max(p[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2].sum() for i in range(H) for j in range(W))


true = (10.3, 20.7)  # 真值位于格子内部
Z1 = -((xx - true[0]) ** 2 + (yy - true[1]) ** 2) / (2 * 1.5 ** 2) * 4 + RNG.normal(0, 0.3, (H, W))
P1 = softmax2d(Z1)
am = np.unravel_index(P1.argmax(), P1.shape)
u1, v1 = soft_argmax(P1)
print(f'真值 {true}   argmax ({am[1] + 0.5:.1f}, {am[0] + 0.5:.1f})   soft-argmax ({u1:.2f}, {v1:.2f})   置信度 {conf3x3(P1):.2f}')

# 失败模式：两只耳朵长得很像，分布出现两个峰 → 期望落在两峰之间，那里什么都没有
Z2 = np.logaddexp(-((xx - 6) ** 2 + (yy - 8) ** 2) / 4.5 * 4, -((xx - 25) ** 2 + (yy - 8) ** 2) / 4.5 * 4)
P2 = softmax2d(Z2)
u2, v2 = soft_argmax(P2)
print(f'双峰情形：soft-argmax ({u2:.1f}, {v2:.1f}) 落在两峰 (6, 8) 和 (25, 8) 之间，那里什么都没有；置信度从 {conf3x3(P1):.2f} 降到 {conf3x3(P2):.2f}')
print('注意：0.43 仍高于机器人端 0.25 的显示门限，所以这种错误点会被画出来。门限是经验值（见第 10 节局限），这个例子说明它值得系统调参。')

fig, ax = plt.subplots(1, 2, figsize=(7, 3.2))
for a, p, t in zip(ax, (P1, P2), ('unimodal: accurate + confident', 'bimodal: mean is wrong, conf low')):
    a.imshow(p, cmap='magma'); u, v = soft_argmax(p); a.plot(u - 0.5, v - 0.5, 'c+', ms=14, mew=2); a.set_title(t, fontsize=9); a.axis('off')
plt.tight_layout(); plt.show()


def pose_loss_np(P, u_hat, u_true, sigma=1.5):
    G = np.exp(-((xx - u_true[0]) ** 2 + (yy - u_true[1]) ** 2) / (2 * sigma ** 2)); G /= G.sum()
    l1 = abs(u_hat[0] - u_true[0]) + abs(u_hat[1] - u_true[1])
    kl = (G * (np.log(G + 1e-10) - np.log(P + 1e-10))).sum()
    return l1, kl


print('单峰的损失 (L1, KL):', np.round(pose_loss_np(P1, soft_argmax(P1), true), 3))
print('双峰的损失 (L1, KL):', np.round(pose_loss_np(P2, soft_argmax(P2), (6, 8)), 3), '→ KL 项明确惩罚了“分散”')
""")

md(r"""
### 2.6 评估指标

- **PCK@α**（Percentage of Correct Keypoints）：误差小于 $\alpha\cdot\max(w,h)$ 的点所占比例（$w,h$ 是包围框尺寸），越高越好。$\alpha=0.1$ 是常用标准，$0.05$ 更严格。
$$\mathrm{PCK}@\alpha=\frac{1}{\sum v}\sum_k v_k\,\mathbb 1\big[\|\hat u_k-u_k\|<\alpha\max(w,h)\big]$$
- **NME**（Normalized Mean Error）：$\frac{1}{\sum v}\sum_k v_k\|\hat u_k-u_k\|/\sqrt{wh}$，越低越好。

下面用机器人上运行的同一个 ONNX 模型，在数据包里 24 张**训练时没见过**的验证图上重新计算这两个指标。
""")

code(r"""
# 2.7 真实模型：身体 20 点 + 脸部 48 点
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
BODY_NAMES = ['left_eye', 'right_eye', 'nose', 'left_ear', 'right_ear', 'left_front_elbow', 'right_front_elbow',
              'left_back_elbow', 'right_back_elbow', 'left_front_knee', 'right_front_knee', 'left_back_knee',
              'right_back_knee', 'left_front_paw', 'right_front_paw', 'left_back_paw', 'right_back_paw',
              'throat', 'withers', 'tailbase']
BODY_SKELETON = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 4), (2, 17), (18, 19), (5, 9), (6, 10), (7, 11), (8, 12),
                 (9, 13), (10, 14), (11, 15), (12, 16)]


class PosePredictor:
    # 与机器人端 catpose_onnx.PosePredictor 一致：仿射裁剪 → 归一化 → ONNX → 映射回原图
    def __init__(self, kind, tta=True):
        m = META[kind]
        self.sess, self.img, self.margin = session(PACK / 'models' / f'{kind}.onnx'), m['img'], m['margin']
        self.flip_map, self.tta = m['flip_map'], tta

    def crop_matrix(self, bbox):
        x0, y0, x1, y1 = bbox
        center = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32)
        side = max(x1 - x0, y1 - y0, 8.0) * (1 + self.margin)
        s = self.img / side
        return np.array([[s, 0, self.img / 2 - s * center[0]], [0, s, self.img / 2 - s * center[1]]], np.float32)

    def run_patch(self, patch):
        views = [patch] + ([np.ascontiguousarray(patch[:, ::-1])] if self.tta else [])
        outs = [self.sess.run(None, {'image': ((v.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]})
                for v in views]  # 导出的 ONNX 固定 batch=1
        uv, conf = np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs])
        pts, c = uv[0] * self.img, conf[0]
        if self.tta:  # 翻转测试时增强：镜像预测按左右映射换回，再取平均
            pf = uv[1] * self.img
            pf[:, 0] = self.img - 1 - pf[:, 0]
            pts, c = (pts + pf[self.flip_map]) / 2, (c + conf[1][self.flip_map]) / 2
        return pts, c

    def predict(self, rgb, bbox):
        m = self.crop_matrix(bbox)
        patch = cv2.warpAffine(rgb, m, (self.img, self.img), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        pts, c = self.run_patch(patch)
        inv = cv2.invertAffineTransform(m)
        return pts @ inv[:, :2].T + inv[:, 2], c


def pck_nme(pred, s):
    gt, vis = np.array(s['pts']), np.array(s['vis']) > 0
    x0, y0, x1, y1 = s['bbox']
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    err = np.linalg.norm(pred - gt, axis=1)[vis]
    return err, max(w, h), np.sqrt(w * h)


body_net, face_net = PosePredictor('body'), PosePredictor('face')
report = {}
for kind, net in (('body', body_net), ('face', face_net)):
    errs = []
    for s in [s for s in SAMPLES if s['kind'] == kind]:
        pred, _ = net.predict(load_rgb(s['file']), s['bbox'])
        e, side, norm = pck_nme(pred, s)
        errs.append((e, side, norm))
    e_all = np.concatenate([e for e, _, _ in errs])
    pck05 = np.concatenate([e < 0.05 * sd for e, sd, _ in errs]).mean()
    pck10 = np.concatenate([e < 0.10 * sd for e, sd, _ in errs]).mean()
    nme = np.concatenate([e / n for e, _, n in errs]).mean()
    report[kind] = (pck05, pck10, nme, len(e_all))
    print(f'{kind:4s}: PCK@0.05 {pck05:6.1%}   PCK@0.1 {pck10:6.1%}   NME {nme:6.2%}   （{len(e_all)} 个可见点）')
print('完整验证集上的报告值：身体 PCK@0.1 79.4% / NME 8.3%；脸部 PCK@0.05 99.0% / NME 1.21%')
print('样本只有 12 张，数字会有几个百分点的波动；这是一次独立的抽查，不是替代完整评估。')
""")

code(r"""
# 2.8 可视化：真值（白）vs 预测（彩色），线段 = 误差
def draw_body(ax, rgb, pts, conf, gt=None, min_conf=0.25):
    ax.imshow(rgb)
    keep = conf >= min_conf
    for a, b in BODY_SKELETON:
        if keep[a] and keep[b]:
            ax.plot(*zip(pts[a], pts[b]), c='orange', lw=2)
    ax.scatter(*pts[keep].T, s=14, c='orange', zorder=3)
    if gt is not None:
        g, v = np.array(gt['pts']), np.array(gt['vis']) > 0
        ax.scatter(*g[v].T, s=10, c='white', edgecolors='k', lw=0.5, zorder=4)
        for p, q in zip(pts[v], g[v]):
            ax.plot(*zip(p, q), c='red', lw=0.8)
    ax.axis('off')


fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, s in zip(axes[:2], body_samples[:2]):
    rgb = load_rgb(s['file']); pts, c = body_net.predict(rgb, s['bbox'])
    draw_body(ax, rgb, pts, c, s); ax.set_title('body: ' + s['file'], fontsize=9)
face_samples = [s for s in SAMPLES if s['kind'] == 'face']
for ax, s in zip(axes[2:], face_samples[:2]):
    rgb = load_rgb(s['file']); pts, c = face_net.predict(rgb, s['bbox'])
    ax.imshow(rgb); g = np.array(s['pts'])
    ax.scatter(*g.T, s=8, c='white', edgecolors='k', lw=0.4); ax.scatter(*pts.T, s=8, c='deepskyblue')
    x0, y0, x1, y1 = s['bbox']; ax.set_xlim(x0 - 10, x1 + 10); ax.set_ylim(y1 + 10, y0 - 10)
    ax.set_title('face: ' + s['file'], fontsize=9); ax.axis('off')
plt.tight_layout(); plt.show()
""")

md(r"""
### 2.9 从 v1 到 v2：改进到底来自哪里（消融实验）

以下数字来自完整验证集（身体 150 个猫实例、1649 个可见点；脸部 207 张、9936 个点），详见 `ml/训练报告.md`。

| 改动（身体模型） | PCK@0.1 | 变化 |
|---|---|---|
| v1：全局回归，180 只猫 | 28.7% | — |
| 换成热力图架构 + 新数据增强（数据不变） | 59.0% | **+30.3** |
| 数据从 180 只猫增加到 1316 只 | 78.4% | **+19.4** |
| 先在 5 种动物上预训练，再用猫微调 | 79.4% | +1.0（在误差范围附近） |

| 部位（身体 PCK@0.1） | v1 | v2 |
|---|---|---|
| 头部（眼、鼻、耳） | 37% | **91%** |
| 肘 / 膝 / 爪 | 19–25% | 64–71% |

**解读。** 最大的提升来自架构（保留空间信息），其次是数据量。“多物种预训练”只多了 1 个点，按科学标准**不能**算作确定结论。头部点的提升对下游最关键，因为耳朵和眼睛的特征都靠它们。腿和爪仍是弱项，所以第 4 节的特征尽量少依赖爪子的位置。
""")

code(r"""
# 2.10 训练曲线：数据包里保存的每轮验证指标
hist_dir = PACK / 'history'
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for name, label in (('body_part2only', 'v2 arch, 180 cats'), ('body_catonly', 'v2, 1316 cats'),
                    ('body_all', 'v2, 5 species'), ('body_ft', 'v2, 5 species -> cat fine-tune')):
    f = hist_dir / f'{name}.history.json'
    if f.exists():
        h = json.loads(f.read_text())
        ax[0].plot([r['epoch'] for r in h], [r['pck@0.1'] for r in h], label=label)
ax[0].axhline(28.7, color='gray', ls=':', label='v1 regression'); ax[0].set_xlabel('epoch'); ax[0].set_ylabel('body PCK@0.1 (%)'); ax[0].legend(fontsize=7)
f = hist_dir / 'face_hm.history.json'
if f.exists():
    h = json.loads(f.read_text())
    ax[1].plot([r['epoch'] for r in h], [r['nme%'] for r in h], label='v2 heatmap')
ax[1].axhline(3.62, color='gray', ls=':', label='v1 regression'); ax[1].set_xlabel('epoch'); ax[1].set_ylabel('face NME (%)'); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.show()
""")

# =============================================================================
# 3. One-Euro
# =============================================================================
md(r"""
## 3. 时序平滑：让关键点不抖 ✅

逐帧独立预测的关键点会抖动。简单的指数平均 $\hat x_t=\hat x_{t-1}+\alpha(x_t-\hat x_{t-1})$ 面临一个两难：$\alpha$ 小，猫静止时不抖，但快速运动时点会“拖在后面”；$\alpha$ 大则反过来。

**One-Euro 滤波**（Casiez et al., 2012）让截止频率随速度自适应：
$$\alpha(f_c,\Delta t)=\frac{1}{1+\tau/\Delta t},\quad \tau=\frac{1}{2\pi f_c},\qquad f_c = f_{\min}+\beta\,|\dot{\hat x}_t|$$
静止时 $|\dot x|\approx0$，$f_c=f_{\min}$ 很低，平滑很强；运动时 $f_c$ 升高，跟得更紧。速度 $\dot x$ 本身也用一个固定截止频率 $d_{c}$ 平滑。我们把速度除以猫的包围框尺寸，所以远近不同的猫用同一组参数。

对下游的意义：**第 4 节的活动量特征就是关键点速度**。不平滑的话，抖动会被误认为“猫在动”。
""")

code(r"""
# 3.1 演算：静止 → 快速移动 → 静止，比较三种滤波
class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    def __call__(self, x, t, scale=1.0):
        x = np.asarray(x, float)
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3); self.t = t
        a_d = 1 / (1 + (1 / (2 * np.pi * self.d_cutoff)) / dt)
        self.dx = self.dx + a_d * ((x - self.x) / dt - self.dx)
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx) / max(scale, 1) * 100
        a = 1 / (1 + (1 / (2 * np.pi * cutoff)) / dt)
        self.x = self.x + a * (x - self.x)
        return self.x.copy()


fps, T = 30, 150
t = np.arange(T) / fps
truth = np.where(t < 2, 100.0, np.where(t < 3, 100 + (t - 2) * 300, 400.0))  # 像素；第 2–3 秒猫快速移动
noisy = truth + RNG.normal(0, 4, T)
ema = np.zeros(T); ema[0] = noisy[0]
for i in range(1, T):
    ema[i] = ema[i - 1] + 0.15 * (noisy[i] - ema[i - 1])
f = OneEuro(); euro = np.array([f(np.array([v]), ti, scale=200)[0] for v, ti in zip(noisy, t)])

still, moving = t < 2, (t >= 2) & (t < 3)
for name, y in (('raw', noisy), ('EMA a=0.15', ema), ('One-Euro', euro)):
    print(f'{name:11s}  静止时抖动(std) {np.std(y[still] - truth[still]):5.2f}px   运动时滞后(mean abs err) {np.mean(abs(y[moving] - truth[moving])):6.2f}px')
plt.figure(figsize=(8, 3))
plt.plot(t, noisy, '.', ms=3, c='lightgray', label='raw'); plt.plot(t, ema, label='EMA'); plt.plot(t, euro, label='One-Euro'); plt.plot(t, truth, 'k--', lw=1, label='truth')
plt.xlabel('s'); plt.ylabel('x (px)'); plt.legend(); plt.tight_layout(); plt.show()
""")

# =============================================================================
# 4. Features
# =============================================================================
md(r"""
## 4. 从关键点到可解释特征 🧪

### 4.1 理论依据：Feline Grimace Scale（FGS）

FGS（Evangelista et al., 2019）是兽医用来评估猫**疼痛**的量表，经过效度验证，由 5 个动作单元组成，每项 0–2 分：
1. **耳朵位置**：耳朵向两侧分开、向外旋转
2. **眼眶收紧**：眼睛眯起
3. **口鼻紧张**：口鼻部变成椭圆形
4. **胡须变化**：胡须变直、向前
5. **头部位置**：头低于肩线

Feighelstein et al. (2022) 证明：**只用猫脸关键点算出的几何特征**，就能以 72% 以上的准确率自动识别疼痛（29 只短毛家猫，绝育手术前后不同时间点的照片；关键点方案和我们相同），这就是我们这条技术路线的直接先例。

Bennett et al. (2017) 用 CatFACS 给加拿大一家收容所里 29 只猫的 275 段 4 秒视频编码，用聚类分析**提出假设**（作者明确说是 generate hypotheses）：约一半的数据可以分成三组，作者认为分别对应**放松投入、恐惧和挫败**。
- 恐惧：眨眼、半眨眼，低强度时头和视线偏向左侧；
- 挫败：哈气、舔鼻子、张嘴、上唇抬起、皱鼻、张口、叫声等，**主要是嘴部动作**；
- 放松投入：头和视线偏向右侧。

这对我们有两点提醒：(1) 这项研究里，和情绪最相关的面部动作**主要在嘴部、眨眼和头部转向，而不是耳朵**，这些我们目前都没有测；(2) “眨眼”在这里和恐惧有关，而 Humphrey et al. (2020) 发现“慢眨眼”和放松有关。**同一个动作在不同情境下意义相反**，这正是只看形态不能判断情绪的原因。

> ⚠️ **诚实的边界**：FGS 验证的是**疼痛**，不是“焦虑”或“开心”。把这些特征用于**情绪**判断，是一个**有文献方向支持、但尚未针对我们的 4 类状态验证过的假设**。所以这一节标为 🧪。

### 4.2 我们用到的关键点编号

CatFLW 没有公布 48 个点各自的名字。我们用训练集的**平均脸形**加上左右对称映射，自己确定了编号（下图可以核对）：

| 区域 | 左（画面左侧） | 右 |
|---|---|---|
| 耳朵轮廓（外侧底 → 耳尖 → 内侧底） | 22, 23, **24（耳尖）**, 25, 26 | 27, 28, **29（耳尖）**, 30, 31 |
| 眼睛：外眼角 / 内眼角 / 上眼睑 / 下眼睑 | 4 / 5 / 6 / 7 | 8 / 9 / 10 / 11 |
| 口鼻两侧（胡须垫） | 32 | 35 |
| 下巴 / 嘴 | 0, 2（中线） | |
""")

code(r"""
# 4.3 核对编号：平均脸 + 区域着色
mean_face = np.array(json.loads((PACK / 'mean_face.json').read_text()))
REG = {'ear_L': [22, 23, 24, 25, 26], 'ear_R': [27, 28, 29, 30, 31], 'eye_L': [4, 5, 6, 7], 'eye_R': [8, 9, 10, 11],
       'muzzle': [32, 35], 'chin': [0, 2]}
colors = {'ear_L': 'tab:red', 'ear_R': 'tab:red', 'eye_L': 'tab:blue', 'eye_R': 'tab:blue', 'muzzle': 'tab:green', 'chin': 'tab:purple'}
plt.figure(figsize=(4.4, 4.4))
plt.scatter(*mean_face.T, c='lightgray', s=20)
for r, idx in REG.items():
    plt.scatter(*mean_face[idx].T, c=colors[r], s=30, label=r if r[-1] != 'R' else None)
for i, (x, y) in enumerate(mean_face):
    plt.text(x + 0.008, y, str(i), fontsize=6)
plt.gca().invert_yaxis(); plt.gca().set_aspect('equal'); plt.legend(fontsize=7, loc='lower right'); plt.title('CatFLW mean face (bbox-normalized)'); plt.tight_layout(); plt.show()
""")

md(r"""
### 4.4 特征定义（全部对尺度、平移、平面内旋转不变）

记左右眼中心为 $e_L, e_R$（各自 4 个眼周点的平均），眼间距 $d=\|e_R-e_L\|$ 作为长度单位，眼线方向 $\hat x=(e_R-e_L)/d$。“向上”的方向 $\hat n$ 是垂直于眼线、背离下巴的单位向量。

| 特征 | 公式 | 对应 FGS / 行为学 | 预期方向 |
|---|---|---|---|
| `ear_height` 耳尖高度 | $\frac{1}{2}\sum_{s\in\{L,R\}}\frac{(t_s-\bar e)\cdot\hat n}{d}$，$t_s$ 为耳尖 | 耳朵位置 | 耳朵压平或后转 → 变小 |
| `ear_angle` 耳朵外展角 | 耳轴（耳底中点 → 耳尖）与 $\hat n$ 的夹角，向外为正，取左右平均（度） | 耳朵向两侧分开 | 紧张 → 变大 |
| `eye_aperture` 眼睛开合度 | $\frac{\|p_{\text{上}}-p_{\text{下}}\|}{\|p_{\text{外}}-p_{\text{内}}\|}$，左右平均 | 眼眶收紧 | 眯眼 → 变小；警觉 → 变大 |
| `muzzle_ratio` 口鼻比 | $\|p_{32}-p_{35}\|/d$ | 口鼻紧张（粗略代理） | 暂不设方向 |
| `head_drop` 头部下垂 | $\frac{(y_{\text{nose}}-y_{\text{withers}})}{\|p_{\text{withers}}-p_{\text{tailbase}}\|}$（身体模型） | 头部位置 | 头低于肩 → 变大。**只适用于站立或坐着的猫**，侧躺时没有意义 |
| `activity` 活动量 | $\frac{1}{K}\sum_k\|\tilde u_{k,t}-\tilde u_{k,t-1}\|\cdot\frac{\text{fps}}{\text{bbox size}}$（平滑后的点） | 活动水平 | 玩耍 → 大；冻结 → 接近 0 |

**缺失的信号**（需要如实说明）：胡须（不在 48 点里）、瞳孔大小（分辨率不够）、尾巴（身体 20 点里只有尾根，没有尾巴姿态）、叫声。尾巴和瞳孔都是猫情绪的重要信号，这是当前方案最大的缺口之一。
""")

code(r"""
# 4.5 特征计算
def face_features(p):
    eL, eR = p[REG['eye_L']].mean(0), p[REG['eye_R']].mean(0)
    d = np.linalg.norm(eR - eL) + 1e-6
    xhat = (eR - eL) / d
    n = np.array([xhat[1], -xhat[0]])
    ebar = (eL + eR) / 2
    if np.dot(p[REG['chin']].mean(0) - ebar, n) > 0:  # 让 n 背离下巴
        n = -n
    feats = {}
    tips = {'L': p[24], 'R': p[29]}
    feats['ear_height'] = np.mean([np.dot(tips[s] - ebar, n) / d for s in 'LR'])
    angles = []
    for s, base_idx, side in (('L', [22, 26], -1), ('R', [27, 31], +1)):
        axis = tips[s] - p[base_idx].mean(0)
        ang = np.degrees(np.arctan2(np.dot(axis, xhat) * side, np.dot(axis, n)))  # 向外为正
        angles.append(ang)
    feats['ear_angle'] = float(np.mean(angles))
    ap = [np.linalg.norm(p[6] - p[7]) / np.linalg.norm(p[4] - p[5]), np.linalg.norm(p[10] - p[11]) / np.linalg.norm(p[8] - p[9])]
    feats['eye_aperture'] = float(np.mean(ap))
    feats['muzzle_ratio'] = float(np.linalg.norm(p[32] - p[35]) / d)
    return feats


def body_features(p, conf=None, min_conf=0.25):
    ok = np.ones(len(p), bool) if conf is None else conf >= min_conf
    nose, withers, tail = 2, 18, 19
    if not (ok[nose] and ok[withers] and ok[tail]):
        return {'head_drop': np.nan}
    L = np.linalg.norm(p[withers] - p[tail]) + 1e-6
    return {'head_drop': float((p[nose, 1] - p[withers, 1]) / L)}


print('平均脸上的特征（“典型的猫”）：', {k: round(v, 3) for k, v in face_features(mean_face).items()})
""")

md(r"""
### 4.6 验证案例：关键点误差会怎样传递到特征上？

这是对“解释是否可信”最基本的检验：如果关键点稍有偏差，特征就剧烈变化，那么基于特征的解释（例如“耳朵低了 20%”）就没有意义。下面对 12 张验证脸分别用**真值关键点**和**模型预测的关键点**计算特征，比较两者的差距，并和特征在不同猫之间的自然变化幅度对比。
""")

code(r"""
# 4.7 特征的“信噪比”：模型误差 vs 猫与猫之间的差异
rows_gt, rows_pred = [], []
for s in face_samples:
    rgb = load_rgb(s['file'])
    pred, _ = face_net.predict(rgb, s['bbox'])
    rows_gt.append(face_features(np.array(s['pts'])))
    rows_pred.append(face_features(pred))
names = list(rows_gt[0])
G = np.array([[r[k] for k in names] for r in rows_gt])
Pm = np.array([[r[k] for k in names] for r in rows_pred])
print(f'{"feature":14s} {"between-cat SD":>15s} {"model error MAE":>16s} {"ratio":>7s}')
for j, k in enumerate(names):
    sd, mae = G[:, j].std(), np.abs(G[:, j] - Pm[:, j]).mean()
    print(f'{k:14s} {sd:15.3f} {mae:16.3f} {mae / sd:7.2f}')
    verdict = '可靠，可作为主要证据' if mae / sd < 0.3 else ('可用，但要谨慎' if mae / sd < 0.6 else '噪声太大，不应作为证据')
    print(f'{"":14s} → {verdict}')
print('\nratio = 模型误差 / 猫与猫之间的差异。远小于 1 → 特征反映的是猫的真实差异；接近 1 → 特征主要反映模型噪声。')
print('结论：耳尖高度最可靠；耳外展角、眼开合度和口鼻比的噪声约为真实差异的一半，')
print('所以在解释里它们应该作为辅助证据，并且需要多帧平均之后再使用。')
""")

md(r"""
### 4.8 个体基线：和“这只猫自己的平常”比，而不是和“一般的猫”比

不同品种的猫，耳朵、眼睛的形状差异很大：苏格兰折耳猫的耳朵天生就“低”。所以每个特征都要相对这只猫自己的基线做标准化：
$$z_{t,j}=\frac{f_{t,j}-\operatorname{median}_j^{\text{cat}}}{\sigma_j},\qquad \sigma_j=\max\big(1.4826\cdot\operatorname{MAD}_j^{\text{cat}},\ \varepsilon_j\big)$$
基线来自猫**安静时**的帧。这里用中位数和 MAD（中位数绝对偏差）而不是均值和标准差，因为它们对偶尔的错误帧更稳健。$1.4826$ 让 MAD 在正态分布下和标准差可比。

**分母为什么要有下限 $\varepsilon_j$？** $\varepsilon_j$ 是 4.7 节测出的**模型测量误差**。如果校准那几秒里猫几乎没动，MAD 会非常小，于是任何一点变化都会被放大成几个“标准差”。在真实视频上，这曾让眼睛开合度的 z 值达到 +7，并造成大量误报（见 4.12）。道理很简单：**比测量误差还小的变化，不能算作证据**。

这一步直接决定了 app 里解释文字的写法：“**比它平时**低 1.8 个单位”，而不是“比一般的猫低”。后者对折耳猫是错的。

从这里开始，所有特征都在 $z$ 空间中表示：0 = 这只猫的平常状态，±1 ≈ 一个“典型波动”。
""")

code(r"""
# 4.9 个体基线标准化（分母以模型测量误差为下限）
MODEL_ERROR = np.abs(G - Pm).mean(0)  # 4.7 节测出的每个特征的测量误差


class Baseline:
    def __init__(self, calm_frames, floor):
        X = np.asarray(calm_frames, float)
        self.med = np.nanmedian(X, 0)
        self.scale = np.maximum(1.4826 * np.nanmedian(np.abs(X - self.med), 0), floor) + 1e-6

    def z(self, x):
        return (np.asarray(x, float) - self.med) / self.scale


# 演示下限的作用：假设校准时猫几乎一动不动，特征只有微小抖动
rng_demo = np.random.default_rng(1)  # 独立的随机数，不影响后面合成数据的结果
calm = G[0] + rng_demo.normal(0, 0.1, (60, len(names))) * MODEL_ERROR * 0.2
probe = G[0] + MODEL_ERROR  # 一次只有“一个测量误差那么大”的变化
no_floor = Baseline(calm, floor=np.zeros(len(names)))
with_floor = Baseline(calm, floor=MODEL_ERROR)
print(f'{"feature":14s} {"z without floor":>16s} {"z with floor":>13s}')
for k, a, b in zip(names, no_floor.z(probe), with_floor.z(probe)):
    print(f'{k:14s} {a:16.1f} {b:13.1f}')
print('→ 没有下限时，一个测量误差大小的抖动就被放大成几十个 SD；有下限时约为 1 SD，不会触发任何标签。')
""")

md(r"""
### 4.10 头部姿态门控：特征只对“画面平面内”的旋转不变

4.4 节的特征以眼线为参照，所以猫在画面里歪头（平面内旋转）不影响结果。但猫**抬头、低头（俯仰 pitch）或转头（偏转 yaw）**时，脸在画面上的投影会被压缩：耳朵看起来“变低”了，其实耳朵根本没动。在真实视频里，这正是“猫抬头往上看”被误判成“耳朵后压”的原因（4.12）。

我们用两个只依赖眼睛和鼻子的**代理指标**来估计头部姿态：

- **偏转**：左右眼到鼻子距离的不对称程度，正脸时约为 0
$$\text{yaw}=\frac{\big|\,\|e_L-n\|-\|e_R-n\|\,\big|}{\|e_L-n\|+\|e_R-n\|}$$
- **俯仰**：鼻子到眼线的垂直距离除以眼间距，平均脸上约为 0.55。抬头或低头时，这个比值会变小
$$\text{pitch}=\frac{\big|(n-\bar e)\cdot\hat n\big|}{d}$$

**门控规则**（与实时 demo 一致）：$\text{yaw}>0.2$，或者 pitch 相对这只猫基线的变化超过 30%，就**暂停脸部特征**，这时的判断只依据身体。宁可说“看不清”，也不输出一个被视角扭曲的证据。
""")

code(r"""
# 4.11 演示：把一张真实猫脸“模拟抬头”，看耳尖高度如何被误判
def head_pose(p):
    eL, eR, nose = p[REG['eye_L']].mean(0), p[REG['eye_R']].mean(0), p[[12, 13]].mean(0)
    a, b = np.linalg.norm(eL - nose), np.linalg.norm(eR - nose)
    d = np.linalg.norm(eR - eL); xhat = (eR - eL) / d
    drop = abs(np.dot(nose - (eL + eR) / 2, np.array([-xhat[1], xhat[0]])))
    return abs(a - b) / (a + b), drop / d


def tilt(p, deg):  # 正交投影下的俯仰：垂直于眼线的分量乘以 cos θ（把脸近似为平面）
    eL, eR = p[REG['eye_L']].mean(0), p[REG['eye_R']].mean(0)
    c = (eL + eR) / 2; xhat = (eR - eL) / np.linalg.norm(eR - eL); n = np.array([-xhat[1], xhat[0]])
    q = p - c
    return c + np.outer(q @ xhat, xhat) + np.outer((q @ n) * np.cos(np.radians(deg)), n)


p0 = np.array(face_samples[0]['pts'])
ref_pitch = head_pose(p0)[1]
bl_one = Baseline(G[:1] + rng_demo.normal(0, 1, (60, len(names))) * MODEL_ERROR, floor=MODEL_ERROR)
rows = []
for deg in range(0, 61, 10):
    q = tilt(p0, deg)
    eh = face_features(q)['ear_height']
    yaw, pitch = head_pose(q)
    gated = abs(pitch / ref_pitch - 1) > 0.3 or yaw > 0.2
    rows.append((deg, eh, bl_one.z([face_features(q)[k] for k in names])[0], pitch / ref_pitch - 1, gated))
print(f'{"tilt":>5s} {"ear_height":>11s} {"z":>6s} {"pitch change":>13s}  gate')
for deg, eh, z, dp, g in rows:
    print(f'{deg:4d}° {eh:11.3f} {z:6.1f} {dp:+12.0%}   {"PAUSED" if g else "used"}')
print('\n发现：抬头 20–40° 时，耳尖高度已经下降了好几个 SD（会被判为“耳朵后压”），但门控要到约 45° 才启动。')
print('也就是说，30% 的门限只能拦住大幅度的抬头，小角度仍然会带来偏差。')
""")

md(r"""
**一个更好的做法（下一步，demo 尚未采用）。** 在“脸近似为平面”的假设下，耳尖高度和 pitch 比值被同一个因子 $\cos\theta$ 压缩。所以两者相除就能把俯仰的影响抵消掉：
$$\text{ear\_height}^{\text{corr}}=\text{ear\_height}\cdot\frac{\text{pitch}_{\text{baseline}}}{\text{pitch}}$$
下面验证这个修正在模拟抬头时是否稳定。真实的耳朵并不在脸的平面上，所以它只是**一阶近似**，需要用真实的抬头片段再检验。
""")

code(r"""
# 4.12 俯仰修正后的耳尖高度
print(f'{"tilt":>5s} {"raw ear_height":>15s} {"corrected":>10s}')
for deg in range(0, 61, 10):
    q = tilt(p0, deg)
    eh = face_features(q)['ear_height']
    pitch = head_pose(q)[1]
    print(f'{deg:4d}° {eh:15.3f} {eh * ref_pitch / pitch:10.3f}')
print('→ 在平面近似下，修正值保持不变；原始值随抬头角度单调下降。')
""")

md(r"""
### 4.13 真实视频验证：在一段家庭监控视频上发现的问题

我们把特征和反应分类（第 5 节第一层）接进实时程序，在一段约 35 秒的网络家庭监控视频上运行：“My cat thinks she's home alone with a ghost – Part 2”，YouTube 频道 Kiki The Dream Cat，https://www.youtube.com/watch?v=twzzFMcScww 。视频有版权，不放进数据包，这里只报告数字。视频里，一只猫独自在家，对画面外的东西反复警觉、凑近镜头。

| 版本 | 问题 | 各类帧数（共约 1030 帧） |
|---|---|---|
| 初版 | “警觉”占了 40%。检查后发现，370 帧中有 354 帧只是因为**眼睛开合度**超标：校准时 MAD 太小，z 值被放大到 +7 | alert 411，active 223，no clear change 190，ears back 45 |
| 加上测量误差下限，并规定噪声大的特征只能作辅助证据 | “警觉”降到 2%，现在都由耳尖高度触发（例如 +3.2 SD）。但出现了新问题：猫**抬头往上看**时被标成“耳朵后压” | no clear change 595，active 208，ears back 45，alert 21 |
| 再加上头部姿态门控 | 抬头的那一帧不再被误判；侧脸时暂停脸部特征 | no clear change 525，active 227，ears back 77，freeze 39 |

表中的帧数是修正过程中每一步的记录；之后 demo 又改进了活动量的计算（取关键点速度的中位数、改用视频时间戳），所以现在重新运行的数字会略有不同。

**这个案例说明了什么：**
1. 在合成数据上完全看不出这两个问题，**只有真实数据才能暴露它们**。
2. 两次修正都来自对**证据本身**的检查（这个特征可靠吗？这个视角可信吗？），而不是调阈值去凑结果。这就是 XAI 对开发者的价值。
3. 修正之后的结果**仍然没有经过人工标注验证**。例如猫贴近镜头时出现的 “freeze”，需要有人看视频确认。所以这一层的证据等级是 🧪 真实数据原型，而不是 ✅。
""")

# =============================================================================
# 5. State inference
# =============================================================================
md(r"""
## 5. 从特征到状态：先描述反应，再推断情绪

动物行为学给行为分类时，遵循的顺序是**行为谱（ethogram）**：先客观**描述**看得见的行为，再**解释**它意味着什么。我们照这个顺序把第 5 节分成两层：

| 层 | 问题 | 需要什么 | 证据等级 |
|---|---|---|---|
| **第一层：行为反应** | 相对这只猫的平常，它**现在的样子变成了什么**？ | 第 4 节的特征 + 规则，不需要情绪标签 | 🧪 真实数据原型（已在真实视频上运行） |
| **第二层：情绪推断** | 这个反应**意味着**什么情绪？ | 带情绪标签的真实数据 | 🧪 合成数据（本节只演示方法） |

### 5.0 第一层：行为反应分类（与实时 demo 完全一致）🧪 真实数据原型

| 反应 | 可观察的定义（z 值相对这只猫的基线） | 文献方向 |
|---|---|---|
| **no clear change** 没有明显变化 | 所有特征都在基线范围内 | 描述性 |
| **active** 活跃 / 探索 | 活动量 ≥ +2 | 描述性 |
| **alert** 警觉 / 定向 | 耳尖高度 ≥ +1.5，且不在剧烈运动；眼睛睁大（≥ +2）只能作为**补充证据** | 行为学中的定向反应；主人指南里“好奇时耳朵向前竖起”（Cats Protection） |
| **ears back** 耳朵后压 / 外转 | 耳尖高度 ≤ −1.5；或耳外展角 ≥ +2 **并且**耳尖高度 ≤ −0.5 | FGS 的耳朵动作单元 |
| **freeze** 僵住 | 耳朵后压，同时活动量 ≤ −1 | 恐惧相关行为 |
| **unknown** 看不清 | 没有可信的关键点 | — |

**两条来自证据检查的设计原则**（都在 4.7 和 4.13 节得到支持）：
1. **只有可靠的特征才能单独触发标签。** 耳尖高度的噪声只有真实差异的 0.15，可以单独触发；眼睛开合度和耳外展角的噪声约为真实差异的一半，只能作为补充证据。在真实视频上，违反这条原则曾造成 40% 的误报。
2. **一个新标签要在最近 0.3 秒内占多数（≥ 60%）才会显示。** 这样可以避免单帧噪声让标签闪来闪去。按时间而不是帧数计算，慢电脑上的延迟也一样；用“占多数”而不是“连续 N 帧”，标签在两个相邻状态之间来回跳时也能切换。
3. **进入和退出用不同的阈值（迟滞，v4 新增）。** 例如耳尖高度要低到 −1.5 才进入“耳朵后压”，回到 −1.0 以内才退出；活跃是 +2.0 进入、+1.5 退出；僵住的“静止”是 −1.0 进入、−0.5 退出。只用一个阈值时，数值在阈值附近徘徊，标签就会来回跳，事件被拆成碎片（5.0b 演示）。
""")

code(r"""
# 5.0 第一层规则：与 ml/catpose/reactions.py 相同的逻辑
Z_EAR_H, Z_AUX, Z_ACTIVE, Z_STILL = 1.5, 2.0, 2.0, -1.0
Z_EAR_H_STAY, Z_ACTIVE_STAY, Z_STILL_STAY = 1.0, 1.5, -0.5  # 迟滞：已经显示的标签用更宽松的退出阈值


def react(ear_h=np.nan, ear_a=np.nan, eye=np.nan, act=np.nan, prev=''):
    have_face = any(np.isfinite(v) for v in (ear_h, ear_a, eye))
    if not have_face and not np.isfinite(act):
        return 'unknown', []
    back_bar = Z_EAR_H_STAY if prev in ('ears back', 'freeze') else Z_EAR_H
    up_bar = Z_EAR_H_STAY if prev == 'alert' else Z_EAR_H
    active_bar = Z_ACTIVE_STAY if prev == 'active' else Z_ACTIVE
    still_bar = Z_STILL_STAY if prev == 'freeze' else Z_STILL
    back = [f'ear tip {ear_h:+.1f}'] if np.isfinite(ear_h) and ear_h <= -back_bar else []
    if np.isfinite(ear_a) and ear_a >= Z_AUX and (back or (np.isfinite(ear_h) and ear_h <= -0.5)):
        back.append(f'ear angle {ear_a:+.1f}')
    alert = [f'ear tip {ear_h:+.1f}'] if np.isfinite(ear_h) and ear_h >= up_bar else []
    if alert and np.isfinite(eye) and eye >= Z_AUX:
        alert.append(f'eyes {eye:+.1f}')
    still, active = np.isfinite(act) and act <= still_bar, np.isfinite(act) and act >= active_bar
    if back and still:
        return 'freeze', back + [f'activity {act:+.1f}']
    if back:
        return 'ears back', back
    if alert and not active:
        return 'alert', alert
    if active:
        return 'active', [f'activity {act:+.1f}']
    return 'no clear change', ['within baseline' if have_face else 'face paused, body only']


cases = {
    '平常状态': dict(ear_h=0.2, ear_a=0.1, eye=-0.3, act=0.0),
    '只有眼睛睁大（噪声大的特征单独出现）': dict(ear_h=0.1, ear_a=0.0, eye=3.0, act=0.0),
    '耳朵竖起 + 眼睛睁大': dict(ear_h=2.1, ear_a=-0.4, eye=2.4, act=0.3),
    '耳朵放低': dict(ear_h=-1.9, ear_a=1.0, eye=0.0, act=0.2),
    '只有耳外展（耳尖高度正常）': dict(ear_h=0.0, ear_a=2.6, eye=0.0, act=0.0),
    '耳朵放低 + 静止': dict(ear_h=-2.0, ear_a=2.5, eye=-1.0, act=-1.4),
    '快速移动': dict(ear_h=0.3, act=3.1),
    '侧脸（脸部特征被姿态门控暂停）': dict(act=0.4),
}
for name, z in cases.items():
    label, ev = react(**z)
    print(f'{name:22s} → {label:16s} {"; ".join(ev)}')
""")

code(r"""
# 5.0b 迟滞：耳尖高度在 −1.5 附近徘徊 10 秒（30 帧/秒），比较单阈值和进入/退出两个阈值
rng_h = np.random.default_rng(3)
t_h = np.arange(300) / 30
ear_h_seq = -1.5 + 0.35 * rng_h.standard_normal(300) + np.where((t_h > 3) & (t_h < 7), -0.4, 0.4)
switches = {}
for hyst in (False, True):
    prev, labs = '', []
    for e in ear_h_seq:
        lab, _ = react(ear_h=e, act=0.0, prev=prev if hyst else '')
        labs.append(lab); prev = lab
    switches[hyst] = sum(a != b for a, b in zip(labs, labs[1:]))
    print(f"{'进入/退出两个阈值' if hyst else '单一阈值        '}：标签切换 {switches[hyst]:3d} 次，'ears back' 占 {labs.count('ears back') / len(labs):.0%}")
print(f'真实情况：3–7 秒耳朵偏低，只应切换 2 次。迟滞把逐帧的切换从 {switches[False]} 次减到 {switches[True]} 次；')
print('剩下的抖动由 0.3 秒多数投票（5.0 的第 2 条原则）再压掉。两者是互补的：迟滞管阈值附近的徘徊，投票管单帧噪声。')
""")

md(r"""
### 5.1 第二层：情绪推断——必须先说清楚的数据现状

我们**没有**带情绪标签的真实数据。所以本节的所有数字都来自**合成数据**：我们按照文献中的**方向性**描述，写了一个数据生成器。本节演示的是**算法流程和解释方法本身是否正确**，**不能**用来说明“模型在真实猫身上的准确率”。真实标签的采集方案见第 9 节。

### 5.2 状态空间与合成数据的假设

| 状态 | 耳尖高度 | 耳外展 | 眼开合 | 头下垂 | 活动量 | 文献方向 |
|---|---|---|---|---|---|---|
| relaxed 放松 | 0 | 0 | −0.5（半闭） | 0 | −0.5 | 慢眨眼、半闭眼与放松相关（Humphrey et al., 2020） |
| alert 警觉/好奇 | +1.0 | −0.5 | +1.0 | −0.5 | 0 | 耳朵竖起前指、眼睛睁大 |
| playful 想玩 | +0.5 | 0 | +0.8 | +0.3 | +2.0 | 高活动、捕猎姿态（Leyhausen, 1979） |
| anxious 焦虑 | −1.5 | +1.5 | +0.3 | +1.0 | −0.3 | 耳朵压平外展、头低（FGS 的疼痛动作单元；主人指南里“有压力时耳朵压平或朝两侧”） |

每个类别的特征 $\sim\mathcal N(\mu_{\text{class}}, 0.9^2)$，特征之间加入相关性，再随机翻转 8% 的标签，模拟人工标注时的分歧。类别之间**有意设计了大量重叠**，因为真实的猫不会像教科书那样表现。
""")

code(r"""
# 5.3 合成数据
FEATS = ['ear_height', 'ear_angle', 'eye_aperture', 'head_drop', 'activity']
CLASSES = ['relaxed', 'alert', 'playful', 'anxious']
MU = np.array([[0, 0, -0.5, 0, -0.5],
               [1.0, -0.5, 1.0, -0.5, 0],
               [0.5, 0, 0.8, 0.3, 2.0],
               [-1.5, 1.5, 0.3, 1.0, -0.3]])
COV = 0.9 ** 2 * (np.eye(5) + 0.25 * (np.ones((5, 5)) - np.eye(5)) * np.array([[0, -1, 0, 0, 0], [-1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))


def make_data(n, rng, flip=0.08):
    y = rng.integers(0, 4, n)
    X = np.stack([rng.multivariate_normal(MU[c], COV) for c in y])
    noisy = rng.random(n) < flip
    y_obs = np.where(noisy, rng.integers(0, 4, n), y)
    return X, y_obs, y


X, y, y_clean = make_data(3000, RNG)
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
X_tr, X_cal, y_tr, y_cal = train_test_split(X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=0)
print('训练', len(y_tr), '/ 校准', len(y_cal), '/ 测试', len(y_te))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for c, name in enumerate(CLASSES):
    m = y_tr == c
    ax[0].scatter(X_tr[m, 0], X_tr[m, 1], s=5, alpha=0.5, label=name)
    ax[1].scatter(X_tr[m, 2], X_tr[m, 4], s=5, alpha=0.5)
ax[0].set_xlabel('ear_height (z)'); ax[0].set_ylabel('ear_angle (z)'); ax[0].legend(fontsize=7, markerscale=3)
ax[1].set_xlabel('eye_aperture (z)'); ax[1].set_ylabel('activity (z)')
plt.suptitle('Synthetic data (NOT real cats): heavy overlap is intentional'); plt.tight_layout(); plt.show()
""")

md(r"""
### 5.4 三种模型，从最透明到最强

**(a) 规则基线。** 直接把 5.0 节第一层的反应映射成情绪类别：没有明显变化 → relaxed，警觉 → alert，活跃 → playful，耳朵后压或僵住 → anxious。完全透明，不需要训练数据，是没有标签时唯一能上线的方案，也是后面所有模型的比较基准。注意：这个映射本身就是一个**需要验证的假设**，例如“活跃”也可能是被吓得逃跑，而不是想玩。

**(b) CART 决策树**（Breiman et al., 1984）。每次选择一个特征和阈值切分数据，使子节点的 Gini 不纯度下降最多：
$$\mathrm{Gini}(S)=1-\sum_c p_c^2,\qquad \Delta=\mathrm{Gini}(S)-\tfrac{|S_L|}{|S|}\mathrm{Gini}(S_L)-\tfrac{|S_R|}{|S|}\mathrm{Gini}(S_R)$$
深度限制在 3–4 层时，整棵树可以打印出来让人阅读，本身就是一种解释（*intrinsically interpretable*）。

**(c) 梯度提升树 GBDT**（Friedman, 2001）。把很多棵小树加起来：
$$F_m(x)=F_{m-1}(x)+\nu\,h_m(x),\qquad p(c\mid x)=\operatorname{softmax}_c\big(F^{(c)}(x)\big)$$
每棵新树 $h_m$ 拟合交叉熵损失的负梯度 $y_c-p_c$，也就是“上一轮还错了多少”。它通常更准，但一次决策由上百棵树共同决定，**不再能直接读懂**，所以需要第 6 节的事后解释方法。

**评价指标**：宏平均 F1（每个类别的 F1 取平均，不会因为类别不平衡而被多数类主导）和混淆矩阵。
""")

code(r"""
# 5.5 训练与比较
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix


def rule_model(X):
    # 把 5.0 的第一层反应直接映射成情绪类别，作为“不需要标签”的基线：
    # no clear change → relaxed，alert → alert，active → playful，ears back / freeze → anxious
    to_class = {'no clear change': 0, 'unknown': 0, 'alert': 1, 'active': 2, 'ears back': 3, 'freeze': 3}
    return np.array([to_class[react(ear_h=r[0], ear_a=r[1], eye=r[2], act=r[4])[0]] for r in X])


tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=0).fit(X_tr, y_tr)
gbdt = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.1, random_state=0).fit(X_tr, y_tr)

for name, pred in (('rules', rule_model(X_te)), ('tree d=3', tree.predict(X_te)), ('GBDT', gbdt.predict(X_te))):
    print(f'{name:9s} macro-F1 = {f1_score(y_te, pred, average="macro"):.3f}')
print('\n决策树全文（这就是模型本身，可以逐条读）：')
print(export_text(tree, feature_names=FEATS, class_names=CLASSES, decimals=2))
cm = confusion_matrix(y_te, gbdt.predict(X_te), normalize='true')
plt.figure(figsize=(3.8, 3.2)); plt.imshow(cm, cmap='Blues', vmin=0, vmax=1)
plt.xticks(range(4), CLASSES, rotation=30); plt.yticks(range(4), CLASSES)
for i, j in itertools.product(range(4), range(4)):
    plt.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center', fontsize=8)
plt.title('GBDT confusion (row-normalized)', fontsize=9); plt.tight_layout(); plt.show()
""")

md(r"""
### 5.6 概率校准：说“80% 把握”时，真的有 80% 是对的吗？

对**信任**来说，这可能比准确率更重要。如果系统说“80% 焦虑”，而这类判断实际只有 50% 是对的，主人就会被误导（过度信任）。反过来，如果系统总是很犹豫，主人就会忽视它（信任不足）。Lee & See (2004) 把这称为**信任校准**（trust calibration）：信任的程度应该和系统的实际可靠性相匹配。

**期望校准误差（ECE）**：把预测按置信度分成 $M$ 个桶，
$$\mathrm{ECE}=\sum_{m=1}^M\frac{|B_m|}{n}\big|\operatorname{acc}(B_m)-\operatorname{conf}(B_m)\big|$$

**温度缩放**（Guo et al., 2017）：只学习一个参数 $T$，$p=\operatorname{softmax}(z/T)$。在单独的校准集上最小化负对数似然。$T>1$ 让模型“谦虚一点”。它不改变预测的类别，所以准确率不变，只修正置信度。
""")

code(r"""
# 5.7 ECE 与温度缩放
from scipy.optimize import minimize_scalar


def ece(probs, y, bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def logits(model, X):
    return np.log(np.clip(model.predict_proba(X), 1e-9, 1))


def softmax(z, T=1.0):
    z = z / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


nll = lambda T: -np.log(softmax(logits(gbdt, X_cal), T)[np.arange(len(y_cal)), y_cal] + 1e-12).mean()
T_star = minimize_scalar(nll, bounds=(0.3, 5), method='bounded').x
p_raw, p_cal = softmax(logits(gbdt, X_te)), softmax(logits(gbdt, X_te), T_star)
print(f'温度 T* = {T_star:.2f}   ECE 校准前 {ece(p_raw, y_te):.3f} → 校准后 {ece(p_cal, y_te):.3f}   （准确率不变：{(p_raw.argmax(1) == p_cal.argmax(1)).mean():.0%} 的预测相同）')


def reliability(ax, p, y, title):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, 11); mids, accs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 5:
            mids.append(conf[m].mean()); accs.append((pred[m] == y[m]).mean())
    ax.plot([0, 1], [0, 1], 'k:', lw=1); ax.plot(mids, accs, 'o-'); ax.set_title(title, fontsize=9); ax.set_xlabel('confidence'); ax.set_ylabel('accuracy')


fig, ax = plt.subplots(1, 2, figsize=(7, 3))
reliability(ax[0], p_raw, y_te, f'before: ECE {ece(p_raw, y_te):.3f}'); reliability(ax[1], p_cal, y_te, f'after T={T_star:.2f}: ECE {ece(p_cal, y_te):.3f}')
plt.tight_layout(); plt.show()
""")

md(r"""
### 5.8 时序：猫的状态不会每一帧都跳变

单帧分类器的输出会在相邻帧之间“闪烁”（relaxed → anxious → relaxed）。如果机器人跟着这些闪烁行动，交互就会显得很神经质。我们用**隐马尔可夫模型（HMM）的前向滤波**（Rabiner, 1989）把状态的持续性加进去：
$$\alpha_t(s)\ \propto\ \underbrace{\frac{p(s\mid f_t)}{\pi(s)}}_{\text{分类器给出的似然}}\ \sum_{s'}A_{s's}\,\alpha_{t-1}(s'),\qquad A_{ss}=0.95$$
$A$ 是状态转移矩阵：留在原状态的概率是 0.95，意味着平均每个状态持续约 $1/(1-0.95)=20$ 步。这只用到过去的帧，所以可以实时运行。
""")

code(r"""
# 5.9 HMM 前向滤波：去掉闪烁
def hmm_filter(P, stay=0.95, prior=None):
    k = P.shape[1]
    A = np.full((k, k), (1 - stay) / (k - 1)); np.fill_diagonal(A, stay)
    prior = np.full(k, 1 / k) if prior is None else prior
    alpha, out = prior.copy(), []
    for p in P:
        alpha = (p / prior) * (alpha @ A)
        alpha /= alpha.sum(); out.append(alpha.copy())
    return np.array(out)


true_seq = np.repeat([0, 0, 3, 3, 0, 2], 40)  # relaxed → anxious → relaxed → playful
Xs = np.stack([RNG.multivariate_normal(MU[c], COV) for c in true_seq])
P_frame = softmax(logits(gbdt, Xs), T_star)
P_hmm = hmm_filter(P_frame)
switch = lambda s: int((np.diff(s) != 0).sum())
print(f'单帧：准确率 {(P_frame.argmax(1) == true_seq).mean():.0%}，状态切换 {switch(P_frame.argmax(1))} 次')
print(f'HMM ：准确率 {(P_hmm.argmax(1) == true_seq).mean():.0%}，状态切换 {switch(P_hmm.argmax(1))} 次（真实切换 {switch(true_seq)} 次）')
fig, ax = plt.subplots(2, 1, figsize=(9, 3.6), sharex=True)
ax[0].plot(P_frame[:, 3], lw=0.8, label='per-frame P(anxious)'); ax[0].plot(P_hmm[:, 3], lw=2, label='HMM P(anxious)'); ax[0].legend(fontsize=7)
ax[1].step(range(len(true_seq)), true_seq, 'k', lw=1, label='truth'); ax[1].step(range(len(true_seq)), P_hmm.argmax(1) + 0.05, 'C1', label='HMM')
ax[1].set_yticks(range(4)); ax[1].set_yticklabels(CLASSES, fontsize=7); ax[1].legend(fontsize=7); plt.tight_layout(); plt.show()
""")

md(r"""
### 5.10 实际部署的第二层：用随后的行为更新推测 🧪（概率是手设假设）

5.4–5.9 节在合成数据上比较了几种分类器。但真实系统**没有情绪标签**，训练不了它们。所以部署的版本换了一个思路：**同一个表情可以有几种含义，看它接下来做了什么再判断。** 例如耳朵压低，随后躲开更像紧张，随后去查看更像只是在专注听。这和行为学里“结合情境和后续行为解读信号”的做法一致。

**数学：朴素贝叶斯。** 候选状态 $s\in\{$紧张, 好奇, 想玩, 专注, 放松$\}$。第一层的反应 $r$ 给出起点，事件后 5 秒内观察到的线索 $c_1,\dots,c_m$ 逐条更新：
$$P(s\mid r,c_{1..m})\ \propto\ P(s\mid r)\ \prod_{i=1}^{m}P(c_i\mid s)$$
“朴素”指假设各条线索在给定状态时相互独立。这是简化，但每一步都能打印出来：主人看到的“为什么这么推测”就是这条更新轨迹。

**把握程度**（$p_1,p_2$ 是最高和第二高的概率）：

| 把握 | 条件 | 主人看到什么 |
|---|---|---|
| 高 | $p_1\ge0.6$ 且 $p_1-p_2\ge0.3$ | “看起来像：X”，附另一种可能和怎么分辨 |
| 中 | $p_1\ge0.45$ 且 $p_1-p_2\ge0.15$ | 同上 |
| 低 | 其他 | **不给情绪**，只写看到的行为 |

两条附加规则：依据脸部的反应（耳朵后压、僵住、警觉）在脸**大部分时间看不清**（< 30%）时直接定为“低”；**任何推测在脸看不清时最多给“中”**（v4）——只凭身体动作不应该得到高把握。
""")

code(r"""
# 5.10 朴素贝叶斯更新（数值与 catpose/interpret.py 的 DEFAULT_CONFIG 相同）
NB_STATES = ['tense', 'curious', 'playful', 'attentive', 'calm']
NB_TEXT = {'tense': 'tense', 'curious': 'curious', 'playful': 'playful', 'attentive': 'attentive', 'calm': 'calm'}
NB_PRIOR = {  # P(state | reaction)
    'ears_back': [.40, .15, .05, .30, .10], 'freeze': [.45, .05, .20, .25, .05],
    'alert': [.15, .50, .05, .25, .05], 'active': [.15, .30, .40, .05, .10]}
NB_LIK = {  # P(cue | state)
    'moves_away': [.60, .20, .10, .10, .05], 'flees': [.55, .08, .10, .04, .02],
    'approaches': [.05, .50, .50, .20, .10], 'plays': [.05, .20, .70, .05, .05],
    'settles': [.15, .30, .20, .50, .60], 'stays_still': [.50, .20, .30, .40, .30],
    'robot_before': [.30, .50, .60, .40, .20]}
FACE_BASED = {'ears_back', 'freeze', 'alert'}


def nb_interpret(reaction, cues, face_visible=1.0):
    post = np.array(NB_PRIOR[reaction], float); post /= post.sum()
    trail = [f'start ({reaction}): {NB_STATES[post.argmax()]} {post.max():.0%}']
    for c in cues:
        post = post * NB_LIK[c]; post /= post.sum()
        trail.append(f'+ {c:12s} -> {NB_STATES[post.argmax()]} {post.max():.0%}')
    order = np.argsort(-post); p1, p2 = post[order[0]], post[order[1]]
    if reaction in FACE_BASED and face_visible < 0.3:
        band = 'low'
    elif p1 >= 0.6 and p1 - p2 >= 0.3:
        band = 'high'
    elif p1 >= 0.45 and p1 - p2 >= 0.15:
        band = 'medium'
    else:
        band = 'low'
    if band == 'high' and face_visible < 0.3:
        band = 'medium'  # v4: body movement alone never earns high confidence
    return NB_STATES[order[0]], NB_STATES[order[1]], band, trail


examples = [
    ('ears_back', [], 1.0, 'ears down, nothing else seen'),
    ('ears_back', ['moves_away'], 1.0, 'ears down, then leaves'),
    ('ears_back', ['approaches', 'settles'], 1.0, 'ears down, then goes to look and calms'),
    ('active', ['plays'], 0.8, 'moving fast and pouncing, face visible'),
    ('active', ['plays'], 0.0, 'same, face never visible'),
    ('active', ['flees'], 0.0, 'moving fast while leaving'),
]
for reaction, cues, fv, title in examples:
    top, second, band, trail = nb_interpret(reaction, cues, fv)
    shown = f'looks like {top} ({band}), could also be {second}' if band != 'low' else 'no guess shown (low)'
    print(f'{title}\n   ' + '\n   '.join(trail) + f'\n   => {shown}\n')
""")

md(r"""
### 5.11 随后行为线索的定义：逃跑不是扑玩，蹲低不是离开 🧪

线索来自事件后 5 秒的轨迹：猫在不在画面里、它的框占画面的比例 $a$（离镜头远近的粗略代理）、活动量 $z_{act}$。记 $a_0$ 为事件前 1.5 秒的面积中位数，$a_{end}$ 为最后一段的面积中位数：

$$\text{fast}=\big[\,\text{time}(z_{act}\ge2.5)\ge0.5\,\text{s}\,\big],\qquad \text{still}=\big[\,\text{time}(z_{act}\le-1)\ge2\,\text{s}\,\big]$$
$$\text{away}=\big[\text{不在画面}\ge30\%\big]\ \lor\ \big[a_{end}<0.8\,a_0\ \land\ \lnot\,\text{still}\big]$$

| 线索 | v3 的规则 | v4 的规则 | 为什么改 |
|---|---|---|---|
| 扑玩 plays | fast | fast **且不是 away** | 受惊逃跑也动得很快，曾被推测成“想玩，高把握” |
| 快速跑开 flees | — | away **且** fast | 新增，指向紧张 |
| 离开 moves_away | 面积变小 | 面积变小**且没有保持不动** | 蹲低时框也会变小，曾被读成“离开” |
| 小车摄像头 | 同上 | **不判断远近** | 画面跟着小车动，面积变化可能来自小车 |

这些阈值同样没有经过标注验证。下面用合成轨迹检查规则本身是否按设计工作。
""")

code(r"""
# 5.11 线索规则：v3 和 v4 在四条合成轨迹上的对比（30 帧/秒，事件后 5 秒）
def cues_v3(area, act, present, a0):
    away = (1 - present.mean()) >= 0.3 or np.median(area[-40:]) < 0.8 * a0
    out = ['moves_away'] if away else (['approaches'] if area.max() >= 1.25 * a0 else [])
    if (act >= 2.5).sum() / 30 >= 0.5: out.append('plays')
    return out


def cues_v4(area, act, present, a0, moving_camera=False):
    fast = (act >= 2.5).sum() / 30 >= 0.5
    still = (act <= -1).sum() / 30 >= 2.0
    out = []
    if not moving_camera:
        away = (1 - present.mean()) >= 0.3 or (np.median(area[-40:]) < 0.8 * a0 and not still)
        if away: out.append('flees' if fast else 'moves_away')
        elif area.max() >= 1.25 * a0: out.append('approaches')
        if fast and not away: out.append('plays')
    if still: out.append('stays_still')
    return out


n = 150; ramp = np.linspace(0, 1, n); ones = np.ones(n)
tracks = {
    'runs off (fast, gets smaller)': (0.10 - 0.05 * ramp, 3.0 * ones),
    'pounces in place (fast, same size)': (0.10 * ones, 3.0 * ones),
    'crouches (still, gets smaller)': (0.10 - 0.05 * ramp, -1.5 * ones),
    'runs off, seen from the robot': (0.10 - 0.05 * ramp, 3.0 * ones),
}
for name, (area, act) in tracks.items():
    present = np.ones(n, bool)
    v4 = cues_v4(area, act, present, 0.10, moving_camera='robot' in name)
    print(f'{name:36s} v3: {cues_v3(area, act, present, 0.10)!s:28s} v4: {v4}')
print('\n这些线索进入 5.10 的推测后（脸看得见，只看线索规则本身的影响）：')
for name, (area, act) in list(tracks.items())[:3]:
    present = np.ones(n, bool)
    for tag, cues in (('v3', cues_v3(area, act, present, 0.10)), ('v4', cues_v4(area, act, present, 0.10))):
        top, _, band, _ = nb_interpret('active' if act[0] > 0 else 'freeze', cues, face_visible=1.0)
        print(f'{name:36s} {tag}: {top} ({band})')
print('-> v3 把逃跑推测成“想玩，高把握”；v4 改为“紧张，中等把握”。扑玩的结果不变。')
print('   蹲低：v3 把“变小”读成离开，这条假线索把“紧张”抬到了高把握；v4 只算保持不动，把握回到中等。')
""")

md(r"""
### 5.12 从主人的反馈学习：几条标注只“推动”，不推翻 🧪（模拟）

主人在 app 里对每个推测点“对 / 不对（那它更像是……）/ 说不准”，这些回答会变成标签 $y$。用它们更新 5.10 的概率时，把手设的数值当作 $k$ 条“虚拟标注”（**伪计数**，Dirichlet / Beta 先验）：
$$\hat P(s\mid r)=\frac{k\,P_0(s\mid r)+n_{r,s}}{k+n_r},\qquad \hat P(c\mid s)=\frac{k\,P_0(c\mid s)+n_{c,s}}{k+n_s}$$
$n_{r,s}$ 是反应为 $r$、标签为 $s$ 的事件数。$k=10$ 时，3 条标注只能小幅移动概率，几十条之后数据才开始主导。这样做是为了防止几次巧合就把系统带偏。

**要注意的偏差：** 主人的判断不是“真实情绪”，而是另一种推测；主人也可能被系统的推测影响（锚定）。所以标注时应先让主人看视频、写下自己的判断，再显示系统的推测；最好有第二个人独立标注，算一致性（9.2 节）。
""")

code(r"""
# 5.12 模拟：假设“耳朵后压”的真实含义比手设的更常是“专注”，看标注数量增加时先验如何移动
rng_l = np.random.default_rng(11)
p0 = np.array(NB_PRIOR['ears_back'], float)
p_true = np.array([.20, .10, .05, .55, .10])  # 模拟用的“真实”分布（未知）
k = 10
print(f'{"labels":>7s}  ' + '  '.join(f'{s:>9s}' for s in NB_STATES))
print(f'{"hand-set":>7s}  ' + '  '.join(f'{v:9.2f}' for v in p0))
for n_lab in (3, 10, 30, 100, 300):
    y = rng_l.choice(5, size=n_lab, p=p_true)
    post = (k * p0 + np.bincount(y, minlength=5)) / (k + n_lab)
    print(f'{n_lab:7d}  ' + '  '.join(f'{v:9.2f}' for v in post))
print(f'{"truth":>7s}  ' + '  '.join(f'{v:9.2f}' for v in p_true))
print('-> 3 条标注几乎不改变结论；约 30 条后，“专注”超过“紧张”。真实系统里这一步由')
print('   python -m catpose.interpret --events events.json --labels labels.csv 完成，标签可以直接从主人 app 导出。')
""")

# =============================================================================
# 6. XAI
# =============================================================================
md(r"""
## 6. 可解释性（XAI）：解释什么、给谁、怎么检验 🧪

### 6.1 先回答“XAI 到底解释的是什么”

这里最容易出现混淆，所以先把边界说清楚：

> **XAI 解释的是模型为什么这样判断，而不是猫真正在想什么。**

模型说“焦虑”，是因为耳朵特征偏低、外展角偏大。这个解释可以是**忠实的**（faithful，确实是模型做判断的原因），但模型本身仍然可能是错的。所以一个诚实的解释系统必须同时给出三样东西：**依据**（看到了什么）、**推理**（为什么据此得出这个判断）、**把握程度**（有多确定，以及哪些信号没看到）。

### 6.2 三层解释，对应不同的受众和问题

| 层 | 问题 | 方法 | 受众 |
|---|---|---|---|
| **感知层** | 模型是不是“看”对了地方？ | 遮挡敏感度（Zeiler & Fergus, 2014） | 开发者 |
| **推断层** | 哪些特征让它得出这个判断？如果怎样，它就会判断成别的？ | Shapley 值（Shapley, 1953；Lundberg & Lee, 2017）、反事实（Wachter et al., 2017）、对比解释（Miller, 2019） | 开发者 + 主人 |
| **呈现层** | 主人看到什么？能据此做什么？ | 关键点高亮 + 模板化的文字 + 有约束的 LLM 改写 + 建议 | 主人 |

然后是很多 XAI 项目缺少的一步：**评估解释本身**（6.8 节）。
""")

md(r"""
### 6.3 感知层：遮挡敏感度

**思路**：用一个灰色方块依次遮住图像的不同位置，看关键点的预测会移动多少。移动越多，说明那个区域对这个关键点越重要：
$$S(r)=\big\|\hat u_k(x)-\hat u_k(x\odot\bar M_r)\big\|$$
$\bar M_r$ 表示遮住区域 $r$。这种方法不需要访问模型内部（适用于任何黑盒，包括机器人上的 ONNX 文件），结果也容易理解。

**验证目标**：敏感区域应该落在**这只猫身上**。如果主要落在背景（沙发、地毯）上，说明模型学到的是“捷径”（shortcut），那样的话后面所有基于耳朵的解释都不可信。我们同时看它落在猫身上的**哪个部位**。
""")

code(r"""
# 6.4 在真实身体模型上做遮挡分析（约 20 秒）
s = body_samples[0]
rgb = load_rgb(s['file'])
m = body_net.crop_matrix(s['bbox'])
patch = cv2.warpAffine(rgb, m, (256, 256), borderValue=(0, 0, 0))
net_notta = PosePredictor('body', tta=False)
base_pts, base_conf = net_notta.run_patch(patch)
K_EAR = BODY_NAMES.index('left_ear')
size, stride = 40, 16
grid = list(range(0, 256 - size + 1, stride))
S = np.zeros((len(grid), len(grid)))
for i, y0 in enumerate(grid):
    for j, x0 in enumerate(grid):
        occ = patch.copy(); occ[y0:y0 + size, x0:x0 + size] = 127
        p, _ = net_notta.run_patch(occ)
        S[i, j] = np.linalg.norm(p[K_EAR] - base_pts[K_EAR])
S_img = cv2.resize(S.astype(np.float32), (256, 256), interpolation=cv2.INTER_CUBIC)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(patch); ax[0].plot(*base_pts[K_EAR], 'r*', ms=14); ax[0].set_title(f'left_ear (conf {base_conf[K_EAR]:.2f})', fontsize=9)
ax[1].imshow(patch); ax[1].imshow(S_img, cmap='jet', alpha=0.5); ax[1].plot(*base_pts[K_EAR], 'w*', ms=14); ax[1].set_title('occlusion sensitivity (px moved)', fontsize=9)
for a in ax: a.axis('off')
plt.tight_layout(); plt.show()
def region_mask(pts):  # 把原图坐标的点映射到裁剪图，取外接框
    q = np.asarray(pts, float) @ m[:, :2].T + m[:, 2]
    (x0, y0), (x1, y1) = q.min(0), q.max(0)
    mask = np.zeros((256, 256), bool); mask[max(int(y0), 0):int(y1) + 1, max(int(x0), 0):int(x1) + 1] = True
    return mask


bx0, by0, bx1, by1 = s['bbox']
cat_mask = region_mask([[bx0, by0], [bx1, by1]])
gt = np.array(s['pts']); vis = np.array(s['vis']) > 0
head_idx = [i for i in range(5) if vis[i]]
head_pts = gt[head_idx]
pad = 0.3 * np.ptp(head_pts, axis=0).max()
head_mask = region_mask(np.vstack([head_pts.min(0) - pad, head_pts.max(0) + pad]))
total = S_img.clip(0).sum()
print(f'敏感度落在这只猫的包围框内：{S_img.clip(0)[cat_mask].sum() / total:.0%}（包围框只占裁剪图面积的 {cat_mask.mean():.0%}）')
print(f'敏感度落在头部区域内      ：{S_img.clip(0)[head_mask].sum() / total:.0%}（头部区域只占 {head_mask.mean():.0%}）')
print('解读：敏感区域集中在猫的脸上，不在背景上，所以没有发现“捷径”。但它也不在耳朵本身：')
print('模型是借助整张脸的结构来推断耳朵位置的。这意味着脸被遮住时，耳朵的点也会跟着出错；')
print(f'这时应该依靠置信度门控，而不是继续相信耳朵特征（这只猫左耳的置信度只有 {base_conf[K_EAR]:.2f}）。')
""")

md(r"""
### 6.5 推断层：Shapley 值，完全按照定义计算

**问题**：模型对这一只猫输出 $P(\text{anxious})=0.83$，而所有猫的平均值是 $0.25$。多出来的 $0.58$ 应该归功于哪些特征？

**Shapley 值**（来自合作博弈论）是唯一同时满足“效率、对称、虚拟、可加”四条公理的分配方式：
$$\phi_j=\sum_{S\subseteq F\setminus\{j\}}\frac{|S|!\,(|F|-|S|-1)!}{|F|!}\Big[v(S\cup\{j\})-v(S)\Big]$$
$v(S)$ 表示“只知道特征集合 $S$ 的取值时，模型的期望输出”。我们用背景样本来填补未知的特征（*interventional* 版本）：$v(S)=\mathbb E_{b\sim\mathcal D}\big[f(x_S, b_{\bar S})\big]$。

**效率公理**保证了解释是“记得上账”的：$\sum_j\phi_j=f(x)-\mathbb E[f]$。

我们只有 5 个特征，一共 $2^5=32$ 个子集，所以**不用近似，直接按定义精确计算**。这比调用 `shap` 库更透明：每一步都能看到。在真实系统里，特征更多时可以换成 TreeSHAP（对树模型精确且快速）。
""")

code(r"""
# 6.6 精确 Shapley 值
BG = X_tr[RNG.choice(len(X_tr), 150, replace=False)]  # 背景分布（代表“平均的猫”）


def f_prob(model, X, cls):
    return softmax(logits(model, X), T_star)[:, cls]


def shapley_exact(model, x, cls, bg=BG):
    d = len(x)
    cache = {}

    def v(S):
        key = tuple(sorted(S))
        if key not in cache:
            Xb = bg.copy()
            if S:
                Xb[:, list(S)] = x[list(S)]
            cache[key] = f_prob(model, Xb, cls).mean()
        return cache[key]

    phi = np.zeros(d)
    for j in range(d):
        others = [i for i in range(d) if i != j]
        for r in range(d):
            for S in itertools.combinations(others, r):
                w = math.factorial(len(S)) * math.factorial(d - len(S) - 1) / math.factorial(d)
                phi[j] += w * (v(set(S) | {j}) - v(set(S)))
    return phi, v(set()), v(set(range(d)))


# 找一只被判为 anxious 的测试猫
ANX = CLASSES.index('anxious')
idx = int(np.argmax((p_cal.argmax(1) == ANX) & (y_te == ANX)))
x0 = X_te[idx]
phi, base, fx = shapley_exact(gbdt, x0, ANX)
print('这只猫的特征 (z):', dict(zip(FEATS, x0.round(2))))
print(f'E[f] = {base:.3f}   f(x) = {fx:.3f}   Σφ = {phi.sum():.3f}   f(x) − E[f] = {fx - base:.3f}   ← 效率公理成立')
order = np.argsort(-np.abs(phi))
plt.figure(figsize=(5.5, 2.6))
plt.barh([FEATS[i] for i in order][::-1], phi[order][::-1], color=['C3' if v > 0 else 'C0' for v in phi[order][::-1]])
plt.axvline(0, color='k', lw=0.8); plt.xlabel('contribution to P(anxious)'); plt.title(f'P(anxious): {base:.2f} (avg cat) -> {fx:.2f} (this cat)', fontsize=9); plt.tight_layout(); plt.show()
""")

md(r"""
### 6.7 对比解释与反事实：人真正想问的是“为什么是 A 而不是 B”

Miller (2019) 综述了社会科学里关于“人如何解释”的研究，其中一个核心结论是：人需要的解释通常是**对比式**的。主人不会问“为什么是焦虑”，而是会问“**为什么是焦虑，而不是只是警觉？**”。对比解释只需要看两个类别的 Shapley 值之差：$\phi_j^{A}-\phi_j^{B}$。

**反事实解释**（Wachter et al., 2017）回答另一个问题：“**要改变什么，结论才会不同？**”形式化为：
$$x'=\arg\min_{x'}\ \|x'-x\|_1\quad\text{s.t.}\quad g(x')\neq\text{anxious}$$
在我们的系统里，“改变”最自然的方向是**回到这只猫的平常状态**（$z=0$，见 4.8 节）。特征只有 5 个，所以可以**精确枚举**全部 $2^5=32$ 种组合：把组合里的特征设回 0，找出能让判断不再是焦虑的**最小**组合。反事实对主人很有用，因为它自然地变成了**可以观察的线索**：“只要它的耳朵回到平常的位置，系统就不会再判断为焦虑。”
""")

code(r"""
# 6.8 对比解释 + 反事实
ALERT = CLASSES.index('alert')
phi_alert, _, _ = shapley_exact(gbdt, x0, ALERT)
contrast = phi - phi_alert
print('为什么是 anxious 而不是 alert（正值 = 支持 anxious 的一边）：')
for i in np.argsort(-contrast):
    print(f'  {FEATS[i]:13s} {contrast[i]:+.3f}')


def counterfactual_to_baseline(model, x, avoid):
    # 枚举所有特征组合，把组合内的特征设回 0（这只猫的平常状态），找最小的、能改变判断的组合
    prob = lambda v: softmax(logits(model, v[None]), T_star)[0]
    for r in range(1, len(x) + 1):
        found = []
        for S in itertools.combinations(range(len(x)), r):
            x_cf = x.copy(); x_cf[list(S)] = 0
            p = prob(x_cf)
            if p.argmax() != avoid:
                found.append((p[avoid], S, p))
        if found:
            return min(found)  # 同样大小的组合里，取 P(avoid) 最低的
    return None


cf = counterfactual_to_baseline(gbdt, x0, ANX)
if cf is None:
    print('\n即使所有特征都回到平常，判断仍然是焦虑')
else:
    p_anx, S, p_new = cf
    print(f'\n最少需要 {len(S)} 个特征回到平常（z=0），系统才不再判断为焦虑：')
    for j in S:
        print(f'  {FEATS[j]:13s} {x0[j]:+.2f} → 0')
    print(f'  → 判断变为: {CLASSES[p_new.argmax()]}（{p_new.max():.0%}）；P(anxious) 从 {fx:.2f} 降到 {p_anx:.2f}')
    print('  给主人的说法：“如果它的' + '、'.join({'ear_height': '耳朵高度', 'ear_angle': '耳朵角度', 'eye_aperture': '眼睛',
                                               'head_drop': '头部位置', 'activity': '活动量'}[FEATS[j]] for j in S)
          + '回到平常的样子，系统就不会再判断为焦虑。”')
""")

md(r"""
### 6.9 评估解释本身：忠实度与稳定性

一个解释“看起来合理”（plausible）并不代表它“忠实”（faithful）。检验的方法有两种：

**删除测试（fidelity / deletion test）。** 如果 Shapley 值说特征 A 最重要，那么把 A 换成背景值，模型输出应该掉得最多，至少要比随机换掉一个特征掉得多。如果做不到，说明解释在“编故事”。

**稳定性（stability）。** 对输入加上很小的噪声（小于关键点误差带来的特征噪声，见 4.7），解释的特征排序不应该大幅改变。我们用 Spearman 秩相关来衡量。

**人的层面的评估**（Doshi-Velez & Kim, 2017 提出三个层次：功能层、人的层、应用层）要在用户研究里做。

**一个常见误解是：必须先有准确的模型，才能研究可解释性。** 其实 XAI 研究的核心问题之一恰恰是：**当模型会犯错时，解释能不能帮人发现错误、合理地决定信任多少？**（信任校准，Lee & See, 2004）。一个完美的模型反而无法研究这个问题。我们的系统会犯**真实的、有规律的**错误（例如 4.13 节里抬头被误判成耳朵后压），每个判断又都能追溯到看得见的依据，正适合做这样的研究。

**现在就能做的研究设计：**
- **研究问题**：基于可见特征的解释，能否帮助猫主人判断一个不完美的识别系统什么时候可信？
- **材料**：20–30 个真实事件片段（由 demo 的事件报告自动导出），**有意包含系统判断错的片段**。
- **每个片段系统对不对**：两名组员独立判断**可见形态**（“耳朵真的放低了吗”），计算一致性。这比标注情绪容易得多，也可靠得多。
- **三种条件**（组内设计，每位参与者都看）：① 只有标签；② 标签 + 关键点高亮图 + 依据文字；③ 在 ② 的基础上，再加把握程度和“没有看到的信号”。
- **测量**：参与者能否判断出系统这次对不对（主要指标）；信任量表（Hoffman et al., 2018）；解释满意度；每个片段的用时。
- **参与者**：15–30 位猫主人。

这个设计研究的是“**解释对人的作用**”，结论完全建立在已有的东西上，不需要声称“AI 读懂了猫”。
""")

code(r"""
# 6.10 删除测试 + 稳定性（在 40 只测试猫上）
from scipy.stats import spearmanr
bg_mean = BG.mean(0)
drops_top, drops_rand, rhos = [], [], []
for i in RNG.choice(len(X_te), 40, replace=False):
    x = X_te[i]; c = int(p_cal[i].argmax())
    ph, _, fx_i = shapley_exact(gbdt, x, c)
    top = int(np.argmax(ph))
    x_del = x.copy(); x_del[top] = bg_mean[top]
    drops_top.append(fx_i - f_prob(gbdt, x_del[None], c)[0])
    r = int(RNG.integers(len(x)))
    x_r = x.copy(); x_r[r] = bg_mean[r]
    drops_rand.append(fx_i - f_prob(gbdt, x_r[None], c)[0])
    ph2, _, _ = shapley_exact(gbdt, x + RNG.normal(0, 0.1, len(x)), c)
    rhos.append(spearmanr(ph, ph2)[0])
print(f'删除 Shapley 最重要的特征：该类概率平均下降 {np.mean(drops_top):.3f}')
print(f'删除一个随机特征        ：该类概率平均下降 {np.mean(drops_rand):.3f}   → 前者明显更大 = 解释是忠实的')
print(f'加噪声后特征排序的 Spearman 相关：中位数 {np.median(rhos):.2f}（越接近 1 越稳定）')
""")

md(r"""
### 6.11 呈现层：给主人的“变化报告”

对应 deck 里的 NO.1 Visualization → NO.2 Explanation → NO.3 Action。主人真正想问的，通常不是“猫现在是什么情绪”，而是**“它为什么突然感兴趣了 / 突然不对劲了？”** 这是一个**对比式问题**（Miller, 2019）：为什么是现在，而不是刚才？所以报告以**状态变化**为单位组织，每次变化回答两个问题：

| 问题 | 内容 | 可信度 |
|---|---|---|
| **它身上变了什么？**（依据） | 变化前后两张图，圈出变化的部位；文字列出变化量 | 高：看得见，主人能自己核对 |
| **变化之前发生了什么？**（线索） | 变化前 3 秒内：机器人的动作、声音突然变大、画面其他地方有东西在动 | 只能说明**先后顺序**，不能证明因果 |

**第一个问题的数学：和几秒前的自己比。** 对每个特征 $j$，比较“变化被确认的时刻”和“变化前 1–2.5 秒”的 z 值：
$$\Delta_j = z_j(t_0) - \frac{1}{|W|}\sum_{t\in W} z_j(t),\qquad W=[t_0-2.5\,\text{s},\ t_0-1\,\text{s}]$$
只列出 $|\Delta_j|\ge 1$ SD 的特征。和几秒前的**同一只猫、差不多的位置**比，天然避开了品种、个体和大部分视角的差异，比和“平均的猫”比可靠得多。

**第二个问题的数学：线索要和它平时出现的频率比。** 单次“声音之后猫警觉了”说明不了什么，因为声音可能本来就一直有。所以对每种线索 $c$ 计算**提升度（lift）**：
$$\text{lift}(c)=\frac{P(\text{事件前 3 秒内出现 } c)}{P(\text{任意 3 秒窗口里出现 } c)}$$
lift ≈ 1 表示这条线索和事件无关；只有 lift 明显大于 1（例如 ≥ 2）、并且出现次数足够时，报告才会说“它似乎对……有反应”。在 4.13 节那段真实视频上，声音突变的 lift 是 1.0（视频全程有配乐），所以报告明确写出“**不能**当作线索”。

**写作原则：**
1. **每一句话都必须能追溯到一个数字。** 文字由模板从 $\Delta_j$、$z$ 值和 Shapley 值生成，而不是让 LLM 自由发挥。
2. **说“比它平时”，不说“比一般的猫”**（4.8 节）。
3. **把握程度写在明处**，包括哪些信号**没看到**（尾巴、胡须、叫声）。
4. **建议来自规则表**（兽医和行为学指南，例如 Ellis et al., 2013），不由模型生成；持续异常时建议就医。
5. **LLM 只做改写**：输入是结构化的事实 JSON，要求它“只能使用给定事实、不能做诊断、必须保留不确定性”，输出再做一次事实核对。这回答了 deck 里“Use LLM…”的问题：LLM 适合负责措辞，但不适合负责判断。
6. **单次事件说“在……之后”，不说“因为……”。** 只有同样的先后关系反复出现、lift 足够高，才说“它似乎对……更感兴趣”，并且写出次数。

**v4 的呈现层改动：**
- **事件合并**：同一个反应在 2 秒内再次出现，算同一个事件。测试视频里，一次“活跃”原来被拆成 3 张卡片，现在是 1 张。
- **声音只写“突然出现较大的声音”**，不写分贝数。安静背景下的分贝差可以大到 60 dB 以上，对主人没有意义，还会吓到人。
- **主人 app**（`python -m catpose.app`）：手机上的三个页面（今天 / 时刻 / 记录）和每个时刻的详情页。详情页把**视频、推测、另一种可能、怎么分辨、依据**放在一起；主人点“不对”时要选“那它更像是……”，这个回答保存在分析电脑上，并能导出成 5.12 节的标签文件。解释由此从单向展示变成**可纠正的中间层**（类似概念瓶颈模型里的人工干预）。

实时 demo 用 `--report-dir` 参数就能生成这样一份 HTML 报告（`ml/catpose/events.py`）。下面先用一段模拟的时间线演示 lift 的计算，再生成一张单个事件的卡片。
""")

code(r"""
# 6.11b 模拟一天的互动：哪种机器人动作之后，猫更常开始参与？
T_s, rng_c = 3 * 3600, np.random.default_rng(7)  # 3 小时，每秒一步
cues = {
    'robot: prey_slow': rng_c.random(T_s) < 0.01,
    'robot: sound': rng_c.random(T_s) < 0.01,
    'doorbell / noise': rng_c.random(T_s) < 0.003,
}
p_event = np.full(T_s, 0.002)  # 没有任何线索时，猫偶尔也会自己开始参与
for dt in range(1, 4):  # 隐藏的真实关系：prey_slow 之后 1–3 秒更可能开始参与；sound 无关
    p_event[dt:] += 0.12 * cues['robot: prey_slow'][:-dt]
events_t = np.nonzero(rng_c.random(T_s) < p_event)[0]


def lift(cue, events, window=3):
    before = np.mean([cue[max(0, t - window):t].any() for t in events])
    base = np.mean([cue[max(0, t - window):t].any() for t in range(window, len(cue), 1)])
    return before, base, before / base if base else float('inf')


print(f'一共 {len(events_t)} 次“开始参与”事件\n')
print(f'{"cue":18s} {"before events":>14s} {"any window":>11s} {"lift":>6s}  verdict')
for name, cue in cues.items():
    b, base, l = lift(cue, events_t)
    verdict = 'likely related, report with counts' if l >= 2 and b >= 0.3 else 'no evidence of a link'
    print(f'{name:18s} {b:14.0%} {base:11.0%} {l:6.1f}  {verdict}')
print('\n→ lift 找回了隐藏的关系：只有 prey_slow 明显更常出现在参与之前。')
print('  注意：lift 仍然只是相关。要确认因果，需要机器人主动、随机地切换玩法（第 8 节的 bandit 天然会这样做）。')
""")

code(r"""
# 6.12 生成一张 app 事件卡片：关键点高亮 + 文字 + 建议
FEATURE_TEXT = {
    'ear_height': ('ears', 'lower than usual', 'higher than usual'),
    'ear_angle': ('ears', 'pointing more forward than usual', 'rotated outward more than usual'),
    'eye_aperture': ('eyes', 'narrower than usual', 'wider open than usual'),
    'head_drop': ('head', 'held higher than usual', 'held lower than usual'),
    'activity': ('movement', 'less active than usual', 'more active than usual'),
}
FEATURE_POINTS = {'ear_height': [24, 29], 'ear_angle': [22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                  'eye_aperture': [4, 5, 6, 7, 8, 9, 10, 11], 'head_drop': [], 'activity': []}
SUGGESTIONS = {
    'anxious': ['Check for a trigger at this time (noise, visitor, delivery).', 'Make sure a hiding place is available near where it rests.',
                'If this repeats for 2-3 days, talk to a vet.'],
    'alert': ['No action needed; alertness is normal.'],
    'playful': ['A good moment for a play session; end with a "catch" so play feels complete.'],
    'relaxed': ['No action needed.'],
}
MISSING = 'tail, whiskers, pupils and sounds'


def explain_card(x, probs, phi, top_k=3):
    c = int(probs.argmax())
    facts = {'state': CLASSES[c], 'confidence': round(float(probs[c]), 2), 'evidence': []}
    for j in np.argsort(-phi)[:top_k]:
        if phi[j] <= 0.01:
            continue
        part, low, high = FEATURE_TEXT[FEATS[j]]
        facts['evidence'].append({'feature': FEATS[j], 'z': round(float(x[j]), 2), 'contribution': round(float(phi[j]), 2),
                                  'text': f'{part} {low if x[j] < 0 else high} ({x[j]:+.1f} SD vs its own baseline)'})
    level = 'likely' if probs[c] >= 0.7 else 'possibly'
    lines = [f'{level.capitalize()} {CLASSES[c]} (confidence {probs[c]:.0%}).', 'Why:']
    lines += [f'  - {e["text"]}' for e in facts['evidence']]
    lines += [f'Not observed: {MISSING}.', 'Suggestions:'] + [f'  - {t}' for t in SUGGESTIONS[CLASSES[c]]]
    return facts, '\n'.join(lines)


facts, text = explain_card(x0, softmax(logits(gbdt, x0[None]), T_star)[0], phi)
print(text)
print('\n给 LLM 的结构化事实（LLM 只能改写这些事实，不能增加内容）：')
print(json.dumps(facts, indent=1))

# 在一张真实猫脸上高亮对应的关键点（演示呈现方式；这张脸的 z 值来自合成模型，不是它真实的状态）
s = face_samples[1]; rgb = load_rgb(s['file']); pts, _ = face_net.predict(rgb, s['bbox'])
plt.figure(figsize=(4, 4)); plt.imshow(rgb); plt.scatter(*pts.T, s=6, c='white', alpha=0.6)
for e in facts['evidence']:
    idxs = FEATURE_POINTS[e['feature']]
    if idxs:
        plt.scatter(*pts[idxs].T, s=40, facecolors='none', edgecolors='red', lw=2)
x0b, y0b, x1b, y1b = s['bbox']; plt.xlim(x0b - 10, x1b + 10); plt.ylim(y1b + 10, y0b - 10); plt.axis('off')
plt.title('App card image: evidence regions circled', fontsize=9); plt.tight_layout(); plt.show()
""")

md(r"""
### 6.13 小结：我们的 XAI 能说什么、不能说什么

| 可以说 ✅ | 不能说 ❌ |
|---|---|
| “模型之所以判断为焦虑，主要是因为耳朵比它平时低、而且向外转” | “你的猫很焦虑”（这是事实判断，模型没有这个权威） |
| “这个判断的把握是 70%，而且没有看到尾巴和叫声” | “AI 读懂了你的猫” |
| “如果耳朵回到平常的位置，判断会变成警觉” | 基于单帧就给出的医学建议 |
| “这类情况连续出现 3 天，建议咨询兽医” | 用 LLM 生成、无法追溯到数据的解释 |
| “耳朵在 2 秒内明显竖起（+2.1 SD），之前机器人刚切换到慢速猎物模式；这周 7 次里有 5 次如此” | “它因为机器人而开心了” |
| “声音突变在这段录像里一直都有，不能当作线索” | 把一次巧合的先后顺序说成原因 |

**XAI 在这个系统里的附加价值**，一句话总结：它把一个无法验证的黑盒结论，变成了**主人可以用自己的眼睛核对的证据**（看图里被圈出来的耳朵），同时让我们（开发者）能够发现模型的错误（例如第 1 节的通道 bug、6.4 节的捷径检查）。
""")

md(r"""
### 6.14 部署系统的解释，是不是真正的依据？（课程第二层检验）✅ 真实视频

6.5–6.10 是在合成模型上演示解释方法。这一节检验的是**我们真正部署的系统**：主人卡片上写的“为什么”，是不是规则实际依据的东西。

因为标签和推测都来自透明的规则，这件事可以**精确地**检验：每个事件都记下了判断时用到的数值（投票给新标签的那几帧的中位数）和之前的标签（决定用进入还是退出阈值），再用**同一份规则**一次改动一处，看结论会不会变：

| 检验 | 做法 | 通过的意思 |
|---|---|---|
| 能重现 | 用记录的数值重跑规则 | 得到同样的标签 |
| 写出的就是原因 | 把卡片写出的特征改回它平时的值（$z=0$） | 标签改变。单独改回就会改变的叫“主要依据”，否则叫“补充” |
| 没有隐藏的原因 | 把卡片**没写**的特征全部改回平时的值 | 标签不变 |
| 推测能重现 | 用同样的后续线索重算推测 | 推测和把握都相同 |
| 哪一步是关键 | 逐条去掉后续线索 | 去掉后推测或把握会变的线索，写成“关键的一步” |

**这一节不再手抄规则**：下面直接导入数据包 `code/catpose_core/` 里的文件，它们就是 `ml/catpose/` 中的 `reactions.py`、`interpret.py`、`faithfulness.py` 和 `report_ui.py`，原样复制，只依赖 numpy。所以这里检验的就是程序本身。

**这件事为什么要做**：旧版卡片只列“和 2 秒前相比”变了什么，而标签是按“和它平时比”的偏离决定的。实际出现过：卡片标题写“耳朵压低”，依据里却只有“身体静下来了”，主人无从核对。下面也会统计，旧版卡片有几次漏掉了主要依据。
""")

code(r"""
# 6.14a 导入程序本身的规则文件，在测试视频的真实事件上逐条检验
sys.path.insert(0, str(PACK / 'code'))
from catpose_core.faithfulness import check_event
from catpose_core.interpret import DEFAULT_CONFIG as RULE_CFG

events_real = json.loads((PACK / 'real_events' / 'ghost_events.json').read_text())
print(f'测试视频（4.13 节那段）里的 {len(events_real)} 个事件，报告由 catpose.realtime 生成\n')
print(f'{"#":>2}  {"label":10s} {"cited: role":52s} {"all cited -> usual":18s} {"hidden?":8s} {"decisive cues":20s} result')
n_pass = n_old_miss = 0
for i, ev in enumerate(events_real):
    r = check_event(ev['reaction'], ev['z'], ev['prev_reaction'], ev['drivers'], ev['interp'], RULE_CFG)
    roles = ', '.join(f'{f}: {role}' for f, role in r['roles'].items())
    main = {f for f, role in r['roles'].items() if role == 'main'}
    old_named = bool(main & {k for k, _ in ev['changes']})
    n_pass += r['passed']
    n_old_miss += not old_named
    print(f'{i:>2}  {ev["reaction"]:10s} {roles:52s} {r["if_all_usual"]:18s} {"none" if r["nothing_hidden"] else "YES":8s} '
          f'{", ".join(r.get("decisive_cues", [])) or "-":20s} {"PASS" if r["passed"] else "FAIL"}')
print(f'\n通过 {n_pass}/{len(events_real)}。旧版卡片（只列“和 2 秒前相比”的变化）没有写出主要依据的事件：{n_old_miss}/{len(events_real)}。')
""")

code(r"""
# 6.14b 对照：一张写错依据的卡片，检验能不能把它查出来？
# 取第一个“耳朵压低”的事件，把卡片上的依据换成“身体动作”（真正的原因是耳朵）
ev = next(e for e in events_real if e['reaction'] == 'ears_back')
wrong = [{'feature': 'activity', 'z': ev['z'].get('activity', 0.0), 'test': '>=', 'bar': 2.0}]
r_bad = check_event(ev['reaction'], ev['z'], ev['prev_reaction'], wrong, ev['interp'], RULE_CFG)
print('写错依据的卡片：', 'PASS' if r_bad['passed'] else 'FAIL（被查出来了）')
print('  把写出的“身体动作”改回平时 -> 标签', r_bad['if_all_usual'], '（没有变：写出的不是原因）')
print('  把没写出的特征改回平时     ->', '标签不变' if r_bad['nothing_hidden'] else '标签变了（有没写出来的原因：耳朵）')
""")

code(r"""
# 6.14c 主人在卡片上读到的就是这些（文字由程序的 report_ui.py 生成，和 app 完全相同）
from types import SimpleNamespace
from catpose_core.report_ui import explain_basis, label_counterfactuals, guess_counterfactuals, faith_line, REACTION_PLAIN

ev = events_real[4]
r = check_event(ev['reaction'], ev['z'], ev['prev_reaction'], ev['drivers'], ev['interp'], RULE_CFG)
card = SimpleNamespace(reaction=ev['reaction'], drivers=ev['drivers'], faith=r, interp=ev['interp'])
print(f'为什么标为「{REACTION_PLAIN[ev["reaction"]]}」')
for b in explain_basis(card):
    print(f'  [{"主要依据" if b["role"] == "main" else "补充"}] {b["text"]}')
for line in label_counterfactuals(card) + guess_counterfactuals(card, RULE_CFG):
    print('  ' + line)
print('  ' + faith_line(card))
""")

md(r"""
**这一节能说明什么，不能说明什么。**
- 能说明：主人卡片上写的“为什么”，就是部署系统实际依据的东西，没有漏写、也没有多写；写错依据的卡片会被查出来（6.14b）。这是课程第二层检验在部署系统上的结果，不是在合成模型上的演示。
- 不能说明：规则对猫的判断是否正确。规则可能忠实地按照一个错误的阈值做判断。判断对不对，要用人工标注来检验（课程第一层，见 9.2 和 10.4）。
- 这个检验之所以能精确完成，是因为系统是**设计即可解释**的（关键点 → 几何特征 → 规则）。对一个黑盒模型，同样的问题只能近似回答（6.5–6.10 的方法）。这也是我们选择这种结构的理由之一。
""")

# =============================================================================
# 7. Interaction theory
# =============================================================================
md(r"""
## 7. 机器人交互的理论依据与形式化 📐

deck 里的交互目前只有“探索”和“捕猎式玩耍”两种策略，缺少理论支撑。这一节把它们接到动物行为学和 Animal-Computer Interaction（ACI）的文献上，并写成可以计算的形式。

### 7.1 行为学依据

| 设计决策 | 依据 | 对机器人的具体要求 |
|---|---|---|
| 用“猎物”的方式玩 | 猫的捕猎序列：**定位 → 潜伏/跟踪 → 追逐 → 扑击 → 抓住 → 处理**（Leyhausen, 1979）；物体玩耍在结构上与捕猎相同（Hall & Bradshaw, 1998） | 一次游戏要**完整**地经历这个序列，而不是一直逃跑 |
| 游戏要以“抓住”结束 | 永远抓不到的玩法（如激光笔）与主人报告的强迫性重复行为有关联（Kogan & Grigg, 2021） | deck 里的“装死结束”有依据：让猫最后“抓住”猎物 |
| 不断变换刺激 | 成年猫对同一个玩具会**快速习惯化**，换成新玩具后兴趣恢复（Hall, Bradshaw & Robinson, 2002） | 同一种玩法重复几次后兴趣下降，所以机器人要**学会轮换**（→ 第 8 节） |
| 可以选择不玩 | 猫的环境需求包括**可以预测、可以自己控制的社交接触**（Ellis et al., 2013） | 猫回避或焦虑时，机器人必须后退，不得追逐 |
| 独处的问题是真实存在的 | 猫也存在分离相关问题（Schwartz, 2002：136 例） | 目标是丰富独处时的环境，但**不是治疗**，也不应这样宣传 |
| 以动物为中心的设计伦理 | ACI 宣言（Mancini, 2011）：尊重动物的福祉、同意和选择 | 猫的“同意”以行为来体现：接近 = 同意，离开 = 不同意 |

### 7.2 形式化：部分可观测的马尔可夫决策过程（POMDP）

- **隐状态** $s_t$：猫的真实状态和兴趣水平（看不到）
- **观测** $o_t$：第 5 节的状态分布 $p(s_t\mid f_{1:t})$、猫和机器人之间的距离、猫是否在接近
- **动作** $a_t\in$ {`light` 灯光探索, `sound` 声音探索, `prey_slow` 慢速猎物移动, `prey_fast` 快速逃跑, `retreat` 后退/停止}
- **奖励**：$r_t = \text{engagement}_t - \lambda\cdot\text{stress}_t$，即“猫参与了”减去“猫受到惊吓”
- **安全屏蔽（shield）**（Alshiekh et al., 2018）：学习算法**永远不能**越过的硬规则：
$$\text{if } P(\text{anxious})>\tau \text{ or 猫在远离：}\quad a_t := \texttt{retreat}$$

**会话结构（有限状态机）**，由行为学决定，而不是学出来的：
```
IDLE ──猫靠近/醒着──▶ INVITE ──猫注意到──▶ HUNT(跟踪/追逐) ──扑击 N 次──▶ CATCH(装死) ──▶ COOL-DOWN ──▶ IDLE
   ▲                                              │
   └──────────── 任何时候检测到焦虑/回避 → RETREAT ─┘
```
**学习只发生在 HUNT 阶段内部**：选择哪一种刺激。这样学习的范围被限制在安全的、有理论支持的框架里。这就是第 8 节要做的事情。
""")

# =============================================================================
# 8. RL
# =============================================================================
md(r"""
## 8. 强化学习：让机器人学会这只猫喜欢怎么玩 🧪（模拟）

> **先澄清一个常见误解：强化学习不能让“识别”变得更准。**
> - 把猫的反应或情绪**认得更准**，靠的是**监督学习**：用带正确标签的样本训练和检验（第 5、9 节）。
> - **强化学习**解决的是“**机器人该怎么做**”：通过试错和奖励学习互动策略。
> - 两者的交汇点是**结果标签**：猫在机器人发起互动后是**靠近（参与）还是走开（回避）**，这件事可以直接观察到。它既是强化学习的奖励，也是一种不需要人工标注的训练标签（见 9.6）。

### 8.1 为什么先用 contextual bandit，而不是完整的强化学习

完整的深度强化学习需要成千上万次交互，而一只猫一天可能只玩几次，每次的“试错”都有真实的福祉成本。**上下文多臂老虎机**（contextual bandit）是强化学习的最简形式：每一步根据当前情境 $x$（猫的状态）选一个动作，立即得到奖励，不做长期规划。它的样本效率高、行为可以预测，而且决策过程可以解释（“选 A 是因为它在类似情况下 7 次里成功了 5 次”）。

### 8.2 Thompson sampling（Thompson, 1933；Russo et al., 2018）

对每一对（情境 $x$，动作 $a$），把“猫会参与”的概率 $\theta_{x,a}$ 看作一个 Beta 分布：
$$\theta_{x,a}\sim\mathrm{Beta}(\alpha_{x,a},\beta_{x,a}),\qquad \text{参与 }\Rightarrow\alpha\mathrel{+}=1,\quad \text{不参与 }\Rightarrow\beta\mathrel{+}=1$$
每一步从每个动作的 Beta 分布里**各抽一个样本**，选样本最大的那个动作。不确定的动作偶尔会抽到高值，因此会被试一试（探索）；已知好的动作大多数时候会被选中（利用）。

为了应对**习惯化**（猫对同一种玩法会厌倦），我们给旧的统计加上折扣：$\alpha\leftarrow\gamma\alpha$，$\beta\leftarrow\gamma\beta$，$\gamma=0.97$。这样近期的经验更重要。

### 8.3 模拟猫

模拟的猫是按照 7.1 节的文献方向写的，**参数全部是假设**：
- **偏好**：对每种刺激有一个隐藏的基础兴趣 $\bar I_a$。这只猫最喜欢 `prey_slow`（0.8），其次是 `prey_fast`（0.7）和 `light`（0.6），不太喜欢 `sound`（0.3）。
- **习惯化**（Hall et al., 2002）：刺激被使用后兴趣下降，$I_a\leftarrow 0.8\,I_a$；不用时慢慢恢复，$I_a\leftarrow I_a+0.1(\bar I_a-I_a)$。
- **紧张度** $A\in[0,1]$：每步自然衰减 $A\leftarrow0.9A$；外部事件（快递、噪音）以 3% 的概率让 $A$ 增加 0.6；猫已经紧张（$A>0.4$）时再使用强刺激（`sound`、`prey_fast`），算一次**应激事件**，$A$ 再增加 0.2。
- **参与概率**：$p=I_a(1-A)$，也就是兴趣越高、越不紧张，就越可能参与。
- **观测**：分类器看到的状态有 15% 的概率出错。

奖励定义为“参与了，而且没有造成应激”。我们比较四种策略：总是用 `prey_fast`、随机轮换、Thompson sampling，以及 Thompson sampling 加安全屏蔽。

> 模拟**只能**检验算法的逻辑是否成立（例如：能不能学到偏好？安全屏蔽是否有效？）。它**不能**说明真实的猫会这样反应。
""")

code(r"""
# 8.4 模拟环境与四种策略
ACTIONS = ['light', 'sound', 'prey_slow', 'prey_fast', 'retreat']
PLAY = ACTIONS[:-1]
INTENSE = {'sound', 'prey_fast'}
CONTEXTS = ['relaxed', 'anxious']


class SimCat:
    PREF = {'light': 0.6, 'sound': 0.3, 'prey_slow': 0.8, 'prey_fast': 0.7}  # 隐藏偏好，机器人看不到

    def __init__(self, rng, habit=0.8):
        self.rng, self.I, self.A, self.habit = rng, dict(self.PREF), 0.0, habit

    def observe(self):  # 分类器看到的状态：有 15% 的概率出错
        true = 'anxious' if self.A > 0.4 else 'relaxed'
        return true if self.rng.random() > 0.15 else CONTEXTS[int(self.rng.integers(2))]

    def step(self, a):
        for k in self.I:  # 没被使用的刺激慢慢恢复兴趣
            if k != a:
                self.I[k] += 0.1 * (self.PREF[k] - self.I[k])
        stressed = a in INTENSE and self.A > 0.4
        engaged = 0
        if a != 'retreat':
            engaged = int(self.rng.random() < self.I[a] * (1 - self.A))
            self.I[a] *= self.habit  # 习惯化
        self.A *= 0.7 if a == 'retreat' else 0.9  # 后退让猫更快平静
        if stressed:
            self.A += 0.2
        if self.rng.random() < 0.03:  # 外部事件：快递、噪音
            self.A += 0.6
        self.A = min(self.A, 1.0)
        return engaged, int(stressed)


def shield(obs, a):  # 硬规则：学习算法不能越过
    return 'retreat' if obs == 'anxious' else a


class Thompson:
    def __init__(self, rng, gamma=0.97):
        self.rng, self.gamma = rng, gamma
        self.ab = {(c, a): [1.0, 1.0] for c in CONTEXTS for a in PLAY}

    def act(self, obs):
        return max(PLAY, key=lambda a: self.rng.beta(*self.ab[(obs, a)]))

    def update(self, obs, a, r):
        if a not in PLAY:
            return
        for k in self.ab:  # 折扣旧经验，适应习惯化带来的变化
            if k[0] == obs:
                self.ab[k] = [max(1.0, v * self.gamma) for v in self.ab[k]]
        self.ab[(obs, a)][0 if r else 1] += 1


def run(policy, use_shield, steps=500, seeds=30, habit=0.8):
    eng, stress = np.zeros((seeds, steps)), np.zeros((seeds, steps))
    counts = {c: {a: 0 for a in ACTIONS} for c in CONTEXTS}
    for sd in range(seeds):
        rng = np.random.default_rng(100 + sd)
        cat, agent = SimCat(rng, habit), Thompson(rng)
        for t in range(steps):
            obs = cat.observe()
            if policy == 'fixed':
                a = 'prey_fast'
            elif policy == 'random':
                a = PLAY[int(rng.integers(4))]
            else:
                a = agent.act(obs)
            if use_shield:
                a = shield(obs, a)
            e, st = cat.step(a)
            if policy == 'thompson':
                agent.update(obs, a, e and not st)
            eng[sd, t], stress[sd, t] = e, st
            counts[obs][a] += 1
    return eng, stress, counts


results = {name: run(p, sh) for name, p, sh in (
    ('always prey_fast', 'fixed', False), ('random rotation', 'random', False), ('Thompson', 'thompson', False),
    ('random + shield', 'random', True), ('Thompson + shield', 'thompson', True))}
print(f'{"policy":20s} {"engagement / step":>18s} {"stress events / 100 steps":>27s}')
for name, (e, s_, _) in results.items():
    print(f'{name:20s} {e.mean():18.3f} {100 * s_.mean():27.2f}')
fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
for name, (e, s_, _) in results.items():
    ax[0].plot(np.cumsum(e.mean(0)), label=name); ax[1].plot(np.cumsum(s_.mean(0)), label=name)
ax[0].set_title('cumulative engagement (higher is better)', fontsize=9); ax[1].set_title('cumulative stress events (lower is better)', fontsize=9)
for a in ax: a.set_xlabel('step'); a.legend(fontsize=7)
plt.tight_layout(); plt.show()
for name in ('Thompson', 'Thompson + shield'):
    c = results[name][2]['relaxed']; tot = max(sum(c[a] for a in PLAY), 1)
    print(f'{name:18s} 在猫放松时的动作分布:', {a: f'{c[a] / tot:.0%}' for a in PLAY})
c = results['Thompson'][2]['anxious']; tot = max(sum(c.values()), 1)
print(f'{"Thompson":18s} 在猫焦虑时的动作分布:', {a: f'{c[a] / tot:.0%}' for a in PLAY})
print('隐藏偏好:', SimCat.PREF)
""")

md(r"""
### 8.5 诚实的结果：学习在这里帮了多少？

上表最重要的一行对比是 **random + shield 和 Thompson + shield**。在默认参数下（习惯化很强，每用一次兴趣降到 80%），两者的参与度几乎一样。原因是：习惯化越强，“不断轮换”本身就越接近最优，而随机策略天然就在轮换。所以在这组假设下，**带来好处的是安全屏蔽和轮换，而不是学习**。

另外，没有屏蔽的 Thompson 在猫焦虑时的动作分布几乎是均匀的：焦虑时参与率接近 0，所有动作的奖励都一样差，**学习算法从奖励里学不到“应该停下来”**。这正是安全规则必须写成硬约束、而不能指望学出来的原因。

那么学习什么时候才有用？下面扫描习惯化强度：习惯化越弱，猫的**稳定偏好**就越重要，学到偏好的价值也就越大。这个问题最终要靠真实数据回答：猫的习惯化到底有多快（Hall et al., 2002 提供了方向，但没有给出我们场景下的数值）。
""")

code(r"""
# 8.6 参数扫描：习惯化强度 vs 学习的价值
habits = [0.8, 0.9, 0.95, 0.99]
gain = []
print(f'{"habituation factor":>19s} {"random+shield":>14s} {"Thompson+shield":>16s} {"learning gain":>14s}')
for h in habits:
    e_r = run('random', True, seeds=15, habit=h)[0].mean()
    e_t = run('thompson', True, seeds=15, habit=h)[0].mean()
    gain.append(e_t / e_r - 1)
    print(f'{h:19.2f} {e_r:14.3f} {e_t:16.3f} {gain[-1]:+13.0%}')
plt.figure(figsize=(4.5, 2.8)); plt.plot(habits, [100 * g for g in gain], 'o-'); plt.axhline(0, color='k', lw=0.8)
plt.xlabel('habituation factor (1 = no habituation)'); plt.ylabel('Thompson vs random (%)'); plt.tight_layout(); plt.show()
""")

md(r"""
### 8.7 从主人的反馈中学习：偏好模型

参与度只是一个代理指标。猫“参与”了，并不一定代表它“开心”。主人的判断可以补上这一点。更可靠的做法不是让主人打分，而是让他们**比较**：“这两段视频里，哪一段猫看起来更享受？”**Bradley–Terry 模型**（Bradley & Terry, 1952）把这种比较转成每个玩法的潜在得分 $r_a$：
$$P(a\succ b)=\frac{e^{r_a}}{e^{r_a}+e^{r_b}}=\sigma(r_a-r_b)$$
这相当于一个没有截距的逻辑回归。它也是 RLHF（Christiano et al., 2017）里奖励模型的基础。学到的 $r_a$ 可以加进 bandit 的奖励：$r=\text{engagement}+\mu\,\hat r_a$。
""")

code(r"""
# 8.8 Bradley–Terry：从主人的“哪段更好”比较里恢复玩法得分
from sklearn.linear_model import LogisticRegression
true_r = np.array([0.8, -1.0, 1.5, 0.3])  # light, sound, prey_slow, prey_fast（隐藏的“真实享受程度”）
true_c = true_r - true_r.mean()


def fit_bt(n, rng):
    pairs, wins = [], []
    for _ in range(n):
        a, b = rng.choice(4, 2, replace=False)
        v = np.zeros(4); v[a], v[b] = 1, -1
        pairs.append(v); wins.append(int(rng.random() < 1 / (1 + np.exp(-(true_r[a] - true_r[b])))))
    import warnings
    with warnings.catch_warnings():  # 部分 numpy/BLAS 组合会在这里报无害的数值警告
        warnings.simplefilter('ignore', RuntimeWarning)
        coef = LogisticRegression(fit_intercept=False, C=10).fit(np.array(pairs), np.array(wins)).coef_[0]
    return coef - coef.mean()


print(f'{"":10s} {"true":>7s} {"60 comparisons":>15s} {"400 comparisons":>16s}')
est60, est400 = fit_bt(60, np.random.default_rng(1)), fit_bt(400, np.random.default_rng(2))
for a, t_, e1, e2 in zip(PLAY, true_c, est60, est400):
    print(f'{a:10s} {t_:+7.2f} {e1:+15.2f} {e2:+16.2f}')
rank = lambda v: [PLAY[i] for i in np.argsort(-v)]
print('\n真实排序        :', rank(true_c))
print('60 次比较的排序 :', rank(est60))
print('400 次比较的排序:', rank(est400))
print('→ 比较次数少时，得分和排序都不稳定。一个主人每天给 5 次比较，大约要 2–3 个月才够 400 次，')
print('  所以偏好模型要跨家庭共享一个先验，再用每只猫自己的数据微调。')
""")

# =============================================================================
# 9. Plan
# =============================================================================
md(r"""
## 9. 数据、标签与部署：提高可用性与信任度 📐

### 9.1 缺口总览（按对“信任”的影响排序）

| 缺口 | 影响 | 下一步 | 验收标准 |
|---|---|---|---|
| 没有真实的状态标签 | 第 5、6 节的数字都来自合成数据 | 9.2 标注方案 | 至少 3 只猫、每类 ≥ 200 段 |
| 标签本身是否可靠 | 人类都判断不一致，模型也不可能判断得准 | 两名评分者独立标注，计算 Cohen's κ | κ ≥ 0.6 的类别才进入训练 |
| 摄像头画面 ≠ 验证集照片 | 关键点指标可能偏乐观 | 收集 100–200 帧自家摄像头画面，标注后加入测试 | 摄像头测试集 PCK@0.1 ≥ 70% |
| 没有独立测试集 | 训练报告里的数字略偏乐观 | 按**猫**划分训练/验证/测试（不能按帧划分，否则会数据泄漏） | 测试集只在最后使用一次 |
| 尾巴、胡须、声音缺失 | 解释不完整 | 尾巴关键点（自行标注）；麦克风 + 叫声分类 | — |
| 交互效果未经验证 | 不知道机器人是否真的有帮助 | 9.4 用户与动物研究 | — |

### 9.2 真实标签的采集方案
1. **录像**：每只猫 2 周，每天在固定时段录像。同时记录环境事件（主人离开和回家、快递、吸尘器等）。
2. **切片**：每 3 秒切一个片段，用第 2 节的关键点模型筛掉看不清的片段。
3. **标注**：两名受过 FGS / CatFACS 训练的标注者，**只看视频**独立打标签（4 类 + “不确定”），看不到模型的输出。有条件的话请一位动物行为学家标注一个子集，作为金标准。
4. **一致性**：Cohen's κ（Cohen, 1960）：
$$\kappa=\frac{p_o-p_e}{1-p_e}$$
其中 $p_o$ 是实际一致率，$p_e$ 是随机情况下的期望一致率。不一致的片段交给第三人裁决，或者标为“不确定”。**不确定本身也是有用的信息**：模型在这些片段上也应该表现出低置信度。

### 9.3 训练流程与主动学习
- **先用规则模型上线**（5.5 节），因为它不需要标签。
- 每周用新标签重新训练决策树和 GBDT，并在**按猫划分**的测试集上和规则模型比较。**只有在测试集上更好、而且 ECE ≤ 0.05 时才替换**。
- **主动学习**（Settles, 2009）：优先把模型**最不确定**的片段（熵 $H(p)=-\sum p\log p$ 最大的片段）交给标注者，这样同样多的标注工作带来的提升最大。
- **主人反馈**（app 里的“判断正确/错误”按钮）作为**弱标签**：它的噪声比专家标签大，训练时权重更低，但数量多，而且覆盖的正是这只猫。
- **校准要持续监控**：每次更新都在校准集上重新拟合温度 $T$。

### 9.4 验证交互与 XAI 的效果：用户与动物研究
| 问题 | 设计 | 指标 |
|---|---|---|
| 机器人对猫有没有帮助？ | 同一只猫交替进行 A/B 周（有机器人 / 没有机器人），录像由不知道分组的人评分 | 活动量、焦虑片段比例、分离相关行为（Schwartz, 2002 中列出的行为）的次数 |
| 安全吗？ | 每一次安全屏蔽触发都记录下来并复查 | 屏蔽之后猫是否恢复；有没有漏掉的回避 |
| 解释能不能帮助主人？ | 主人分成两组：只看标签 vs 看完整解释卡片，组间比较 | 主人预测系统判断的准确度；对猫状态的判断和专家的一致度；信任量表（Hoffman et al., 2018） |
| 信任是否校准？ | 对比主人的信任度和系统的实际准确率 | 在系统判断错误时，主人能不能识别出来 |

### 9.5 外部数据集评估：能直接拿来用吗？

**Hugging Face 上的 cat-emotion-dataset**（KAITANG2003，13,592 张图，4 类情绪）。我们下载了它的标注文件并做了检查：

| 发现 | 数量 | 说明 |
|---|---|---|
| 推理文字里把猫写成了 “dog” | 346 条 | 标注者认错了动物，或者图里不是猫 |
| 视觉特征一栏是没填的模板 `ears,eyes,body,tail,env` | 26 条 | 生成失败，仍然给了情绪标签 |
| 推理说“警觉/好奇”，标签却是 Relaxed 或 Happy | 6,187 条（45%） | 数据集没有“警觉”这一类，只能硬塞进 4 类 |
| 类别不平衡 | Relaxed 62%，Sad 3.4% | — |
| 标注方法、标注者、一致性、图片来源 | 均未说明 | 许可证只写“仅供研究” |

每条标注都带有流畅的“思维链”解释，格式高度模板化，还出现了把猫写成狗这种模型典型的错误。所以这些标签几乎可以确定是**视觉语言模型自动生成的**，只能代表那个模型的看法。**结论**：不能当作标准答案，也不建议直接用来训练；它的**逐项形态描述**（例如 “ears back”）可以用来**检验我们的几何特征**：被描述为“耳朵后压”的图，耳尖高度是否确实更低？

**更可靠的数据集**：它们的标签都来自**实验情境**或**兽医量表**，而不是有人看图猜情绪。

| 数据集 | 标签怎么来的 | 能否获取 | 对我们的价值 |
|---|---|---|---|
| 猫疼痛数据集（Finka / Feighelstein / Martvel，Tech4Animals 实验室） | 绝育手术前 / 术后 1 小时，用 UNESP-Botucatu 量表核对 | 向作者申请 | 使用同样的 48 个 CatFLW 点，可直接检验耳朵和眼睛特征 |
| Feline Grimace Scale 评分图（Steagall 实验室） | 兽医逐项打分（每个动作单元 0–2 分） | 向作者申请；官网有练习图 | 逐项检验耳尖高度、眼睛开合度 |
| CatMeows（Zenodo，CC BY 4.0） | 梳毛 / 等待喂食 / **被单独放在陌生环境** | 公开 | 和“独处”直接相关，但数据是声音 |
| PMTNet 行为视频（Tu et al., 2026） | 5 类行为，1,283 段 | 待确认 | 和第一层“行为反应”最接近 |
| Ethogram of the Domestic Cat（2024） | 行为学家从 170 只猫、55 小时视频中整理出 117 种行为 | 论文公开 | 作为反应类别的定义依据 |

### 9.6 标签的多种来源：不只是人工标注

| 来源 | 怎么得到 | 成本 | 噪声 |
|---|---|---|---|
| **情境标签** | 自动记录时间戳：主人出门后 30 分钟、陪玩时段、门铃后 1 分钟、固定位置睡觉 | 几乎为零 | 中：情境不等于每一刻的状态 |
| **结果标签** | 机器人发起互动后 10 秒内，猫靠近还是走开（关键点和距离可以测量） | 几乎为零 | 低：直接观察到的结果 |
| **主人反馈** | app 里的“对 / 错”按钮 | 低 | 高，但量大，而且覆盖的正是这只猫 |
| **专家标注** | 行为学家或兽医看 100–200 个事件片段 | 高 | 最低，作为**测试集** |
| **AI 逐项评分** | 让视觉语言模型按 FGS 细则只评可见形态（耳朵 0/1/2…），不判断情绪 | 低 | 未知，必须先人工抽查 |

把这些来源组合起来有三种成熟的方法：
- **弱监督 / data programming**（Snorkel，Ratner et al., 2017）：多个“不太准”的来源各有各的错法，根据它们之间的一致与矛盾，在**没有标准答案**的情况下估计每个来源的可靠度，合成带概率的标签。
- **confident learning**（Northcutt et al., 2021）：找出标签很可能标错的样本。
- **无监督行为发现**：Keypoint-MoSeq（Weinreb et al., 2024）、B-SOiD（Hsu & Yttri, 2021）直接从关键点时间序列里找出反复出现的动作片段。人只需要给每一**类**起名字，而不必逐帧标注。输入正好是 demo 输出的特征 CSV。

### 9.7 闭环，以及三类规则各自的来源

```
① 实地数据 → ② 标签（情境 / 结果 / 反馈 / 专家）→ ③ 学习 → ④ 规则与策略 → ⑤ 部署 → ⑥ 效果评估 ─┐
     ▲                                                                                        │
     └────────────────────────────────────────────────────────────────────────────────────────┘
```

| 规则类型 | 例子 | 内容来自哪里 | 数据起什么作用 |
|---|---|---|---|
| **识别规则**：猫在做什么 | 耳尖高度 ≤ −1.5 且静止 → 僵住 | **数据**：从标签中学习，取代手写阈值 | 决定阈值和组合 |
| **互动规则**：机器人做什么 | 安全屏蔽；游戏以“抓住”结束 | **行为学**，固定，不允许学习修改 | 在安全范围内用 bandit 学习这只猫的偏好 |
| **养护建议**：主人做什么 | 在门口附近放一个藏身处 | **兽医指南**（Ellis et al., 2013），专家审核 | 决定**何时触发**，并**检验是否有效** |

**为什么养护建议不能直接从数据里学？** 数据只能告诉我们“哪些事情一起出现”，不能告诉我们“什么导致了什么”。“猫在主人出门后活动量下降”可能是焦虑，也可能只是在睡觉。直接从相关性生成建议，可能给出错误甚至有害的建议。

**如何检验建议有效：ABA 单个体设计**（动物行为研究常用的反转设计）：A 基线期 1–2 周 → B 采纳建议 1–2 周 → A 撤回 1 周。只有事件频率跟着 A → B → A 下降又回升，才能说明是建议起了作用。

### 9.8 部署到移动机器人：识别与分析策略

| 问题 | 为什么重要 | 策略 | 现在能否测试 |
|---|---|---|---|
| **摄像头自己在动** | 活动量会把“机器人在动”误认为“猫在动”（最严重） | 用猫以外区域的光流或轮速里程计估计自身运动，从关键点位移里减去 | ✅ 手机贴地边走边拍 |
| **机位低、距离多变** | 近距离时身体点会乱，远距离时脸太小 | **帧质量评分**（脸的像素大小、模糊度、置信度、姿态），只用好帧 | ✅ |
| **视角经常不正** | 姿态门控会频繁暂停脸部特征 | **主动感知**：脸部质量低时，机器人调整位置，让脸保持在 80–200 px 且正对镜头；猫回避时安全规则优先 | 需要机器人 |
| **算力有限**（UNO Q 约 2 FPS） | 跟不上追逐和扑击 | **两级速度**：每帧跑轻量检测和跟踪；关键点低频、只在好帧上跑；INT8 量化或 MobileNetV3-small | 部分 |
| **多猫家庭** | 基线必须对应到同一只猫 | 外观重识别（毛色特征 + 人工命名） | ✅ |
| **判断更碎片化** | 每帧能看清的时间更短 | 从逐帧判断改成**累积证据、判断事件**（例如 CUSUM 变化检测） | ✅ |
| **视角影响基线** | 同一只猫在不同距离、角度下特征不同 | 基线按视角分档（脸的大小、姿态），只和同一视角比较 | ✅ |
| **数据要能复用** | 以后训练和强化学习都依赖它 | 统一记录：时间、情境、机器人动作、特征序列、结果、画面质量 | ✅ |
| **安全** | 猫的福祉优先于数据 | 安全规则在机器人本地运行，不依赖网络 | 需要机器人 |

**机器人做出来之前**，可以把手机固定在一个贴地的小车上移动着拍猫，录 10–20 段作为“机器人视角测试集”，衡量检测召回率、可信脸部帧的比例，以及扣除自身运动之后的活动量是否正确。
""")

# =============================================================================
# 10. Disclosure
# =============================================================================
md(r"""
## 10. 版本演进的完整披露与局限

### 10.1 已经发现并修复的问题
| 问题 | 怎么发现的 | 修复 | 效果 |
|---|---|---|---|
| 检测器通道顺序 bug（RGB 当成 BGR） | 建立正式评估后，逐项排查召回率偏低的原因 | 统一使用 BGR | 召回率 85.6% → 90.2% |
| 身体模型全局回归丢失位置信息 | VOC 真实场景 PCK 只有 20.5% | 换成热力图架构 | +30.3 个点 |
| 身体训练数据太少 | 只有 180 张 | 加入 VOC，扩充到 1316 只猫 | +19.4 个点 |
| 脸框依赖身体模型的头部点 | 头部点错了，脸就跟着错 | 加入置信度门控 + 两遍脸部精修 | 粗框下 NME 1.58% → 1.19% |
| 脸部不能做翻转增强 | CatFLW 没有公布左右对应关系 | 用匈牙利匹配从平均脸自动推导 | 翻转增强与翻转 TTA 可用 |
| 实时帧率低 | 4–5 FPS | GPU 推理 + 隔帧检测 | Mac 约 30 FPS；UNO Q 约 550 ms/帧 |
| 受惊逃跑被推测成“想玩，高把握”（v4） | 逐条检查测试视频的推测 | 扑玩必须“不是在离开”；新增“快速跑开”线索（5.11） | 测试视频：高把握推测 6 → 2 个 |
| 蹲低被读成“离开”（v4） | 同一事件同时出现“离开”和“保持不动” | 保持不动时，面积变小不算离开 | 矛盾线索 3 → 0 次 |
| 一次反应被拆成多张卡片（v4） | 6 秒内 3 个“活跃”事件 | 2 秒内同一反应合并 | 事件 9 → 7 个 |
| 标签在阈值附近来回跳（v4） | 统计标签切换次数 | 进入 / 退出两个阈值（5.0） | 每分钟切换 41 → 35 次 |
| 只凭身体动作给出高把握（v4） | 卡片注明“脸看不清”却是“高” | 脸看不清时最多给“中”（5.10） | — |
| 手机串流越来越延迟（v4） | 模拟推流测试 | 只处理最新一帧，同时录下原始画面 | 每秒约 29 帧，无累积延迟 |
| 一路视频坏掉，整次测试的报告失败（v4） | 用一个损坏的文件测试 | 跳过并在报告里提醒 | 其余视频照常生成报告 |
| 卡片写的依据和实际判断不一致（v4.1） | 卡片标题“耳朵压低”，依据里只有“身体静下来了” | 记录判断真正依据的条件和数值；卡片和红圈只写这些；每个事件自动做反事实检验（6.14） | 测试视频 7/7 通过；旧版卡片在 3/7 个事件里漏写了主要依据 |

### 10.2 仍然存在的局限
1. 没有独立测试集；报告的数字来自用于挑选最佳轮次的验证集，略偏乐观。
2. 验证图是照片，不是家里摄像头的画面。
3. 腿和爪的精度仍然偏低（64–71%）。
4. 机器人端只假设画面里有一只猫，没有 NMS 和多目标跟踪。
5. 机器人上为了速度关闭了翻转 TTA，检测器用的是 YOLOv8n、输入 320。这个配置还没有在完整验证集上单独评估；第 1.3 节的 12 张抽查里召回率只有 75%，低于 Mac 端配置。
6. 情绪状态的定义、特征与状态之间的关系都还没有用真实数据验证（第 5、6 节是 🧪）。
7. 模拟猫的参数全部是假设（第 8 节是 🧪）。
8. 缺少尾巴、胡须、瞳孔和声音这几类信号。
9. 头部姿态门控只能拦住大幅度的抬头和转头；小角度仍会让耳尖高度产生偏差（4.11）。俯仰修正（4.12）只在平面近似下成立，还没有在真实的抬头片段上检验。
10. 第一层反应分类只在一段约 35 秒的网络视频上运行过，没有人工标注的对照。
11. 身体特征（头部下垂、活动量）的测量误差下限是经验值，没有像脸部特征那样在验证集上测量过。
12. **结果对校准非常敏感（v4 实测）。** 同一段测试视频只改校准长度：60 帧时没有任何“耳朵后压 / 僵住”；90 帧时“僵住”占 22%；300 帧时“耳朵后压”占 10%。所以测试前应该用一段猫平静的视频单独校准并保存基线，而不是依赖视频开头。
13. 5.10–5.12 的概率、阈值和线索定义都是假设，**没有任何一条经过人工标注检验**。v4 的修复让输出更一致、减少了明显的错误，但**不能据此说准确度提高了多少**。

### 10.3 现在能说什么，实地测试后能说什么

| 研究问题 | 现在 | 做出机器人并实地测试之后 | 那时仍然缺什么 |
|---|---|---|---|
| 可视化解释能否帮主人**发现系统错误** | ✅ 可以用录好的片段做实验 | ✅✅ 在主人自己家里测试，结论更有说服力 | — |
| 主人的信任是否与系统的可靠度相符 | ✅ 短期实验 | ✅✅ 可以观察几周内的变化 | 仍需抽样标注每次判断的对错 |
| 解释是否忠实于模型 | ✅ 已做（遮挡分析、删除测试） | ✅ 在机器人视角上重新验证 | — |
| 机器人行为的解释有没有用 | ❌ | ✅ | — |
| 识别准确度 | ⚠️ 只有感知层验证过 | ✅ 有了情境、结果和专家标签，可以训练并检验 | 只对少数几只猫有效 |
| 猫会参与还是回避 | ❌ | ✅ 结果标签自动产生 | 新奇感会消退，需要几周 |
| 情绪解释是否正确 | ❌ | ⚠️ 只能说“和专家、情境一致” | 情绪本身看不见 |
| 养护建议是否有效 | ❌ | ⚠️ 用 ABA 设计做少数猫的初步检验 | 更多的猫、更长的时间 |
| 能否改善猫的福祉 | ❌ | ⚠️ 只能看到初步迹象 | 兽医评估、对照组、长周期 |

做出机器人并实地测试之后，XAI 研究可以从“实验室里看片段”升级到**真实使用场景中的评估**，也就是 Doshi-Velez & Kim (2017) 所说的最高层次（application-grounded）。

### 10.4 实地测试的最低要求

| 要求 | 最低标准 | 原因 |
|---|---|---|
| 猫的数量 | 3–5 只，来自不同家庭 | 1 只猫无法区分“方法有效”和“这只猫刚好这样” |
| 时长 | 每只猫至少 4 周 | 前一两周主要是新奇感 |
| 对照 | ABA：有机器人 / 无机器人 / 再有机器人；另加一段“机器人在场但不互动” | 否则无法说明变化来自机器人（包括观察者效应） |
| 标准答案 | 每只猫 50–100 个片段，两人独立标注并计算一致性，最好有专家参与 | 没有它，任何准确度数字都不成立 |
| 记录 | 统一格式（见 9.8） | 事后分析和再训练都依赖它 |
| 安全 | 安全规则在本地运行；主人可随时远程停止；每次屏蔽都记录 | 猫的福祉优先 |
| 伦理 | 学校的动物伦理和人类受试者伦理审批；主人知情同意；视频只在本地处理 | 涉及动物和人的研究都需要 |

### 10.5 对本项目的四条批评与回应

**批评 1：证据太薄。** 情绪推测的概率是手设的，没有用人工标注的数据验证。
- **事实**：成立。第一层规则只在一段约 35 秒的网络视频和合成数据上运行过；5.10–5.12 全是假设或模拟。
- **因此能说的**：这个项目展示的是**在没有验证的情况下怎样负责任地做**——证据等级标在每一节、推测写成“看起来像”并给出把握和另一种可能、把握低时不给情绪、主人的反馈能变成标签。它**不能**展示“这个方法被证明有效”。
- **怎样改变这一点**：30–50 个片段，两人独立标注，先算人和人的一致性（上限），再算系统和人的一致性（10.4）。

**批评 2：评估只覆盖了技术层。** 课程站点把评估分成三层：**技术层**（归因、稳定性、运行时间）、**理解层**（人能否解释新案例、预测系统下一步）、**协作层**（任务结果、错误发现、采纳）（Doshi-Velez & Kim, 2017；Rong et al., 2024）。
- **事实**：本项目做了技术层（遮挡分析、删除测试、稳定性，6.4–6.10；**以及 6.14 对部署系统本身的解释检验，测试视频 7/7 通过**）和表达设计（报告与 app 的措辞、把握、另一种可能），**“主人读懂报告了吗”这一层没有人测过**，计划在 9.4，但还没做。
- **如果只做一组测试，意味着什么**：一组 = 一位主人、一只猫、一次测试，样本量为 1。它是**形成性研究 / 可行性检查**：能发现主人在哪里看不懂、哪里不信，能检查流程是否跑得通；**不能**推广到其他主人，也不能做“有解释 vs 没有解释”的组间比较。
- **一组也能得到的、有用的证据**：(1) 出声思考：主人边看报告边说出判断；(2) **预测任务**：先只给视频，让主人写下判断，再显示系统的推测和依据，看他们是否改变判断、为什么；(3) **故意放入 2–3 个系统推测错误的片段**，记录主人能否发现。结果按次数和原话报告，不做显著性检验。

**批评 3：模型类型偏窄。** 站点的 6 个方法（可解释模型、TreeInterpreter、SHAP / TreeSHAP、LIME、Grad-CAM、反事实 / DiCE）大多面向表格数据和分类；本项目解释的是**关键点回归**。
- **事实（第 6 节实际做到的）**：6.4 的遮挡敏感度确实在**真实的关键点回归模型**上运行，但只做了 **1 张图、1 个关键点（左耳）**，结论是“敏感区域在脸上、不在背景上，也不在耳朵本身”——这是一次健全性检查，不是系统的评估。6.6–6.10 的 Shapley、对比解释、反事实、删除测试都在**合成的表格模型**上，不在部署的系统里。
- **因此建议**：(a) 部署系统本身更准确的定位是站点里的**第一类：可解释模型 / 设计即可解释**（关键点 → 几何特征 → 规则 → 朴素贝叶斯，每一步可打印），反事实可以自然地写成“如果耳朵回到平常的位置……”；(b) 不要声称对部署系统做了 SHAP 或 LIME；(c) 如果要把“关键点回归的解释”作为贡献，至少要把遮挡分析扩展成小规模评估（例如数据包里的 24 张图、耳朵和眼睛的关键点、敏感度落在头部 vs 背景的比例），否则只放在附录里作为健全性检查。

**批评 4：面向受众偏窄。** 站点读者是工业产品设计和 HCI，本项目只有动物行为这一个领域。
- **回应：把可迁移的部分说出来。** 这个项目的核心问题是“**替一个不会说话的对象，向第三方解释它的状态**”，同样出现在婴儿监护、老人照护、病人监测，甚至设备状态监测里。可迁移的设计模式：
  1. 和**它自己的平常状态**比，而不是和“平均的个体”比；
  2. 结论旁边放**看得见的依据**（视频、圈出的部位），让读者自己核对；
  3. 把握不够时**不下结论**，并给出**另一种可能和怎么分辨**；
  4. 读者的纠正**变成标签**，让系统随使用改进；
  5. 具身系统里，**机器人的动作也被记录和解释**（之前发生了什么，只说先后、不说因果）。
- **仍然成立的局限**：这些模式只在一个领域、一只猫的视频上演示过；在其他领域是否同样有效，需要另外的研究。
""")

md(r"""
## 11. 参考文献

标注 ⚠️ 的条目是根据记忆整理的，引用前请核对卷号、页码和 DOI。

**数据集与感知模型**
- Cao, J., Tang, H., Fang, H.-S., Shen, X., Lu, C., & Tai, Y.-W. (2019). Cross-domain adaptation for animal pose estimation. *ICCV*, 9498–9507.
- Casiez, G., Roussel, N., & Vogel, D. (2012). 1€ filter: A simple speed-based low-pass filter for noisy input in interactive systems. *CHI*.
- Howard, A., et al. (2019). Searching for MobileNetV3. *ICCV*. arXiv:1905.02244
- Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics
- Lin, T.-Y., et al. (2014). Microsoft COCO: Common objects in context. *ECCV*. arXiv:1405.0312
- Lin, T.-Y., et al. (2017). Feature pyramid networks for object detection. *CVPR*. arXiv:1612.03144
- Martvel, G., Farhat, N., Shimshoni, I., & Zamansky, A. (2023). CatFLW: Cat facial landmarks in the wild dataset. arXiv:2305.04232
- Martvel, G., Shimshoni, I., & Zamansky, A. (2024). Automated detection of cat facial landmarks. *International Journal of Computer Vision*.
- Nibali, A., He, Z., Morgan, S., & Prendergast, L. (2018). Numerical coordinate regression with convolutional neural networks. arXiv:1801.07372
- Sun, X., Xiao, B., Wei, F., Liang, S., & Wei, Y. (2018). Integral human pose regression. *ECCV*. arXiv:1711.08229

**猫的面部表情、情绪与福祉**
- Bennett, V., Gourkow, N., & Mills, D. S. (2017). Facial correlates of emotional behaviour in the domestic cat (*Felis catus*). *Behavioural Processes, 141*, 342–350. https://doi.org/10.1016/j.beproc.2017.03.011
- Cats Protection. (n.d.). *Cat body language*. Retrieved October 6, 2026, from https://www.cats.org.uk/help-and-advice/cat-behaviour/cat-body-language（面向主人的科普，没有引用研究，只作措辞参考）
- Ellis, S. L. H., et al. (2013). AAFP and ISFM feline environmental needs guidelines. *Journal of Feline Medicine and Surgery, 15*(3), 219–230.
- Evangelista, M. C., et al. (2019). Facial expressions of pain in cats: The development and validation of a Feline Grimace Scale. *Scientific Reports, 9*, 19128.
- Feighelstein, M., et al. (2022). Automated recognition of pain in cats. *Scientific Reports, 12*, 9575.
- Finka, L. R., et al. (2019). Geometric morphometrics for the study of facial expressions in non-human animals, using the domestic cat as an exemplar. *Scientific Reports, 9*, 9883.
- ⚠️ Humphrey, T., Proops, L., Forman, J., Spooner, R., & McComb, K. (2020). The role of cat eye narrowing movements in cat–human communication. *Scientific Reports, 10*, 16503.
- Schwartz, S. (2002). Separation anxiety syndrome in cats: 136 cases (1991–2000). *JAVMA, 220*(7), 1028–1033.
- Tu, C., Wu, J., Huang, Z., & Xie, J. (2026). PMTNet: A part-centric missing-aware temporal network for cat behavior recognition in unconstrained videos. *Animals, 16*(11), 1589.

**捕猎、玩耍与交互**
- ⚠️ Hall, S. L., & Bradshaw, J. W. S. (1998). The influence of hunger on object play by adult domestic cats. *Applied Animal Behaviour Science, 58*, 143–150.
- ⚠️ Hall, S. L., Bradshaw, J. W. S., & Robinson, I. H. (2002). Object play in adult domestic cats: The roles of habituation and disinhibition. *Applied Animal Behaviour Science, 79*, 263–271.
- ⚠️ Kogan, L. R., & Grigg, E. K. (2021). Laser light pointers for use in companion cat play: Association with guardian-reported abnormal repetitive behaviors. *Animals, 11*(8), 2178.
- Leyhausen, P. (1979). *Cat behavior: The predatory and social behavior of domestic and wild cats*. Garland STPM Press.
- Mancini, C. (2011). Animal-computer interaction: A manifesto. *Interactions, 18*(4), 69–73.
- West, M. J. (1977). Exploration and play with objects in domestic kittens. *Developmental Psychobiology, 10*(1), 53–57.

**机器学习、XAI 与信任**
- Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). *Classification and regression trees*. Wadsworth.
- Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. arXiv:1702.08608
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics, 29*(5), 1189–1232.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
- ⚠️ Hoffman, R. R., Mueller, S. T., Klein, G., & Litman, J. (2018). Metrics for explainable AI: Challenges and prospects. arXiv:1812.04608
- Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson, E., Kim, B., & Liang, P. (2020). Concept bottleneck models. *ICML*.
- Lee, J. D., & See, K. A. (2004). Trust in automation: Designing for appropriate reliance. *Human Factors, 46*(1), 50–80.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Rong, Y., Leemann, T., Nguyen, T.-T., Fiedler, L., Qian, P., Unhelkar, V., Seidel, T., Kasneci, G., & Kasneci, E. (2024). Towards human-centered explainable AI: A survey of user studies for model explanations. *IEEE TPAMI, 46*(4), 2104–2122. https://doi.org/10.1109/TPAMI.2023.3331846
- Miller, T. (2019). Explanation in artificial intelligence: Insights from the social sciences. *Artificial Intelligence, 267*, 1–38.
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE, 77*(2), 257–286.
- Shapley, L. S. (1953). A value for n-person games. In *Contributions to the Theory of Games II*, 307–317.
- Wachter, S., Mittelstadt, B., & Russell, C. (2017). Counterfactual explanations without opening the black box. *Harvard Journal of Law & Technology, 31*(2).
- Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. *ECCV*.

**强化学习与数据**
- ⚠️ Alshiekh, M., et al. (2018). Safe reinforcement learning via shielding. *AAAI*.
- Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika, 39*, 324–345.
- Christiano, P., et al. (2017). Deep reinforcement learning from human preferences. *NeurIPS*.
- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement, 20*(1), 37–46.
- Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018). A tutorial on Thompson sampling. *Foundations and Trends in Machine Learning, 11*(1), 1–96.
- Settles, B. (2009). *Active learning literature survey* (Tech. Rep. 1648). University of Wisconsin–Madison.
- Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. *Biometrika, 25*(3/4), 285–294.

**标签、弱监督与行为发现**
- ⚠️ Hsu, A. I., & Yttri, E. A. (2021). B-SOiD, an open-source unsupervised algorithm for identification and fast prediction of behaviors. *Nature Communications, 12*, 5188.
- ⚠️ Northcutt, C. G., Jiang, L., & Chuang, I. L. (2021). Confident learning: Estimating uncertainty in dataset labels. *Journal of Artificial Intelligence Research, 70*, 1373–1411.
- ⚠️ Ratner, A., et al. (2017). Snorkel: Rapid training data creation with weak supervision. *Proceedings of the VLDB Endowment, 11*(3), 269–282.
- ⚠️ Weinreb, C., et al. (2024). Keypoint-MoSeq: Parsing behavior by linking point tracking to pose dynamics. *Nature Methods, 21*, 1329–1339.

**可参考的猫数据集**
- KAITANG2003. (2025). *cat-emotion-dataset* [Data set]. Hugging Face. https://huggingface.co/datasets/KAITANG2003/cat-emotion-dataset（审查结果见 9.5）
- Ludovico, L. A., Ntalampiras, S., Presti, G., Cannas, S., Battini, M., & Mattiello, S. (2021). CatMeows: A publicly-available dataset of cat vocalizations. *MultiMedia Modeling (MMM 2021)*. https://zenodo.org/records/4008285
- Martvel, G., et al. (2024). Automated video-based pain recognition in cats using facial landmarks. *Scientific Reports*. https://www.nature.com/articles/s41598-024-78406-2
- ⚠️ Steagall, P. V., et al. (2023). Fully automated deep learning models with smartphone applicability for prediction of pain using the Feline Grimace Scale. *Scientific Reports, 13*. https://www.nature.com/articles/s41598-023-49031-2
- ⚠️ Ethogram of the domestic cat (2024). https://www.mdpi.com/2813-9372/1/3/21（作者与期刊信息请核对）
""")


def build(path: Path) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [],
    }
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"c{i:03d}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(f"wrote {path} ({len(CELLS)} cells)")


if __name__ == "__main__":
    build(Path(__file__).resolve().parent / "CatPose_XAI_Tutorial.ipynb")
