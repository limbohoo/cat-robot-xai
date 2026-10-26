# 小车数据对接说明

本说明讲清楚测试时要做三件事：

- 组员怎么遥控小车；
- 小车的每个动作怎么记录下来；
- 这些记录怎么和视频对上时间，最后进到主人报告里。

采用的是方案 A：组员通过「网桥」页面遥控小车。网桥记录每一条命令，再把命令转发给小车。

---

## 1. 整体结构

```
组员手机浏览器 ──按钮──▶ 网桥（测试用笔记本，robot_bridge.py）──HTTP 命令──▶ 小车 Wi‑Fi 板（ESP32）
                               │
                               ▼
                       robot_log.csv（用笔记本的时钟）
                               │
   视频（房间摄像头、小车摄像头）+ session.json + robot_log.csv
                               │
                               ▼
                  python -m catpose.session  →  主人报告 report.html
```

- 所有时间都以**笔记本的时钟**为准，小车板子的时钟不用管。
- 笔记本时钟和摄像头时钟之间的差，用**拍一次手**来对齐（见第 6 节）。
- 小车、笔记本、组员手机必须连**同一个 Wi‑Fi**，可以用一部手机开热点。

| 文件 | 作用 |
|---|---|
| `ml/catpose/robot_bridge.py` | 网桥：遥控页面、记录日志、转发命令、远程停止 |
| `hardware/car-bridge/car_server_example.ino` | 方案 A：小车接收命令的示例固件 |
| `hardware/car-bridge/car_event_example.ino` | 方案 B（备用）：保留原遥控、由小车上报动作 |
| `ml/catpose/session.py` | 读取视频和小车日志，生成主人报告 |
| `ml/catpose/realtime.py` | 实时分析：电脑摄像头、iPhone 或手机推流（见第 11 节） |

---

## 2. 命令约定（网桥和小车必须一致）

网桥发给小车的地址格式：`http://<小车IP>/cmd?c=<命令>`

| 命令 | 页面按钮 | 小车应做的事 |
|---|---|---|
| `ping` | （网桥启动时自动发） | 不动，只回复 `ok` |
| `slow_prey` | 慢速逗猫（逃跑的猎物） | 慢速之字形、远离猫的方向移动 |
| `light` | 灯光 | 地面灯光图案 |
| `sound` | 声音 | 短促、小音量的声音 |
| `approach` | 慢慢靠近 | 低速前进 |
| `retreat` | 后退离开 | 低速后退 |
| `round_end` | 本轮结束 | 全部停止 |
| `stop` | 红色「停 止」 | **立即全部停止** |

- 小车收到命令后返回 HTTP 200（内容随意，例如 `ok slow_prey`）。
- 不认识的命令返回 400。
- 如果要加新动作，网桥和小车两边都要改：网桥改 `robot_bridge.py` 里的 `ACTIONS` 列表，小车在 `handleCmd()` 里加一个分支。命令名用英文小写加下划线，**不要包含 `stop` 或 `sync`**，因为这两个词有特殊含义。

---

## 3. 小车端（负责小车的组员）

以 `car_server_example.ino` 为起点：

1. 填 `WIFI_SSID` 和 `WIFI_PASS`，填测试当天用的热点。
2. 把这几个空函数换成你们自己的代码：
   - `stopMotors()` 和 `lightOff()`：停止。**最重要，先写这两个。**
   - `doSlowPrey()`、`doLight()`、`doSound()`、`doApproach()`、`doRetreat()`。
3. 如果某个动作需要持续控制（例如之字形要一步一步走），写在 `loop()` 里 `busy` 为真时执行的部分，**不要在函数里用很长的 `delay()`**。否则在动作执行期间，小车收不到停止命令。
4. 烧录后打开串口监视器（波特率 115200），会显示 `car ip: 192.168.x.x`。把这个 IP 告诉负责网桥的组员。

示例固件里已有三道保险，请保留：

- 收到 `stop`、`safety_stop` 或 `round_end` 时立即停止；
- 每个动作最多运行 `ACTION_MAX_MS`（默认 8 秒），超时自动停止；
- Wi‑Fi 断开时自动停止。

自测方法：小车和电脑连同一个 Wi‑Fi，在电脑浏览器里输入 `http://<小车IP>/cmd?c=slow_prey`，小车应该动起来；再输入 `http://<小车IP>/cmd?c=stop`，小车应该停下。

---

## 4. 网桥端（负责笔记本的组员）

网桥只需要 Python 3，不用安装额外的包。在仓库的 `ml/` 目录下运行：

```bash
python -m catpose.robot_bridge sessions/test1 --car-url "http://192.168.x.x/cmd?c={cmd}"
```

- `sessions/test1`：这次测试的文件夹，`robot_log.csv` 会写在这里。
- `--car-url`：把 `192.168.x.x` 换成小车的 IP，`{cmd}` 保持原样。
- `--port`：可选，页面端口，默认 8800。

启动后终端显示：

```
control page: http://192.168.0.23:8800   log: sessions/test1/robot_log.csv
car: reachable
```

- `car: reachable`：表示连上了小车。
- `car: NOT reachable (...)`：检查小车是否开机、IP 是否正确、是否连在同一个热点。

用手机浏览器打开 `control page` 后面的地址，就是遥控页面：

- **红色「停 止」**：任何时候都可以按。网桥会记录 `safety_stop`，并给小车发 `stop`。
- **动作按钮**：按一次就记录一次，并转发给小车。
- **备注栏**：按按钮前先填，备注会和这次动作写进同一行。例如「第 1 轮」「猫躲到沙发下」。
- **同步按钮**：只做记录，不发给小车。见第 6 节。
- 页面底部显示最近 12 条记录。小车没有回应时，会显示红字「小车没有回应」。

在终端按 `Ctrl+C` 结束网桥。

---

## 5. 日志格式 `robot_log.csv`

每写一行就立即存盘，中途断电也不会丢。

```
time,event,detail,source
2026-10-20T14:00:02.115+02:00,bridge_start,,bridge
2026-10-20T14:00:30.402+02:00,sync_clap,,page
2026-10-20T14:15:00.881+02:00,slow_prey,第 1 轮,page
2026-10-20T14:15:41.020+02:00,round_end,,page
2026-10-20T14:30:12.530+02:00,safety_stop,猫耳朵压平,page
```

| 字段 | 含义 |
|---|---|
| `time` | 笔记本时钟，带时区，精确到毫秒 |
| `event` | 命令名，或下表里的标记 |
| `detail` | 备注 |
| `source` | `page` 表示遥控页面，`car` 表示小车上报，`bridge` 表示网桥自己写的 |

报告程序这样理解这些记录：

| event | 在报告里 |
|---|---|
| 含 `stop` | 安全停止，会单独列出 |
| `sync_clap` | 对时标记，不算动作 |
| `bridge_start` / `bridge_end` / `round_end` | 标记，不算动作 |
| `car_unreachable` | 记入报告的测试记录（可折叠部分）和 `sync.json`：「小车没有回应命令」 |
| 其他（如 `slow_prey`） | 小车动作。报告会统计动作之后 10 秒内猫的反应：参与、回避或没反应 |

如果网桥出了问题，也可以手写这个文件，只需要 `time,event,detail` 三列。

---

## 6. 时间对齐：拍一次手

1. 所有摄像头**先开始录像**，再启动网桥。
2. 在遥控页面按「同步：按下后马上拍一次手」，**立刻**在所有摄像头都能看到、听到的地方用力拍一次手。
3. 测试结束后，用播放器打开房间摄像头的视频，找到拍手那一下是第几秒（例如 12.3 秒）。
4. 填进 `session.json`：

```json
{
 "name": "用户测试 1 · 主人不在家",
 "videos": [
  {"path": "room.mp4", "name": "房间摄像头"},
  {"path": "robot_cam.mp4", "name": "小车摄像头", "moving": true}
 ],
 "robot_log": "robot_log.csv",
 "clap": {"video": "room.mp4", "s": 12.3}
}
```

装在小车上的摄像头要标 `"moving": true`：它的画面跟着小车动，程序就不会用猫在画面里的大小判断“靠近 / 离开”，小车每轮的结果也只看固定摄像头。

报告程序会平移整份小车日志，让日志里的拍手时间和视频里的拍手对齐。平移了多少秒会写在 `sync.json` 和报告的测试记录（可折叠部分）里。

不同摄像头之间的时间，程序会按声音自动对齐，不用手动处理。拍手的声音也能帮助这一步。

如果忘了拍手，程序会改用视频文件里的录制时间。这种情况下可能有几秒误差，报告里会提示。

---

## 7. 测试当天流程

**测试前（不放猫）**
1. 开热点，把小车、笔记本、手机都连上去。
2. 启动网桥，确认终端显示 `car: reachable`。
3. 完成第 9 节的安全检查。
4. 删除测试用的 `robot_log.csv`，或者换一个新的 session 文件夹。

**测试中**
1. 开始所有摄像头录像。
2. 启动网桥，按「同步」，然后拍手。
3. 校准期：猫进入画面后，程序先用约 300 帧（30 fps 时约 10 秒）记录猫平时的样子作为基准。为了留出余量，**开始后至少 1 分钟内小车不要动**。如果校准期间小车动了，报告会提醒。
4. 按测试计划进行每一轮：先在备注栏写「第 N 轮」，再按动作按钮，结束时按「本轮结束」。
5. 发现猫耳朵压平、躲藏、哈气，或者小车卡住，**直接按停止**，不用犹豫。
6. 测试结束：先在网桥终端按 `Ctrl+C`，再停止录像。

**测试后**
1. 把所有视频复制进 session 文件夹，和 `robot_log.csv` 放在一起。
2. 在 `session.json` 里填写 `clap`。
3. 先检查时间对齐：
   ```bash
   python -m catpose.session sessions/test1 --dry-run
   ```
   查看 `report/sync.json` 里每个视频的偏移和提醒。
4. 生成报告：
   ```bash
   python -m catpose.session sessions/test1
   ```

---

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| `car: NOT reachable` | 小车没开机、IP 写错，或不在同一个热点。在浏览器里试 `http://<小车IP>/cmd?c=ping` |
| 手机打不开遥控页面 | 手机和笔记本不在同一个 Wi‑Fi；或者 Mac 防火墙拦住了 Python，在弹窗里允许即可 |
| 小车 IP 每次都变 | 在热点设置里给小车固定 IP，或每次看串口输出 |
| 按了按钮，小车反应慢 | 动作函数里有长 `delay()`，改成在 `loop()` 里一步一步执行 |
| 报告提醒「小车动作不在任何视频时间内」 | 摄像头开得晚、关得早，或 `clap` 填错 |
| 报告提醒「无法换算时间」 | 手写日志的时间格式不对。用 `2026-10-20T14:15:00` 这种格式 |

---

## 9. 安全检查（每次测试前，不放猫）

- [ ] 小车运动时按「停 止」，0.5 秒内停下（试 3 次）
- [ ] 小车运动时关掉热点，小车自动停下
- [ ] 每个动作在 8 秒内会自己结束
- [ ] 声音音量小、灯光不直射眼睛
- [ ] 至少一名组员全程远程看画面，手机上随时能按停止

---

## 10. 备用方案 B：保留原来的遥控

如果小车不方便改成接收 HTTP 命令，可以继续用你们原来的遥控方式，只在固件里加两件事。示例见 `car_event_example.ino`：

- **上报动作**：每执行一个动作，调用一次 `reportEvent("slow_prey", "round1")`。网桥收到后，用笔记本的时钟记录。
- **检查停止**：在 `loop()` 里调用 `checkStop()`，每 0.3 秒查询网桥的 `/status`。有人在遥控页面按了停止，小车就停下。

启动网桥时去掉 `--car-url`，网桥就只记录、不转发：

```bash
python -m catpose.robot_bridge sessions/test1
```

---

## 11. 用手机摄像头实时分析

用于测试前调试，或现场看识别效果。以下命令都在 `ml/` 目录下运行。

**1. 查看摄像头编号**
```bash
.venv/bin/python -m catpose.realtime --list-cameras
```
iPhone 和 Mac 登录同一个 Apple ID 时，iPhone 会出现在列表里（连续互通相机），不需要装 App。

**2. iPhone（连续互通相机）**：手机靠近 Mac，锁屏，横放。`--source` 填列表里 iPhone 的编号。
```bash
.venv/bin/python -m catpose.realtime --source 1 --detect --report-dir sessions/live1
```

**3. 安卓或其他手机（Wi‑Fi 推流）**：装一个 IP 摄像头 App（例如 IP Webcam），手机和电脑连同一个 Wi‑Fi，把 App 显示的视频地址填进去。
```bash
.venv/bin/python -m catpose.realtime --source http://手机IP:8080/video --detect --report-dir sessions/live1
```

**常用参数**

| 参数 | 作用 |
|---|---|
| `--detect` | **必须加**，打开猫检测，否则不会识别 |
| `--report-dir 文件夹` | 结束时生成主人报告和事件片段 |
| `--calib-frames 300` | 开始时用多少帧学习猫平时的样子，默认 300（约 10 秒） |
| `--output out.mp4` | 另外保存带标注的视频 |
| `--record raw.mp4` | 保存原始画面；给了 `--report-dir` 时默认存为其中的 `raw.mp4`，之后可以当作视频文件重新分析 |
| `--moving-camera` | 摄像头装在小车上时加，不再用画面大小判断靠近 / 离开 |
| `--baseline-out` / `--baseline-in` | 存下 / 读取这只猫平静时的基线（见下） |
| `--no-display` | 不弹出窗口 |

按 `q` 结束，报告在 `--report-dir` 文件夹里的 `report.html`。

**先给猫单独校准一次（推荐）**：结果对基线很敏感，同一段视频只改校准长度，标签分布就会明显不同。找一段猫平静、脸朝镜头的时间：
```bash
.venv/bin/python -m catpose.realtime --source 1 --detect --baseline-out cats/kiki.json
```
看到 baseline saved 后按 `q`。之后每次都读取它，不再在开头校准：
```bash
.venv/bin/python -m catpose.realtime --source 1 --detect --baseline-in cats/kiki.json --report-dir sessions/live1
```

**注意**
- 没有读取基线时，开始的几秒是校准期，这时猫应该在画面里，并且状态平静。
- 程序只处理最新的一帧，电脑慢的时候会跳帧，但画面不会越来越延迟。
- 正式测试仍然建议：手机或摄像头**录像**，同时运行网桥记录小车，测试后用第 7 节的 `catpose.session` 生成报告。实时模式还不读取网桥的小车日志，所以实时报告里没有「小车邀请」的统计。
- 打不开推流地址时，先在电脑浏览器里打开这个地址，看能不能看到画面。
