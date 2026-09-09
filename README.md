# NezhaStand：第一阶段基础站立

本项目以桌面版 Nezha URDF 为机器人资产，并在仓库内自带站立训练所需的
`legged_gym` 基础环境、任务注册与参数工具，以及 `rsl_rl` 的 Actor-Critic、
PPO、rollout storage、runner、日志和 checkpoint 组件。运行时不再依赖
LZHMine 源码或 `LZHMINE_ROOT`。NVIDIA Isaac Gym 和带 CUDA 的 PyTorch 因
平台及授权原因仍需在训练机中单独安装。

第一阶段使用普通非对称 PPO，不使用 MINE、历史编码器或 VAE。

```text
NezhaStand/
├── assets/nezha/       # 桌面版 URDF 与全部 STL
├── legged_gym/         # 仿真环境基类、配置、地形与任务注册
├── rsl_rl/             # Actor-Critic、PPO、storage 与 runner
├── nezha_stand/        # Nezha 站立任务配置、环境和奖励
└── scripts/            # 安装检查、资产检查、训练与评估入口
```

## 第一阶段训练结构

- Actor 输入 46 维当前本体观测，网络为 `46 -> 256 -> 128 -> 64 -> 12`；
- 策略只输出 12 个腿关节位置残差，四个轮关节使用零速度阻尼控制；
- Critic 输入 64 维无噪声特权观测，包含真实基座速度、高度、足端接触力和漂移；
- 仿真步长 5 ms，控制周期 20 ms，目标站立高度 0.50 m；
- 保留目标高度奖励，另对四腿对应关节的归一化力矩幅值平方差施加惩罚；
- `default_dof_pos` 采用桌面 `config.yaml` 的 FL/FR/RL/RR 定义，但不设置初始姿态奖励；
- 平地、固定摩擦、无推搡、无载荷/质心/电机随机化；
- 初始腿关节仅加入 +/-0.02 rad 扰动。

配置见 `nezha_stand/config.py`，环境和奖励见 `nezha_stand/env.py`。

## 1. 创建独立环境

必须在带 NVIDIA GPU 的 Linux 训练机上使用 Isaac Gym Preview 4。项目固定
使用 Python 3.8；假设 Isaac Gym 已解压到 `/path/to/isaacgym`：

```bash
cd /path/to/NezhaStand
uv venv --python 3.8 --seed .venv
source .venv/bin/activate

uv pip install "numpy==1.23.5" "setuptools<70" wheel ninja
uv pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 \
  torchaudio==0.10.0+cu113 \
  --find-links https://download.pytorch.org/whl/cu113/torch_stable.html
uv pip install --no-build-isolation -e /path/to/isaacgym/python
uv pip install --no-build-isolation -e .
```

确认导入的训练组件都来自本仓库：

```bash
python scripts/check_install.py
```

脚本会同时检查 Isaac Gym、CUDA、URDF，以及 `legged_gym`、`rsl_rl`、
`nezha_stand` 的实际导入路径。后三条路径都应位于 `/path/to/NezhaStand`，
不应出现 LZHMine。

## 2. 静态检查

```bash
cd /path/to/NezhaStand
source .venv/bin/activate
python3 scripts/validate_assets.py
python3 scripts/validate_phase1.py
```

若训练机上另有原始 `nezha` 目录，可用
`python3 scripts/validate_assets.py --source /path/to/nezha` 做逐文件校验。

## 3. GPU 物理检查

先用四个环境和可视化窗口检查原始碰撞网格：

```bash
python scripts/check_physics.py --num_envs 4 --duration 5
```

观察机器人是否以约 0.54 m 高度出生，轻微下落后四轮触地，是否存在穿透、
弹飞、腿部碰撞误触发或高频抖动。脚本施加严格的零策略动作；物理检查不通过
时先修正碰撞体、出生高度或 PD 参数。

## 4. 正式训练

```bash
python scripts/train.py --headless --num_envs 2048 --max_iterations 5000
```

建议先观察 1000 次迭代的曲线。若存活率没有持续提高，依次检查终止原因、
接触状态、动作是否饱和和各奖励分量，不要同时改多组参数。

## 5. 阶段验收

默认评估最新运行中的最新 `model_*.pt`：

```bash
python scripts/evaluate.py --headless --num_envs 256 --episodes 1024
```

也可指定模型：

```bash
python scripts/evaluate.py --headless --checkpoint_path /path/to/model_5000.pt \
  --num_envs 256 --episodes 1024
```

评估排除每个 episode 开头 1 秒的沉降数据，但沉降期间倒地仍计为失败。
全部指标通过时输出 `PROMOTE_TO_PHASE_2=YES`：

| 指标 | 第一阶段门槛 |
|---|---:|
| 10 秒站立成功率 | >= 95% |
| Roll/Pitch RMS | <= 2 deg |
| 高度 RMSE | <= 0.02 m |
| XY 速度 RMS | <= 0.05 m/s |
| 力矩饱和比例 | <= 1% |
| 四轮同时接触比例 | >= 95% |
| 最大 XY 漂移 | <= 0.10 m |
| 轮速 RMS | <= 0.20 rad/s |
| 四腿对应关节力矩差 RMS | <= 10 N·m |

高度奖励不能在基础站立阶段直接删除。它与直立、四足接触奖励一起阻止机器人
通过蹲低或塌陷来获得“静止”分数。新增的 `torque_balance` 奖励比较四条腿上
同类型关节的力矩幅值（hip 对 hip、thigh 对 thigh、calf 对 calf），并先除以
各关节力矩上限再计算六组腿对的平方差。使用幅值是因为左右髋关节在镜像坐标系
中可能需要符号相反但物理效果对称的力矩；归一化则避免力矩上限较大的关节主导
奖励。该项并不能单独保证合理站立，因此仍保留机身直立、高度、接触、漂移和总力矩
约束。

为避免偶然性，正式进入第二阶段前应使用至少三个不同随机种子训练；至少两个
种子的 checkpoint 通过上述 1024 episode 验收，且可视化中没有明显高频抖动。
第二阶段再加入历史编码和轻量动力学随机化。
