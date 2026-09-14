# NezhaStand：DreamWaQ 四足静态站立

本项目以 ROS 包 `nezha_description` 的 Nezha URDF 为机器人资产，并在仓库内自带站立训练所需的
`legged_gym` 基础环境、任务注册与参数工具，以及 `rsl_rl` 的 DreamWaQ
Actor-Critic、VAE 历史估计器、PPO、rollout storage、runner、日志和 checkpoint
组件。运行时不再依赖
LZHMine 源码或 `LZHMINE_ROOT`。NVIDIA Isaac Gym 和带 CUDA 的 PyTorch 因
平台及授权原因仍需在训练机中单独安装。项目使用两个互不混用的环境：Python
3.8 的 `.venv` 负责训练和导出，Python 3.11 的 `.venv-mujoco` 负责 MuJoCo。

训练网络采用 `DreamWAQ-Go2W` 的完整数据流，不使用 MINE。站立任务本身仍是
Nezha 专用的四轮同时接地、静止、直立和四腿力矩/载荷均衡目标。

```text
NezhaStand/
├── assets/nezha/       # 从 nezha_description 导入的 URDF 与全部 STL
├── legged_gym/         # 仿真环境基类、配置、地形与任务注册
├── rsl_rl/             # Actor-Critic、PPO、storage 与 runner
├── nezha_stand/        # Nezha 站立任务配置、环境和奖励
└── scripts/            # 安装检查、资产检查、训练与评估入口
```

## DreamWaQ 训练结构

- 每个环境维护 5 帧历史，历史编码器输入 `5 x 46 = 230` 维，经过
  `230 -> 128 -> 64` 编码；
- VAE 从编码结果产生 16 维潜变量和 3 维机身线速度估计，并由
  `19 -> 64 -> 128 -> 46` 解码器重建当前观测；
- Actor 输入 `16 维潜变量 + 3 维速度估计 + 46 维当前本体观测`，网络为
  `65 -> 512 -> 256 -> 128 -> 12`；
- 策略只输出 12 个腿关节位置残差，四个轮关节使用零速度阻尼控制；
- Critic 输入 64 维无噪声特权观测，网络为 `64 -> 512 -> 256 -> 128 -> 1`，
  包含真实基座速度、高度、足端接触力和漂移；真实速度仅用作 Critic 输入和
  VAE 速度监督目标，不直接提供给 Actor；
- 仿真步长 5 ms，控制周期 20 ms，目标站立高度 0.50 m；
- 四个轮足必须同时着地，不包含抬腿、腾空、步态相位或速度跟踪目标；
- 使用以0.50 m为中心的连续平方高度惩罚；低于0.45 m时逐步取消静止和四轮
  支撑正奖励，低于0.42 m终止，避免策略通过低趴换取低力矩；
- 保留直立和静止目标，并对四腿对应关节的归一化力矩幅值平方差施加惩罚；
- 增加四轮垂直载荷占比均衡和接触轮水平打滑惩罚；
- 参考 LZHMine，在零速度站立任务中使用全腿默认姿态 L1 惩罚和额外髋关节 L2
  惩罚，避免后腿折叠；
- `default_dof_pos` 使用镜像站立角：左髋 -0.10、右髋 +0.10、四腿 thigh 0.925、
  calf -1.85 rad；PD 刚度为 hip/thigh/calf = 150/150/300 N·m/rad；
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

如果虚拟环境以前安装过 NumPy 1.24 或更高版本，`git pull` 不会自动降级，需执行：

```bash
uv pip install --force-reinstall "numpy==1.23.5"
```

Isaac Gym Preview 4 的 `torch_utils.py` 仍使用已经废弃的 `np.float`，因此训练环境
必须固定使用 NumPy 1.23.5。

确认导入的训练组件都来自本仓库：

```bash
python scripts/check_install.py
```

脚本会同时检查 Isaac Gym、CUDA、URDF，以及 `legged_gym`、`rsl_rl`、
`nezha_stand` 的实际导入路径。后三条路径都应位于 `/path/to/NezhaStand`，
不应出现 LZHMine。

## 2. 同步机器人描述与静态检查

当前资产来自 `/home/crown/nezha_description` 中较新的
`urdf/nezha_description.urdf`。该目录没有 MuJoCo MJCF；`package.xml` 和 launch XML
不能由 MuJoCo 加载。源 URDF 虽可被 MuJoCo 3.12 解析，但它用 fixed joint 把
`base` 固定到 `trunk`，且没有 MuJoCo actuator、传感器和地面，因此不能直接作为
自由基座 sim-to-sim 场景。同步脚本会复制全部 23 个 STL，
把 `package://` URL 改成仓库相对路径，移除 ROS/Gazebo 专用的虚拟 `base`、plugin 和
transmission，并以相同质量、惯量、关节原点、轴、限位和碰撞体重新生成
`mujoco/models/nezha.xml`。保留 `trunk` 为根刚体是 Isaac Gym 接触终止逻辑所需。

以后源目录再次修改时执行：

```bash
python3 scripts/sync_nezha_description.py \
  --source /home/crown/nezha_description
```

随后做源资产逐项校验：

```bash
cd /path/to/NezhaStand
source .venv/bin/activate
python3 scripts/validate_assets.py --source /home/crown/nezha_description
python3 scripts/validate_phase1.py
python3 scripts/check_dreamwaq.py
```

第三条命令无需启动 Isaac Gym，会用 CPU 完成一次 DreamWaQ 前向、rollout storage
和 PPO+VAE 反向更新，输出 `DreamWaQ smoke test: OK` 才表示网络链路完整。

不带 `--source` 时只检查仓库内 URDF 的拓扑、16-DOF 命名和 mesh 引用；带参数时还会
比较所有物理 link/joint 属性和全部 STL 的 SHA-256。

## 3. GPU 物理检查

先用四个环境和可视化窗口检查原始碰撞网格：

```bash
python scripts/check_physics.py --num_envs 4 --duration 5
```

观察机器人是否以约 0.54 m 高度出生，轻微下落后四轮触地，是否存在穿透、
弹飞、腿部碰撞误触发或高频抖动。脚本施加严格的零策略动作；物理检查不通过
时先修正碰撞体、出生高度或 PD 参数。

## 4. 正式训练

先做 100 次迭代的链路测试：

```bash
python scripts/train.py --headless --num_envs 256 --max_iterations 100
```

确认奖励与 `Loss/vae_*` 均为有限值后，再开始正式训练：

```bash
python scripts/train.py --headless --num_envs 2048 --max_iterations 20000
```

本版运行名为 `dreamwaq_stand_nezha_description_pose_v3`。机器人质量、惯量、几何、
碰撞体和关节限位都已变化，旧 checkpoint/`policy.pt` 不能续训或用于结果判定；
runner 和导出器会用 URDF SHA-256 拒绝不匹配的 checkpoint。应从头建立新运行。
TensorBoard除奖励外还应观察`Episode/base_height_m`和
`Episode/standing_height_fraction`。

建议先观察 1000 次迭代的曲线。若存活率没有持续提高，依次检查终止原因、
接触状态、动作是否饱和和各奖励分量，不要同时改多组参数。

## 5. 阶段验收

默认评估最新运行中的最新 `model_*.pt`：

```bash
python scripts/evaluate.py --headless --num_envs 256 --episodes 1024
```

也可指定模型：

```bash
python scripts/evaluate.py --headless --checkpoint_path /path/to/model_20000.pt \
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
| 四轮载荷占比误差 RMS | <= 0.10 |
| 接触轮水平速度 RMS | <= 0.05 m/s |
| 12 个腿关节默认姿态误差 RMS | <= 0.15 rad |

高度约束不能在基础站立阶段直接删除。旧的指数高度奖励在误差较大时趋近零，
导致继续降低机身几乎没有额外代价；现在改为参考LZHMine形式的平方误差，并按
0.10 m归一化，使整个低趴区间都有有效梯度。静止与支撑正奖励还会通过0.45–0.50 m
的连续高度门控。高度、静止和支撑三类约束本身只使用基座高度、接触和速度；
默认关节姿态则由下面两个独立惩罚项负责。`torque_balance`奖励比较四条腿上
同类型关节的力矩幅值（hip 对 hip、thigh 对 thigh、calf 对 calf），并先除以
各关节力矩上限再计算六组腿对的平方差。使用幅值是因为左右髋关节在镜像坐标系
中可能需要符号相反但物理效果对称的力矩；归一化则避免力矩上限较大的关节主导
奖励。该项并不能单独保证合理站立，因此仍保留机身直立、高度、接触、漂移、总力矩
以及默认关节姿态约束。`default_pose=-3.0` 使用 12 个腿关节相对默认角的绝对误差和；
`hip_default=-8.0` 额外约束四个镜像髋角。`foot_force_balance` 使用四轮竖直载荷占
总载荷的比例，与机器人总质量无关；`foot_slip` 只惩罚已经接地轮足中心的水平移动。
任务仍不包含抬脚高度、腾空时间、步态相位或速度跟踪奖励。相较 LZHMine 的复杂地形
运动配置，本任务保留较小的 0.15 动作尺度、平地和零速度命令，使策略只能在默认姿态
附近做必要的重力补偿，而不会为了 locomotion 大幅改变腿形。

为避免偶然性，正式进入第二阶段前应使用至少三个不同随机种子训练；至少两个
种子的 checkpoint 通过上述 1024 episode 验收，且可视化中没有明显高频抖动。
后续阶段只需逐步加入轻量动力学随机化；历史编码和速度估计已经包含在本阶段。

## MuJoCo sim-to-sim 站立验证

仓库内同时包含一个独立于 Isaac Gym 的 MuJoCo 推理框架：

```text
deploy/nezha_stand/policy_runtime.py  # 46 当前观测、230 历史观测及 PD 控制
mujoco/nezha_stand_config.yaml        # 关节顺序、增益、限幅与仿真频率
mujoco/models/nezha.xml               # 由当前 nezha_description URDF 生成的 MJCF
mujoco/models/nezha_scene.xml         # 平地与光照场景
mujoco/nezha_stand_sim.py             # MuJoCo 闭环仿真入口
scripts/export_policy.py              # checkpoint -> TorchScript policy.pt
scripts/validate_mujoco.py            # 模型和策略契约检查
```

MuJoCo 模型与当前导入 URDF 使用相同的 77.840691 kg 总质量、惯量、关节位置与
关节限制。碰撞体严格采用源 URDF：机身和 top box 为 box，四条小腿为 box，四个
轮子为半径 0.105 m、宽度 0.040 m 的 cylinder；髋、大腿、电机、IMU 和雷达只有
视觉几何。新 `trunk.STL` 已降至 5 万三角面，可由 MuJoCo 3.12 直接加载，不再需要
旧的 `mujoco/models/assets/trunk_visual.stl`。

### 1. 创建独立的 MuJoCo 环境

不要把 MuJoCo 安装进 Isaac Gym 使用的 Python 3.8 `.venv`。按照 LZHMine 的
MuJoCo 环境方式，另外创建 Python 3.11 环境；为保证两边可重复对照，NezhaStand
把 MuJoCo 固定为 LZHMine 当前实际使用的 3.12.0：

```bash
cd ~/NezhaStand
uv venv --python 3.11 .venv-mujoco
source .venv-mujoco/bin/activate
uv pip install -r mujoco/requirements.txt
```

检查环境：

```bash
python --version
python -c "import mujoco; print(mujoco.__version__)"
```

应分别显示 Python 3.11.x 和 MuJoCo 3.12.0。该环境无需安装Isaac Gym，也不要在
其中执行训练。

### 2. 导出训练策略

策略导出仍须回到包含训练模型定义的 Python 3.8 环境：

```bash
cd ~/NezhaStand
source .venv/bin/activate
```

默认自动选择 `logs/` 下最近一次训练及其最大编号 `model_*.pt`：

```bash
python scripts/export_policy.py
```

也可以明确指定已经通过评估的 checkpoint：

```bash
python scripts/export_policy.py \
  --checkpoint logs/Sep09_17-21-10_dreamwaq_stand/model_20000.pt
```

默认输出位置为：

```text
logs/exported/nezha_description_pose_v3/policy.pt
logs/exported/nezha_description_pose_v3/policy_metadata.json
```

导出脚本不启动 Isaac Gym，并会验证策略双输入 `[1, 46]`、`[1, 230]`，输出
`[1, 12]`，同时把当前 URDF 哈希写入 metadata。旧版普通 PPO checkpoint 或基于
旧机器人资产训练的 DreamWaQ checkpoint 都不能继续训练或导出。

### 3. 检查模型与策略

切回独立的 MuJoCo 环境：

```bash
source ~/NezhaStand/.venv-mujoco/bin/activate
cd ~/NezhaStand
python scripts/validate_mujoco.py
```

应看到 `model: OK`、`policy: OK`，以及：

```text
contract: current_obs=46, history=5x46=230, action=12, control=50 Hz
```

### 4. 运行仿真

以下命令均在 `.venv-mujoco` 中执行。Linux上打开交互式窗口并持续运行，窗口内
按 `R` 可重置机器人：

```bash
python mujoco/nezha_stand_sim.py --duration 0
```

macOS使用MuJoCo随包提供的`mjpython`启动窗口：

```bash
.venv-mujoco/bin/mjpython mujoco/nezha_stand_sim.py --duration 0
```

无显示器服务器上进行 10 秒测试：

```bash
python mujoco/nezha_stand_sim.py --headless --duration 10
```

指定其他策略文件：

```bash
python mujoco/nezha_stand_sim.py \
  --policy /absolute/path/to/policy.pt \
  --duration 0
```

仿真每 100 个策略步输出机身高度、横滚/俯仰角、水平漂移、轮速 RMS 和四腿对应
关节力矩差 RMS，同时输出 FL/FR/RL/RR 四轮接触状态、四轮平均垂直接触力、载荷
占比和载荷占比误差 RMS。接触力对每个 20 ms 控制周期内的四个 MuJoCo 物理子步
取平均。策略以 50 Hz 推理；每个动作保持四个 5 ms 物理步，并在每个
物理步重新计算 PD 力矩，与 Isaac Gym 训练控制流程保持一致。站立策略从第一个
控制周期立即接管；默认不执行零动作预沉降，因为训练环境同样没有该启动延迟。

需要更密集的实时输出时，可每 10 个策略步（0.2 秒）显示一次：

```bash
python mujoco/nezha_stand_sim.py --duration 0 --log_interval 10
```

同时将每个 50 Hz 控制周期的数据保存为 CSV：

```bash
python mujoco/nezha_stand_sim.py \
  --headless --duration 30 \
  --log_interval 10 \
  --csv stand_contacts.csv
```

CSV 包含四轮接触布尔值、垂直接触力、载荷占比、载荷误差、基座姿态、漂移、
轮速、默认关节姿态 RMSE 及力矩均衡指标，并自动创建和保存到项目根目录的
`results/`。也可以只写
`--csv`，此时使用 `nezha_stand_日期_时间_contacts.csv` 形式的自动文件名。
生成接触与载荷曲线：

```bash
python scripts/plot_mujoco_contacts.py stand_contacts.csv
```

默认输出 `results/stand_contacts.png`；使用 `--show` 可同时打开窗口，
使用 `--output result.png` 可在 `results/` 中指定图片名。只有显式传入绝对路径时，
CSV 或图片才会保存到 `results/` 之外。

MuJoCo 与 PhysX 的接触求解器并不相同，因此 sim-to-sim 的第一目标不是曲线完全
一致，而是机器人连续站立 10 秒、不倒地、不发生明显漂移或高频抖动。若结果不同，
先核对关节方向、观测顺序、PD 增益和碰撞接触，再考虑调整物理参数；不要直接修改
已经训练好的奖励函数来掩盖部署契约错误。
