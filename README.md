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
├── assets/nezha_description/ # 完整架构 URDF 与 23 个源 STL
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
- 使用以0.50 m为中心的连续平方高度惩罚；重载训练中从0.40 m开始逐步解锁静止和
  四轮支撑正奖励，低于0.30 m终止，避免初始策略尚未承载时立即反复重置；
- 保留直立和静止目标，并对四腿对应关节的归一化力矩幅值平方差施加惩罚；
- 增加四轮垂直载荷占比均衡和接触轮水平打滑惩罚；
- `default_dof_pos` 采用 LZHMine 与新版模型一致的镜像髋关节符号，并保留较弱的
  默认关节姿态先验；
- 躯干上方保留独立的 `top_box` 配重刚体，URDF 标称 90 kg，训练环境覆盖 88–92 kg；
- 保持平地和无推搡；启用摩擦系数 0.8–1.2、电机强度 0.95–1.05 以及顶部配重
  88–92 kg 的小范围随机化；
- 初始腿关节仅加入 +/-0.01 rad 扰动。

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

桌面 `nezha_description/urdf/nezha_description.urdf` 是本项目的权威源模型。本实验
已将其 `top_box` 配重设为 90 kg，并依据 `nezha_description_for_usd.urdf` 的 20 kg
惯量基准等比例设为 `Ixx=1.5318`、`Iyy=2.46555`、`Izz=3.1077 kg·m²`。
该 ROS 包没有可直接用于本项目的 MuJoCo MJCF；
`package.xml` 和 launch XML 也不是动力学模型。同步脚本会复制 23 个 STL，并把完整
URDF 写入 `assets/nezha_description/`。训练 URDF 唯一允许的变化是把 23 处
`package://nezha_description/meshes/` 改为仓库相对路径；源文件的注释、节点顺序、
25 个 link、24 个 joint、质量/惯量、16 个 transmission 和 22 个 Gazebo 节点均保留。
MuJoCo MJCF 是从同一 URDF 单独生成的派生文件：转换时才忽略 ROS 虚拟根、
transmission 和 Gazebo 标签，并补充浮动基座、执行器及传感器，不会修改训练 URDF：

```bash
python3 scripts/sync_nezha_description.py \
  --source /path/to/nezha_description
```

```bash
cd /path/to/NezhaStand
source .venv/bin/activate
python3 scripts/validate_assets.py --source /path/to/nezha_description
python3 scripts/validate_phase1.py
python3 scripts/check_dreamwaq.py
```

第三条命令无需启动 Isaac Gym，会用 CPU 完成一次 DreamWaQ 前向、rollout storage
和 PPO+VAE 反向更新，输出 `DreamWaQ smoke test: OK` 才表示网络链路完整。

不带 `--source` 时只检查仓库内拓扑、配重和 mesh 引用；带 `--source` 时还会验证
目标 URDF 除 mesh 路径外与源文件逐字等价，并逐一比较 23 个 STL 的 SHA-256。

## 3. GPU 物理检查

先用四个环境和可视化窗口检查原始碰撞网格：

```bash
python scripts/check_physics.py --num_envs 4 --duration 3
```

观察机器人是否以约 0.54 m 高度出生，四轮触地，是否存在穿透、弹飞、腿部碰撞误触发
或高频抖动。该脚本施加严格的零策略动作，并不是训练后的控制器；90 kg 模型在零动作下
会逐渐下沉，因此这里用约 3 秒做资产/接触检查。稳定承载能力必须用训练后的 checkpoint
执行 `evaluate.py` 验收。

## 4. 正式训练

先做 100 次迭代的链路测试：

```bash
python scripts/train.py --headless --num_envs 256 --max_iterations 100
```

确认奖励与 `Loss/vae_*` 均为有限值后，再开始正式训练：

```bash
python scripts/train.py --headless --num_envs 2048 --max_iterations 30000
```

训练的标称质量直接读取 URDF，不通过训练命令行覆盖。当前 URDF 为 90 kg，训练时每个
并行环境在 88–92 kg 内采样顶部配重；做其他负载实验时，
先修改权威 URDF 中 `top_box/inertial` 的 `mass` 和 `inertia`，再同步、检查并从头训练。
服务器上若直接修改仓库副本，可运行
`python3 scripts/sync_nezha_description.py --source assets/nezha_description` 重新生成 MJCF。
不同负载应建立独立运行，并用 `--run_name payload_90kg` 等名称标记。参考 20 kg 的
`nezha_description_for_usd.urdf`，同形状配重质量为 `m` 时可按
`[Ixx,Iyy,Izz] = m * [0.01702,0.027395,0.03453] kg·m²` 缩放惯量，不要只改质量。

本版默认运行名为`dreamwaq_stand_nezha_description_v4`。不要从旧的
`dreamwaq_stand/model_7000.pt`续训；旧策略已经学到低趴局部最优，应从头建立
新运行。TensorBoard除奖励外还应观察`Episode/base_height_m`和
`Episode/standing_height_fraction`。随机化采样均值会记录为
`Episode/domain/friction_mean`、`Episode/domain/motor_strength_mean`和
`Episode/domain/payload_mass_kg_mean`，可用于确认三项随机化确实生效。

建议先观察 1000 次迭代的曲线。若存活率没有持续提高，依次检查终止原因、
接触状态、动作是否饱和和各奖励分量，不要同时改多组参数。

### 使用 Weights & Biases 查看和分类训练曲线

TensorBoard 会继续照常写入每个 `logs/<run>/` 目录。W&B 是可选功能，并且由于
Isaac Gym 训练环境固定为 Python 3.8，需要安装仍支持 Python 3.8 的 0.23–0.24 版本：

```bash
source .venv/bin/activate
uv pip install --no-build-isolation -e '.[wandb]'
wandb login
```

启动在线记录：

```bash
python scripts/train.py \
  --headless \
  --rl_device cuda:0 \
  --num_envs 2048 \
  --max_iterations 30000 \
  --wandb \
  --wandb_project nezha-stand \
  --wandb_group payload-90kg \
  --wandb_name payload90-seed42-v1 \
  --wandb_tags baseline,friction-rand,motor-rand
```

W&B 中的组织层次为：`project` 管整个 Nezha 站立课题，`group` 管同一负载或同一组
实验，`name` 标识某一次训练，`tags` 用于筛选随机化方式或训练阶段。不指定时，项目
默认为 `nezha-stand`，分组默认为 `payload-90kg`，run 名与本地带时间戳的日志目录一致，
并自动添加 `dreamwaq`、`standing`、`domain-randomization` 和 `payload-90kg` 标签。
当前随机种子也会自动形成 `seed-42` 这类标签。

代码只向 W&B 发送 TensorBoard 中同名的标量指标，包括 `Loss/*`、`Train/*`、
`Episode/*`、`Policy/*` 和 `Perf/*`；同时关闭源码、Git 状态、控制台输出和系统信息
采集，不会自动上传 URDF、mesh、完整配置或 checkpoint。未添加 `--wandb` 时不会
导入、连接或写入 W&B，原有 TensorBoard 行为不变。

网络不稳定时可先离线记录：

```bash
python scripts/train.py --headless --num_envs 2048 --max_iterations 30000 \
  --wandb --wandb_mode offline --wandb_name payload90-seed42-v1
```

离线数据位于 `logs/wandb/`。联网并登录后，可以指定某个离线 run 目录同步：

```bash
wandb sync logs/wandb/offline-run-时间戳-runID
```

续训时默认创建一个新的 W&B run；如果希望曲线继续写入原来的远程 run，需要同时
传入训练 checkpoint 参数和原 W&B 页面显示的 run ID：

```bash
python scripts/train.py --headless --resume --load_run RUN目录 --checkpoint -1 \
  --wandb --wandb_run_id 原W&B运行ID
```

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

高度约束不能在基础站立阶段直接删除。旧的指数高度奖励在误差较大时趋近零，
导致继续降低机身几乎没有额外代价；现在改为参考LZHMine形式的平方误差，并按
0.10 m归一化，使整个低趴区间都有有效梯度。重载训练中的静止与支撑正奖励通过
0.40–0.50 m 的连续高度门控。新增 `nominal_pose=-1.0` 平方惩罚，使12个腿关节靠近
LZHMine 的默认站姿，但其权重低于高度与直立目标，允许策略为承载90 kg而调整姿态。
`torque_balance`奖励比较四条腿上
同类型关节的力矩幅值（hip 对 hip、thigh 对 thigh、calf 对 calf），并先除以
各关节力矩上限再计算六组腿对的平方差。使用幅值是因为左右髋关节在镜像坐标系
中可能需要符号相反但物理效果对称的力矩；归一化则避免力矩上限较大的关节主导
奖励。该项并不能单独保证合理站立，因此仍保留机身直立、高度、接触、漂移和总力矩
约束。`foot_force_balance` 使用四轮竖直载荷占总载荷的比例，与机器人总质量无关；
`foot_slip` 只惩罚已经接地轮足中心的水平移动。配置中没有抬脚高度、腾空时间或步态
奖励。默认关节角度同时用于复位姿态、PD 零动作参考、观测归一化和较弱的姿态先验；
策略仍可通过持续动作残差选择适合90 kg负载的稳态关节角度。

为避免偶然性，正式进入第二阶段前应使用至少三个不同随机种子训练；至少两个
种子的 checkpoint 通过上述 1024 episode 验收，且可视化中没有明显高频抖动。
顶部配重在环境创建时采样，并在大量并行环境间覆盖整个范围；摩擦系数和电机强度在
每个 episode 复位时重新采样。三个参数在单个 episode 内保持不变，便于历史编码器从
状态历史中识别其动力学。质心、Kp/Kd、其他连杆质量、推搡、恢复系数和延迟随机化仍关闭。

## MuJoCo sim-to-sim 站立验证

仓库内同时包含一个独立于 Isaac Gym 的 MuJoCo 推理框架：

```text
deploy/nezha_stand/policy_runtime.py  # 46 当前观测、230 历史观测及 PD 控制
mujoco/nezha_stand_config.yaml        # 关节顺序、增益、限幅与仿真频率
mujoco/models/nezha.xml               # 桌面版 URDF 对应的浮动基座 MJCF
mujoco/models/nezha_scene.xml         # 平地与光照场景
mujoco/nezha_stand_sim.py             # MuJoCo 闭环仿真入口
scripts/export_policy.py              # checkpoint -> TorchScript policy.pt
scripts/validate_mujoco.py            # 模型和策略契约检查
```

MuJoCo 模型由同一份 `nezha_description` URDF 生成，质量、惯量、关节位置、轴、
限位和碰撞体与训练资产对应。新版 `trunk.STL` 为 5 万三角面，可由 MuJoCo 3.12
直接加载。默认运行不覆盖动力学参数，`top_box` 的质量、质心和惯量均来自 URDF
派生的 MJCF，因此严格 sim2sim 与训练模型保持一致。

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
logs/exported/latest/policy.pt
logs/exported/latest/policy_metadata.json
```

导出脚本不启动 Isaac Gym，并会验证策略双输入 `[1, 46]`、`[1, 230]`，输出
`[1, 12]`。旧版普通 PPO checkpoint 与新网络形状不兼容，不能继续训练或导出。

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

不指定 `--payload_mass` 时直接使用 URDF 派生 MJCF 的质量，这才是严格 sim2sim。
`--payload_mass 10` 仅保留为 MuJoCo 单边灵敏度测试；它不会修改训练资产，也不能代替
“修改 URDF、重新生成 MJCF、重新训练”的正式负载实验流程。

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
  --csv logs/mujoco/stand_contacts.csv
```

CSV 包含本次 `payload_mass_kg`、四轮接触布尔值、垂直接触力、载荷占比、载荷误差、基座姿态、漂移、
轮速及力矩均衡指标。生成接触与载荷曲线：

```bash
python scripts/plot_mujoco_contacts.py logs/mujoco/stand_contacts.csv
```

默认输出 `logs/mujoco/stand_contacts.png`；使用 `--show` 可同时打开窗口，
使用 `--output result.png` 可指定图片路径。

MuJoCo 与 PhysX 的接触求解器并不相同，因此 sim-to-sim 的第一目标不是曲线完全
一致，而是机器人连续站立 10 秒、不倒地、不发生明显漂移或高频抖动。若结果不同，
先核对关节方向、观测顺序、PD 增益和碰撞接触，再考虑调整物理参数；不要直接修改
已经训练好的奖励函数来掩盖部署契约错误。
