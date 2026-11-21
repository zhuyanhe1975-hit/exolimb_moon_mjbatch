# exolimb_moon_mujoco_playground

从 `../exolimb_moon_mjlab` 迁出的独立项目。项目目录已按用户要求修正为 `exolimb_moon_mujoco_playground`，Python 包名使用上游的 `mujoco_playground`。

使用本机当前 **MuJoCo Playground 1.0.1 主线：PyTorch + MuJoCo Warp + REAF**，不是 rss2025 JAX/MJX。新 `src/` 和入口没有 MJLab 依赖。原项目未修改，旧源代码留在 `reference/mjlab/` 供核对，未复制历史训练日志及检查点。

## 已迁移

- 默认 v15/detail3 月面任务；23 轴人体，固定头/腕，无额外根支撑；64 块 16m 原生高度场，保存原始 bank，避免运行时重新生成地形。
- 146 维 actor/critic 观测，保留原生 frame/touch/subtree 传感器及躯干 COM 局部速度；无稳定裕度或 LIPM 预测输入。
- 原奖励权重、动作 scale=.25/clip=2、关节到执行器映射、原生 PD、60Hz 控制/120Hz 物理、重力 1.635、10 秒回合、默认8192世界。
- 随机地形/安全出生点/朝向、全鞋底覆盖余量、1m 活动范围及60度跌倒终止；训练默认水平扰动，回放默认关闭。
- Playground 原生 REAF Device/TaskLayer 与 RSL-RL train/play 入口。新日志独立保存，检查点显式核对导出地形 SHA；不匹配需显式 `--terrain-transfer`。

## macOS / 原生 MuJoCo CPU 回放

`./run.sh` 无参数时回放导入的站立策略。`./run_cpu.sh` 默认从 `logs/cpu_train/` 选择最近保存的训练检查点。两个入口都支持 `--best [任务]` 选择固定归档、`--latest-training` 选择本地训练结果和 `--checkpoint` 指定文件。回放默认读取检查点记录的训练地形，可用 `--terrain-idx` 覆盖。CPU 回放使用原生 MuJoCo 4.0.1 动力学和 CPU PyTorch 推理，CUDA/Playground 入口为 `./run_playground.sh`。

| 归档任务 | 检查点 |
| --- | --- |
| `exo_jump`（裸 `--best` 默认） | `bests/exo_jump_position_pd500_2026-11-19/model_199.pt` |
| `forward_jump` | `bests/forward_jump_boot_yaw_2026-11-18/model_104.pt` |
| `jump` | `bests/jump_flow_rebound_2026-11-18/model_149.pt` |

```bash
./run.sh --list-bests
./run.sh --best --viewer studio --viz skin --jump-speed 1.0
./run.sh --best forward_jump --jump-speed 1.5
./run.sh --best jump --jump-height 0.15
./run.sh --latest-training --task exo_jump
```

`--best --task jump` 等价于 `--best jump`。归档缺失时直接报错。`--best`、`--latest-training` 和 `--checkpoint` 每次只选一种。

### Linux / RTX 5090

本机已有 `~/dev-envs/robotics-playground-py312` 时可直接复用。CPU 回放优先使用 `CPU_PYTHON` 或项目内的 `.venv-cpu`，Linux 上两者均未配置时使用现有 Playground 解释器。

```bash
# 首次使用 Linux Studio 时，用现有 MuJoCo 4.0.1 SDK 编译项目内 UI 扩展
~/dev-envs/robotics-playground-py312/bin/python scripts/build_native_ui.py

./run.sh --best --viewer studio --viz skin --jump-speed 1.0
./run.sh --backend playground --device cuda:0 --viewer studio --viz skin
```

第一条回放命令使用 Mac 归档的外骨骼策略，物理和推理都在 CPU 上。第二条使用 RTX 5090 的 CUDA/Warp 站立任务，自动选择对应的 Playground 检查点。指定 Python 可分别使用 `CPU_PYTHON` 和 `PLAYGROUND_PYTHON`。

```bash
uv venv --python 3.12 .venv-cpu
uv pip install --python .venv-cpu/bin/python -r requirements-cpu.txt
./run_cpu.sh                         # 原生 classic 窗口，单人，自动回放最新本地训练
./run.sh --viewer studio --viz skin --num-envs 8  # CPU物理/推理 + Studio八人航天服显示
./run_cpu.sh --terrain-idx 24
./run_cpu.sh --headless --seconds 5 --report outputs/cpu_check.json
```

`./run.sh --help` 查看 CPU CLI 中文帮助；兼容 `--resume-path` / `--resume_path`、`--terrain_idx` 等旧名称。可显式设置 `--device cpu --viewer classic --viz physx --num-envs 1`；当前支持 `--viewer classic|studio` 和 `--viz physx|skin`；`--num-envs N` 设置独立CPU环境数（默认1），支持 `--num-envs 8` 在同一块月面上显示8人；人物互不碰撞、随机分散出生，跌倒/超时独立复位。多环境使用独立 MjData，共享同一物理模型；显示端共享地形、纹理/网格/灯光，蒙皮逐人独立。`--headless` 必须配合正数 `--seconds`。帮助和参数校验不会加载仿真运行时。

macOS classic 窗口自动通过 `mjpython` 启动，Studio使用普通Python主线程。空格暂停/继续，R复位，`[` / `]` 切换64块地形；关闭窗口退出。窗口默认显示物理代理人体；`./run.sh --viewer studio --viz skin` 启动原生 Studio + 航天服 PBR 蒙皮。Studio 在 macOS 主线程运行，使用安装包自带的 ImGui bridge，无需编译旧 Linux UI 扩展。物理模型和显示模型独立，地形切换复用显示资源。动力学120Hz、策略60Hz，窗口按墙钟1×限速，10秒回合自动复位；回放不训练、不施加随机扰动。物理和策略均在 CPU，OpenGL 负责显示。

多人启动优化：构建多人显示时先剥离人体模板里的重复资产声明，再将每个人引用到第一人的共享纹理/材质/网格；保留独立蒙皮、骨骼和全部人体。避免64人拼接完成后才逐个删除大量重复资产。当前Mac实测64人skin显示spec拼接约34秒→0.8秒，编译后模型约1.3秒就绪（不含解释器导入/窗口首帧），未降低模型或贴图精度。Studio/classic各64人的原生窗口短启动检查通过，非长时稳定性或实时帧率保证。启动日志分段显示CPU环境、人体挂接、显示模型编译、窗口和渲染资源上传进度。

交互统一性：`classic + physx`、`classic + skin`、`studio + physx`、`studio + skin` 共用相同按键控制器、人体/body映射和原生扰动力计算。显示样式只改变外观；鼠标点选也会同步当前人体编号。Studio底部状态栏和classic窗口文字提示均显示当前人体及相机模式。四种组合已通过原生窗口集成检查（输入层注入按键事件，覆盖切换、跟踪、总览、地形键及扰动力映射/松手清零）。

相机与人体选择（CPU classic/Studio，恢复原标点键位）：`,` 上一个人体、`.` 下一个人体（首尾循环并进入近距跟随）；`/` 开启/关闭当前人体跟踪；`A` 切换全体总览与原视角；`Esc` 退出跟踪；`[` / `]` 仍切换上一块/下一块地形，`R` 复位。切换人体会选中其骨盆并清除旧鼠标外力，所有独立环境继续运行。Studio底部状态栏及终端显示当前人体编号/相机模式。点击画面后按键，使用英文半角标点；在文本输入框中编辑时不触发快捷键。T/N/P保持原生显示选项，不作为相机或人体切换的别名。


macOS Studio输入焦点：窗口完成首帧渲染后仅请求一次激活，并开启SDL首次点击穿透，防止画面持续刷新但新窗口没有键鼠焦点。切换到其他应用后不会重复抢回焦点。启动日志 `[Studio input] keyboard_focus=True` 表示该窗口已经获得键盘焦点；启动时先False再True是正常的焦点交接。Studio现同时接受Control和Command作为鼠标扰动修饰键；classic原生Control不变，因此两者可统一使用Control。修饰键拖拽不会同时移动相机。

Studio选中高亮：双击选中后，以黄色轮廓标出当前人体；逗号/句号切人也同步轮廓，点击空白/地形取消。高亮只作用于当前人体的renderable，不修改共享材质贴图、蒙皮或施力锚点；skin和physx均有选择反馈。

classic skin显示适配：共享Studio的模型、UV、清洁版服装颜色贴图和太阳方向，额外使用classic支持的补光、布料/面罩高光及RGB颜色空间适配，避免直接使用无法上传的Filament KTX环境光。多人仍共享一套模型资产与场景灯光。classic旧OpenGL不支持同等PBR法线、金属度/粗糙度贴图和环境镜面反射，因此外观为近似对齐，不保证与Studio逐像素一致。

Studio skin拾取修复：项目内 `skin_picking.py` 使用原生更新的蒙皮顶点、相同相机投影和可见性设置进行拾取，避开本机Studio原Pick的多skin顶点缓存累加错误。拖拽锚点取鼠标射线与蒙皮的真实表面交点，不再用最近顶点位置代替；拾取不会推进跟踪相机的平滑状态。仅在本进程替换Python Studio的skin拾取，physx保留原生几何拾取，不改共享安装包或物理模型。原生窗口双击处理入口集成检查及单人/8人蒙皮、遮挡/表面交点检查通过（输入层注入，非本轮真实OS鼠标验收）。

鼠标施力：先左键双击选中人体部位，再按住 **Control（⌃）+鼠标右键拖动** 施加平移外力；**Control+左键拖动** 施加旋转扰动（转矩），配合Shift切换水平操作，Control+Option+左键可代替右键进行平移。classic使用Control；macOS Studio也统一支持Control，并保留Command兼容。施力时保持Run，松开鼠标停止施力；暂停后拖动不用于动态推人。多人模式仅作用于所选人体，实际外力映射回对应CPU世界。建议使用有左右键的鼠标。

检查点原训练地形快照缺失，因此这是当前 detail3 地形上的策略迁移，不是精确重放；原生 CPU 和 Warp 轨迹不保证等价。2026-11-15 在 Apple Silicon 上第12块地形的5秒动态闭环通过，300控制步、0失败复位、无非有限值；另已打开并目视检查原生窗口。结果仅代表本次有限验证。

接触显示：`C` 切换红色接触点，`F` 切换青色接触力箭头（0.003米/牛顿）。CPU classic/Studio、physx/skin都读取各人体真实CPU求解器的接触数据；显示全部环境的接触，关闭开关会清除叠加。箭头代表碰撞接触力，不是鼠标外力或关节力矩。

## 本地 CPU 批量训练

复用回放的 `.venv-cpu`，在项目目录执行：

```bash
./train_cpu.sh --num-envs 1024 --iterations 200
```

默认用原生 `mujoco.batch.Batch` 在 CPU 采样，策略推理也在 CPU。脚本按环境数和可用 CPU 核心数自动设置线程。每轮每个环境采样 32 步，默认在第 12 块地形训练并开启扰动。Apple MPS 可用且每轮至少有 32,768 条转移时，默认用 MPS 做 PPO 更新。较小批次或 MPS 不可用时使用全 CPU。可用 `--update-device cpu` 强制全 CPU。训练从随机初始化开始，保留原站立任务的观测、动作及奖励。

运行 `.venv-cpu/bin/python scripts/benchmark_cpu_batch.py --physics-workers 4`，可复查 16、64 环境下两种后端的观测、奖励和复位一致性，并比较包含策略前向、步进和复位的采样吞吐量。

检查点和 TensorBoard 日志保存在 `logs/cpu_train/<运行时间>/`，也可用 `--log-dir` 指定新目录。用 `./run.sh --terrain-idx 12 --checkpoint <检查点路径>` 回放。将 `--iterations` 改为 3 可快速检查训练链路。

连续跳跃任务沿用相同的采样和 PPO 配置。脚底触地后开始屈膝，达到压缩条件后蹬伸，实际离地后进入腾空。相位会等待触地或离地并根据竖直速度区分上升和下降。腿部关节使用对称下蹲和蹬伸参考，策略学习修正动作、跟踪竖直轨迹、保持平衡和减少横向漂移。检查点记录动作版本，回放也能加载原固定节奏的跳跃策略。

屈膝参考从触地时的实测膝角开始，按下落速度调整缓冲深度。奖励惩罚电机正机械功、触地时突变的竖直速度，以及落地缓冲窗口内最大的下肢力矩占比。峰值力矩项只在落地时生效，减少落地冲击时仍可保留蹬伸用力。

跳跃训练在每个环境复位或完成一次起落后，从指定范围内均匀随机抽取下一跳的目标高度，默认 0.09–0.20 米。策略根据脚底接触、屈膝程度和竖直速度决定什么时候开始蹬伸、蹬伸多快。奖励在实际起跳和落地时评估高度与稳定性，只有落地缓冲和蹬伸使用局部平滑参考。

最新反弹策略用脚底支撑、屈膝程度和竖直速度的反向趋势来衔接缓冲与蹬伸，并在最低点附近惩罚停留。这个版本只调整控制和奖励，沿用原有动力学。

跳跃训练可用 `--init-checkpoint` 从几何版本一致、支持高度指令的检查点初始化策略与价值网络。优化器按新任务重新初始化。

```bash
./train_jump.sh --num-envs 1024 --iterations 200 --jump-height-range 0.10 0.50
./run_cpu.sh --task jump --viewer studio --viz skin
./run_cpu.sh --viewer studio --viz skin --jump-height 0.15
```

训练范围写入检查点。用 `--jump-height` 在推理回放时设定范围内的目标跳高，省略时仍按训练范围逐跳随机选择目标。未记录范围的事件驱动检查点沿用 0.09–0.20 米。用 `./run_cpu.sh --task stand` 选择最新的站立策略。跳跃回放报告包含每人的完整跳跃次数和最长连续次数。

新 CPU 训练默认使用 `fitted` 几何。脚底碰撞盒和脚底传感器按蒙皮靴底对齐，背包盒按蒙皮外壳对齐。原始质量、质心和惯量保留。回放默认读取检查点记录的几何版本，旧检查点继续使用原脚底。可用 `./run_cpu.sh --task jump --geometry fitted --viz physx` 在现有策略上查看新几何，脚底碰撞变化可能改变原策略的站立和跳跃表现。

前进跳跃是独立任务，默认从 `bests` 中归档的连续反弹策略初始化。复位时分别以 10% 的概率采样 0 和 1.5 米/秒，其余速度在两者之间均匀采样。策略跟踪出生朝向上的速度；接近零速时还惩罚沿跑道漂移。

新版奖励在确认完整落地时，按离地到首次触地之间的向前质心位移给分。短距离跳跃得到的跨距分较少，落地次数本身没有固定加分。较低的目标速度适用较小的跨距，零速可以原地跳。跳高由策略决定。回放报告记录每人的平均跨距和最长一跳，单位为米。接近地形边缘时结束当前回合。

高速档额外奖励沿跑道的整人体质心速度。新版姿态按连续立定跳的节奏给奖励，蹬伸末段适度后展，腾空越过最高点后转为向前屈，落地顺势屈膝。姿态参考由实际膝关节伸展和质心升降决定。

完整跳跃会按实测质心上升量和腾空时间获得额外奖励，增益随上升量增加而逐渐收敛。落地后的短时屈膝奖励随下落速度和接触时序调整，继续计入落地冲击与力矩代价。

双臂按镜像关节角度计入协调奖励，允许小幅左右差异。双脚前后错位按脚底中心在身体前向上的距离计入代价，下降和落地缓冲时权重较高。脚距和因地形产生的脚底高度差可随动作调整。

协调奖励与离地前后短时后展的奖励同时生效。完成落地的跳跃按实际离地倾角、下降前屈和腾空时间计入姿态衔接分。

承重屈膝和承重蹬伸会累计有效支撑时间，单步接触力的贡献有上限。质心接近由下降转为上升时进入蹬伸，向上速度的增量提高完成跳跃的支撑分。奖励对足底载荷峰值和落地时突然升高的载荷扣分。回放报告记录有效支撑时间、向上速度增量和控制步采样的足底峰值载荷，载荷以月球体重为单位。

支撑衔接奖励使用实际足底载荷、质心位移和膝角变化。完成跳跃后会给承重下沉屈膝、随后承重回升伸膝的过程加分。前进档同时加强跨距与质心前进速度的奖励。

```bash
./train_forward_jump.sh --num-envs 1024 --iterations 360
./run_cpu.sh --task forward_jump --viewer studio --viz skin --jump-speed 1.5
```

`--jump-speed-range MIN MAX` 可指定其他训练速度范围，单位为米/秒。新版前进策略回放用 `--jump-speed` 设置目标速度。原地跳跃策略的 `--jump-height` 以米为单位。

## 外骨骼助力跳跃

两根伸缩支杆的虎克铰装在背包下部两侧，贴近侧壁并沿背包侧面向前收。靴侧斜支座向外下方连接球铰，静态站姿下球铰座最低点与鞋底齐平。MuJoCo 用两个转动关节和一个直线关节建模支杆，再用点约束闭合靴端球铰。套筒和内杆是有质量的刚体，工作长度为 0.63–1.22 m。对应靴子承重且控制器进入蹬伸时，支杆逐渐施加伸展力。单根支杆的力可设为 0–500 N，新训练默认上限为 500 N。回放默认采用检查点记录的上限，可用 `--assist-force 500` 覆盖。`--assist-force 0` 保留外骨骼的质量和关节约束，作为无助力对照。

`run_exo_jump.sh` 优先加载本地最新的外骨骼策略，首次克隆时直接使用 `bests/exo_jump_position_pd500_2026-11-19/model_199.pt`。用同一个检查点、同一地形和种子跑两次，可以对照跳跃次数、速度、支杆峰值力、支杆正机械功和人体关节正机械功。报告中的机械功按 60 Hz 采样估计。

新外骨骼策略使用 MuJoCo 原生关节位置 PD。左右杆各有一个动作，指定从当前长度向前伸出 0–0.15 m 的目标增量，目标长度受行程限制。沿用承重和蹬伸门控，由 PD 位置误差和伸缩速度决定推力，推力限制为 0 至设定上限。策略观测包含每根杆的实际长度、伸缩速度、实测推力、上一帧的目标长度和对应靴底载荷，总计 26 个动作和 169 个观测。用旧检查点初始化训练时自动迁移网络，回放则根据检查点选用当时的控制方式。

PD 外骨骼的训练和回放都收小肩部的目标摆幅。上臂略向身体内侧收，肘部保留策略给出的平衡动作。

外骨骼策略可用 `--init-checkpoint` 和 `--stance-center-weight` 做双脚居中微调。居中代价测量双脚中点到骨盆矢状面的水平横向距离，留出 3 cm 容差。微调使用固定学习率 `1e-4`，保留速度、落地和支杆做功等奖励。该选项默认关闭。

`--exo-control force` 使用 MuJoCo 原生滑杆力执行器，策略直接请求左右杆推力。承重、蹬伸、行程与杆速门控继续生效。由位置 PD 检查点初始化时，保留人体输出，清除旧支杆目标指令的输入权重与归一化统计，并将两路支杆输出初始化为各 80 N 的门控前请求（受配置上限限制）。检查点记录执行器模式，回放据此构建模型。

`--exo-control force-policy` 移除支杆对全身蹬伸相位和起止时间包络的依赖。策略每个控制步决定左右推力，同侧承重、行程、缩杆释放和推力上限仍生效。有效正功按同侧承重及质心上升计分，可在任意全身相位获得。由上一版原生力控检查点初始化时，直接继承网络和归一化参数。用 `run_exo_force.sh CHECKPOINT` 回放。

`train_exo_force.sh` 默认选择 `--exo-control force-action`。每侧 action 经 `[-1, 1]` 裁剪后线性映射到 `[0, assist_force]`，直接控制原生滑杆执行器。人体代理消耗采用逐关节正功除以 0.25、负功绝对值除以 1.2 后求和，同时计入人体力矩、无承重时的支杆推力和超出接触支撑余量的踝侧倾力矩。支杆正负功用于诊断。双踝侧倾机械限位为 ±8°。

落地缓冲和承重蹬地分别在阶段结束时比较两侧承重期间的平均支杆推力。只有两侧各累计承重至少 30 ms 才计入均衡代价，推力分配在约 40:60 以内时不扣分。阶段均衡影响奖励，推力仍由 action 直接决定。训练报告按阶段和左右杆分别记录承重期间的绝对机械功。

力控训练可加 `--load-sharing` 试用 v3 奖励。它分别累计缓冲期和蹬地期同侧着地时的支杆冲量，并按单脚载荷累计有效承重时间（每脚计到一倍月面体重为止）。每步按左右累计差额的变化计分，允许两脚先后接力。单靴与地形接触点的法向力标量和超过六倍月面体重时收取连续超载代价。评估竖直缓冲效果时，需把各接触点含摩擦的反力转到世界坐标，逐脚求和后取竖直分量。人体代谢代理采用更高权重并保留高功耗样本的完整代价。由直接力控策略初始化时，两路支杆输出层参数各向两侧平均值移动一半，随后独立学习。报告增加左右杆分阶段的冲量，单位为 N·s。

`--vertical-support` 启用 v4 力控训练。模型用原生 MuJoCo 地形到单靴的净力传感器汇总接触法向力和摩擦力。训练与回放都将向上的世界坐标竖直分量送入足底载荷观测和承重代价。每脚连续卸载至少两个控制步后，在下降落地或缓冲期重新承重时开启 50 ms 窗口。单脚竖直载荷超过 4.5 倍月面体重、或窗口内向上冲量累计超过 0.15 倍月面体重·秒时计入软代价。报告分别给出完整窗口次数、向上冲量总和、向上与水平合力峰值、前 12 个接触点水平法向分量绝对值之和的峰值，以及最大靴底接触穿入。传感和计分采样频率为 60 Hz。回放依据检查点奖励版本选择载荷定义。

`--posture-balance` 启用 v5，沿用 v4 的竖直支撑反馈。它用 1.5 秒时间常数平滑背包和腰关节的有符号侧倾角。当持续偏角超过 1.5°（背包）或 4°（腰），并且当前倾斜仍强化同方向偏差时，训练计入姿态代价。蹬地期额外比较两靴累计向上冲量与两杆同侧承重时的累计冲量，允许两脚先后承重。由 v4 初始化时直接继承左右杆的独立输出头。策略继续按 action 请求 0–500 N 推力。

`--stance-transition` 启用 v6，继承 v5 的姿态和人体功耗目标。制动期按重心下降速度及同脚后跟、前掌中较小的竖直反力计分。通过地形接触的法向和两条切向还原各点的世界竖直力，使用每脚按力排序的前 12 个接触点。重心回升时，按同侧足底承重与杆伸长速度的重叠计分，并兼顾表现较弱的一侧。支杆推力仍由策略的两个独立 action 决定。训练示例：

```bash
./train_exo_force.sh --stance-transition --init-checkpoint outputs/assist_posture_v5/train/model_159.pt
```

`--economical-hops` 在 v4 竖直力控奖励上比较跳跃经济性。通过有效落地判定后，每跳的质量奖励按固定系数 1 结算，并取消腾空阶段逐帧的固定加分。两组沿用同一套速度、步幅、抬升、姿态、关节出力与落地约束。用 `--keep-strut-heads` 从同一个 v4 继续训练时，两组均可原样继承左右杆输出头。实验以相同实际速度下累计人体估算能耗除以累计前进距离（J/m）比较，同时报告失败复位及落地载荷。

`--long-hop-period 1.7` 可在 v7 上继续训练低频长跳。每个有效落地事件同时检查与上一次有效落地之间的时间和前进距离。前进距离参考值为速度指令乘以目标周期，两项完成比例取较小值并平方，达到目标后不再增加。首跳用腾空时长和腾空前进距离。接近零速时逐渐取消该权重。目标周期是可调的奖励参考，实际跳频、跳距和节能率以同速回放为准。 长跳出现持续侧倾时，可加 `--roll-balance`，单独启用 v5 的背包和腰部侧倾代价。

```bash
./train_exo_force.sh --long-hop-period 1.7 \
  --init-checkpoint outputs/assist_energy_v7/candidate/model_159.pt
```



`./train_exo_stance_center.sh` 使用 1024 个 MuJoCo Warp CUDA 环境。物理、观测、奖励、策略采样、GAE 和 PPO 更新均在 CUDA 上。启动时在主机编译 MuJoCo 模型并筛选安全出生点，之后在 CUDA 上独立采样位置和朝向。脚本沿用原归档的 169 维观测、26 维动作与两阶段参数。CUDA 和原生 CPU 后端的接触求解与随机序列可能产生不同轨迹。用 `--checkpoint` 指定导出的模型可做原生 CPU Studio 回放。

Warp 默认复用物理步进、传感器刷新和复位的 CUDA Graph，观测也使用固定形状图和合并的地形查询。用 `MOON_WARP_GRAPH=0` 可回退到逐次提交，观测图和地形查询也随之回退到各自默认关闭。可用 `MOON_MEASUREMENT_GRAPH` 和 `MOON_BATCHED_TERRAIN` 单独覆盖观测开关。

```bash
./run_exo_jump.sh
./run_exo_jump.sh --assist-force 0
./train_exo_jump.sh --num-envs 1024 --iterations 200
```

外骨骼训练结果保存在 `logs/cpu_exo_train`，默认从归档的前进跳跃策略 `model_104.pt` 初始化。保留原来的速度跟踪、跳跃、落地和足底载荷奖励，降低人体主动关节的绝对机械功、归一化力矩和下肢峰值力矩。支杆在对应靴底承重且身体正在上升蹬伸时，按输出的正机械功和分担比例获得奖励。训练报告记录人体关节机械功、下肢力矩和支杆做功量。以下命令在固定地形上使用同一归档检查点、同一种子和 1.0 m/s 指令，输出两份对照报告。

```bash
./run_exo_jump.sh --checkpoint bests/forward_jump_boot_yaw_2026-11-18/model_104.pt --headless --seconds 20 --seed 42 --assist-force 0 --report outputs/exo_off.json
./run_exo_jump.sh --checkpoint bests/forward_jump_boot_yaw_2026-11-18/model_104.pt --headless --seconds 20 --seed 42 --assist-force 500 --report outputs/exo_on.json
```

## 原 CUDA / Playground 环境与运行

以下为原 CUDA/REAF 入口说明，使用 `./run_playground.sh`；Mac CPU 回放请使用上面的 `./run.sh`。

共享环境已安装在 `~/dev-envs/robotics-playground-py312`，使用本机 `~/Github/mujoco_playground` 与 vendored REAF-sim 可编辑源码。Torch2.12/CUDA13/TorchVision及其依赖通过硬链接复用，缺失普通依赖优先从清华镜像安装；MuJoCo4开发版来自仓库指定的官方源。原 MJLab/Isaac 环境未修改。

`run_playground.sh` 和 `train.sh` 默认使用此解释器。也可 `source ~/dev-envs/robotics-playground-py312/bin/activate`。版本、约束和检查记录见 `~/dev-envs/robotics-playground-py312.{freeze,constraints}.txt` 与 `~/dev-envs/diagnostics/playground/`。

```bash
./train.sh --overrides='{"max_iterations":300}'
./run_playground.sh --checkpoint=/absolute/path/model_300.pt
# 仅在明确需要把旧策略用于当前地形时：
./run_playground.sh --terrain-transfer --checkpoint=/absolute/path/model_500.pt
```

`./run_playground.sh --help` 查看完整常用参数及示例，不加载仿真。推荐连字符写法，兼容旧名称：

| 推荐参数 | 兼容旧名称 |
| --- | --- |
| `--checkpoint` | `--resume-path`、`--resume_path` |
| `--num-envs` | `--num_envs`、`--nworld` |
| `--env-overrides` | `--env_overrides` |
| `--env-id` | `--env_id` |
| `--zero-actions` | `--zero_actions` |
| `--hot-reload` | `--hot_reload` |
| `--wp-kernel-cache-dir` | `--wp_kernel_cache_dir` |

`--viewer`、`--viz`、`--device`、`--gfx` 名称保持不变，其他上游参数原样透传。`--overrides` 修改 runner，`--env-overrides='{"pushes":false}'` 修改任务。`run.sh` 默认选本项目最新非smoke日志中编号最大的模型，无结果则明确报错。已复制的 v15 groundsync1/model_200 在本机已严格加载成功，包含原网络和观测归一化参数；原文件未修改。其源地形快照缺失，导入清单明确标记 `exact_replay=false`，默认回放会打印当前 detail3 地形迁移说明。其他版本检查点仍需分别验证，不能把 `--terrain-transfer` 当作格式转换。

## 验证范围与限制

```bash
~/dev-envs/robotics-playground-py312/bin/python launch.py check
```

使用新共享环境执行 MuJoCo4 CPU 静态编译、运动学/传感器计算与张量方法检查；不调用 `mj_step`、GPU、训练或回放。报告 `outputs/static_check.json`。该张量方法测试使用明确的 REAF 桩。另已通过真实 Playground/REAF 导入、任务注册/配置、MuJoCo4 原生场景静态编译及上游 train/play CLI 导入，记录在共享环境 diagnostics/playground/validation.json；后续按用户本次明确授权，已完成真实 GPU 单世界无扰动策略闭环5秒/300步，0失败复位、无NaN（outputs/playground_play/check_model200.json），并已修复、启动开发版 Studio 原生窗口回放。未训练；有限短测不代表所有地形已稳定。

当前支持默认 v15；旧 v1–v14/LQR/跳跃实验和 Isaac Sim 脚本只保留在参考快照中，未声称全部改写。当前 Playground 原生 viewer 显示局部物理场景；原共享2km月面多人skin显示、Isaac导出和 sim2sim 尚未接入新运行入口。人体/skin/地形资产已保留。MuJoCo3.11→4.x 数值行为需另行验证，不宣称策略等价、站稳或性能提升。

一次性 `scripts/export_source.py` 使用旧环境静态导出原项目真实 Scene（无动力学推进），生产运行不再需要源项目。导出来源和 Playground commit 记录在 `assets/playground/metadata.json` 与 `manifest.json`。XML 数值经过 MuJoCo 序列化；高度场数组、尺度和地面偏移从精确 NPZ 恢复，其余 XML 舍入误差尚未做全模型跨版本审计。

## 当前直接回放

回放默认使用 **16×16m** 地形 bank，省略编号时随机选择，`--terrain-idx 1–64` 固定选择其中一块；所有人物共享所选地形。

```bash
./run_playground.sh --viewer=classic --viz=physx --num-envs 4 --terrain-idx 12
```

运行中按 `[` / `]` 切换上一块/下一块地形（1与64循环），自动重置全部人物并复用已加载的高度场/渲染资源，当前编号打印到终端。暂停时也可切换，切换后保持暂停；Studio输入控件激活时不触发。

`--terrain-size=100` 可显式选择独立100×100m地形（1001×1001、0.1m采样）；100m不使用1–64编号或这两个地形切换键。100m资产来自已有Site20 DEM与相同细节算法的新裁剪，不拉伸旧地形；原资产与检查点不变，回放打印源/执行地形SHA及非精确迁移说明。训练默认仍为原64块16m bank。

Studio 和 classic 回放默认按墙钟 **1× 实时**限速（60Hz 控制对应每秒最多约 60 个控制步），暂停和重置后重新对时，不积攒追赶步。Studio 的 Speed 控件仍可调整目标倍率。计算或渲染不足时允许慢于实时，不跳物理步；训练不受此限速影响。

```bash
cd ~/myWorks/exolimb_moon_mujoco_playground
./run_playground.sh
```

默认优先最新原生训练 run 的最大编号模型，没有训练结果时兜底已复制的 `logs/2026-11-15_import_mjlab_v15_groundsync1/model_200.pt`，单人、关闭扰动，直接打开 **开发版 MuJoCo Studio 原生 OpenGL 窗口**。不默认浏览器或独立simulate程序。

`src/studio_native.py` 是项目内对当前固定开发版的适配：传递原生窗口句柄、通过mjrf空target直接显示、同步ImPlot上下文；共享MuJoCo/REAF文件保持不变。已实际渲染超过120帧且继续策略回放。当前原生适配针对OpenGL；不宣称Vulkan已验证。日志 `outputs/playground_play/run_model200_studio.log`，进程记录 `outputs/playground_play/viewer.pid`。旧web及失败日志保留作审计。

## Studio 地形同步修复

已参考本机 MuJoCo Studio/rendering 技能，修复 mocap 地形在开发版 Filament 下不可见及reset模型字段未同步的问题。物理仍在原GPU mocap高度场执行；独立CPU显示副本将相同的64块原生hfield作为world下无碰撞渲染资源，专门TerrainSnapshot同步显示选择，不在每次reset重建渲染场景。默认相机拉近到5m。

已检查原生Studio在多次reset后的实际画面：`outputs/playground_play/studio_audit/scene_after_resets.png`。该历史验收使用局部16m原生地形；100m新地形目前仅通过CPU静态检查，尚无动态回放验收。CPU静态几何回归：`tests/test_display_terrain.py`。当前日志：`outputs/playground_play/run_model200_studio_terrain_snapshot.log`。设置`STUDIO_RENDER_AUDIT=1`可在诊断时保存同相机原生读回，正常运行不生成截图。

## 月面站立训练任务

回放的窗口与外观独立选择：`--viewer=classic|studio`、`--viz=physx|skin`，默认`studio + skin`。`physx`按用户命名表示物理代理几何显示，动力学后端仍为MuJoCo/Warp；`skin`显示完整航天服蒙皮和贴图。四种组合都复用相同策略、月面和物理，仅显示副本不同：

```bash
./run_playground.sh --viewer=classic --viz=physx
./run_playground.sh --viewer=classic --viz=skin
./run_playground.sh --viewer=studio --viz=physx
./run_playground.sh --viewer=studio --viz=skin --num-envs=8
```

多人体显示模型只保留一套高度场 bank 和一套场景灯光（1 盏 sun，头灯仍为场景单份），并共享人体静态 mesh、贴图和材质；仅姿态、骨骼绑定和蒙皮变形数据按人体独立保留。

Studio 和 classic 多环境回放默认在同一块地形上显示全部人物（独立物理 world，人物间不相互碰撞），所有 world 使用同一个固定地形块；不做八块地形平铺，启动日志打印 requested_worlds 和 simulated/displayed 数量。回放环境数支持 `./run_playground.sh --viewer=studio --num-envs 8`（等价于`--nworld=8`，两者不要同时传）。回放默认全体共享本次会话首次抽取的固定月面块，Reset只重置人体的出生位置/朝向和状态；人物是独立物理 world，不互相碰撞。单世界需在 Reset 换地形时可用 `--env-overrides='{"resample_terrain":true,"shared_terrain":false}'`；多世界共享显示要求各 world 地形一致，不能关闭 shared_terrain。训练仍独立随机换地形。Studio的C/F/H及0–5显示快捷键可在Reset按钮取得焦点后继续使用，文本输入和正在操作的控件仍保留键盘输入。

经典viewer默认开启真实接触可视化：红点为接触位置，青色箭头为求解器接触力（长度0.003米/牛顿，方向由原生接触坐标系转换到世界系）。使用窗口的Contact point/Contact force显示开关控制。数据从选中world的Warp原生contacts/efc_force读取，非显示模型重新求解；额外同步仅用于经典回放，不影响训练路径。

经典交互窗口可选 `./run_playground.sh --viewer=classic`，默认仍为开发版Studio（`--viewer=studio`）。经典窗口支持鼠标相机控制；双击选中身体后按住Ctrl拖拽鼠标，外力会施加到Playground实际仿真，和任务扰动相加。空格暂停/继续，R复位，N切换world（配合`--nworld=...`），Esc关闭。经典窗口的CPU模型仅用于显示/计算鼠标扰动力，策略与动力学仍由Playground执行；面板中的CPU控制量/物理选项编辑不回写GPU任务。两种窗口使用同样的检查点选择和地形显示同步。示例：`./run_playground.sh --viewer=classic --checkpoint=/absolute/path/model_300.pt`。

训练任务 `WangyuMoonStandLunarTrainV1` 在 `src/lunar_stand_training.py`，训练入口 `scripts/train_stand.py`。默认从头训练，复用现有共享环境；训练不加载 Studio、显示副本或相机。

```bash
cd ~/myWorks/exolimb_moon_mujoco_playground
./train.sh                            # 8192环境，2000次PPO更新
./train.sh --iterations=300            # 同样8192环境，先训练300次更新
./train.sh --num-envs=1024 --iterations=300
```

速度调优：默认保持原有工作量和 eager 策略；可用 `./train.sh --iterations=300 --compile-policy` 单独试验上游的 `torch.compile` default 模式，首次编译会增加启动时间，收益尚未验证。控制组和试验组应使用相同环境数、rollout、PPO、种子与任务设置；训练结束后运行 `~/dev-envs/robotics-playground-py312/bin/python scripts/compare_training_speed.py logs/控制组 logs/试验组`，默认排除前20轮并输出配置、MDP签名、采样/更新耗时与吞吐均值和中位数。改变环境数会改变每轮样本数及PPO批量，需同时比较样本效率和最终训练效果，不能仅按单轮时长判断。

沿用 MJLab 月面站立：detail3 的64块16m高度场、安全随机出生/朝向、23动作、146观测、10秒回合、60Hz策略/120Hz物理、重力1.635，无LQR/课程/稳定裕度项。训练默认水平Fx/Fy各±30N、Fz=0、额外力矩0，脉冲0.1–0.2秒、冷却2–4秒。

奖励率权重：alive=.2，torso_velocity=-4，torso_angular_velocity=-.2，centroidal_angular_momentum=-.01，joint_pose=-.25，action_change=-.002，joint_torques=-2e-6，airborne=-.2，gravity_tilt=0。REAF原生reward provider按控制dt积分。训练任务字典是本任务的权重来源。

PPO对齐原项目：32步/环境/更新；actor/critic各128×128 ELU及观测归一化；初始动作标准差.25；学习率3e-4，自适应KL=.01；gamma=.995、lambda=.95；5 epochs、4 minibatches；entropy=.001、clip=.2、max_grad_norm=1；每25代及结束保存。动作clip2在环境设备执行，上游REAF wrapper的clip保持None。

```bash
# 显式调整训练扰动；关闭则用 --nopushes
./train.sh --iterations=300 --push-force=200 --push-torque=50

# 旧模型仅初始化网络/归一化，重新开始优化器和迭代计数
./train.sh --iterations=300 \
  --init-from=logs/2026-11-15_import_mjlab_v15_groundsync1/model_200.pt

# 本训练任务的同MDP检查点：恢复actor/critic/优化器/迭代，再追加300次更新
./train.sh --iterations=300 --checkpoint=/absolute/path/model_300.pt

# CPU配置/原生模型编译检查：不分配仿真环境，不step、不训练、不创建run目录
./train.sh --dry-run
```

原始导入策略缺训练地形快照，初始化时打印当前detail3迁移说明；不同地形的其他模型需显式 `--terrain-transfer`。完整续训校验任务MDP签名，不能把旧MJLab检查点作为本任务原样续训。续训的环境回合重新随机出生。

日志写入 `logs/<上海时间>_wangyu_moon_playground_stand_v1_detail3/`，含 `training_config.json`（PPO、奖励、来源、代码SHA）、`terrain/`（实际bank状态、资产清单）、`terrain_manifest.json`、TensorBoard、`model_*.pt` 和最终状态。训练产生本项目检查点后，`./run_playground.sh` 优先选最新原生训练run中编号最大的模型，避免同日import目录名遮住新结果；回放仍默认关闭扰动。

已通过CPU配置、网络严格加载、批量脉冲/部分复位和模型选择测试及dry-run。2026-11-15 已完成下述8192世界短训练性能验证；短测不代表策略收敛。

## 已验证的采样与日志优化

2026-11-15训练效果审计修复了两项迁移错误：单term观测会引用REAF可复用buffer，现由本地训练wrapper复制后返回，防止PPO在env.step后存入被覆盖的旧状态；终止函数现返回REAF的TERMINATE枚举，避免把布尔True当作TRUNCATE并错误执行跌倒bootstrap。MDP签名更新为`lunar_stand_v2_terminal_fix`，旧run不能直接`--resume_path`，可以显式`--init-from`仅迁移网络。默认从头训练使用修复语义。性能/收敛是独立指标；旧短测性能结果不能作为学习正确性证据。

`./train.sh` 默认启用五项局部优化：观测/奖励/地形纯张量 CUDA Graph、29点地形查询合批、仅复位world的鞋底覆盖检查、18项奖励日志合批，以及每rollout一次回合统计读回。全部物理子步、NaN/边界保护、随机扰动、PPO批量和奖励权重保持。完整回退：

```bash
MOON_FAST_TRAIN=0 ./train.sh --iterations=300
```

也可单独设置 `MOON_MEASUREMENT_GRAPH=0`、`MOON_BATCHED_TERRAIN=0`、`MOON_SPARSE_RESET=0`、`MOON_BATCHED_LOGS=0` 或 `MOON_ROLLOUT_LOGGER=0`。显式单项设置优先于总开关。开关及源代码SHA保存于run配置，普通Studio入口不自动启用新增的设备优化。

RTX5090同配置对照：8192×32样本/轮，120Hz物理/60Hz控制，5 epochs×4 minibatches，seed42，从头训练各40轮，排除前20轮。采样时间0.738→0.604秒；总吞吐32.63→39.22万步/秒；含日志和保存的墙钟吞吐32.17→38.79万步/秒（+20.6%）。PPO更新时间约0.065秒保持接近。详细配置、时间区间与结果在 `outputs/perf_deep/comparison.json`；各短测run位于同目录，不参与默认回放模型选择。没有GPU kernel trace，主机profile只用于找调用结构，不能视为单内核计时。

CPU等价测试16项通过。GPU测量图在三个状态与eager输出逐位一致，29点合批与原查询逐位一致，输出存储独立、RNG不被测量推进，注入NaN仍触发数值保护。两次PPO短训练均正常结束。跨运行的Warp轨迹不保证逐位一致，40轮性能对照不构成收敛效果验证。新增 `Perf/wall_cycle_fps` 包含前次保存、当前采样/学习/日志，比原 `Perf/total_fps` 更接近实际耗时。

多人体共享地形回放会筛选全脚底安全位置，随机初始化并按与已有人的最小距离分散出生点；部分 reset 避开其他人的当前位置。默认16m配置使用±5m出生、±6m边界、18m多人相机；显式100m配置使用±45m出生、±48m边界、150m多人相机。均保留脚底间隙验证和1m活动范围。只改变共享地形多人回放的出生采样，训练默认仍使用原采样。

## MuJoCo 4.0.1 Filament 与 USD 材质参考

`--viewer=studio --viz=skin` 使用本机4.0.1公共 `mjrf` / `ModelScene` 路径；只改独立CPU显示模型，GPU动力学与策略不变。原项目 `run_isaacsim.sh`、`scripts/preview_isaacsim.py` 和原始 `wangyu_rigged.usdc` 作为参考，USD图记录在 `assets/suit_pbr/usd_reference.json`。原参考项目及共享依赖保持不变。

- 衣服、手套/靴子使用原USD的BaseColor、Metallic、Roughness、Normal及UV，手套按原绑定共用Boot材质。颜色明确sRGB，数据贴图linear。网格逐件显色确认：可见正面面罩是`Suit_Shield`，`Glass`在内层被遮住。Shield还包含白色边框，使用原Helmet图集的金色区域遮罩设置金膜，边框保持非金属；外壳单独将原图集金色区域还原为白色非金属。转换脚本`prepare_visor_materials.py`和`assets/visor_pbr/manifest.json`记录来源SHA；不改UV/面/骨骼，不增加skin或重复人体几何。
- 月壤使用缓存LunarRegolith4k颜色/法线及打包ORM，4m重复，粗糙度影响.13、AO影响.35、法线强度.8/翻转V按参考脚本转换。转换脚本 `scripts/prepare_reference_pbr.py`，输入/输出SHA见 `assets/regolith_pbr/manifest.json`，不启动Kit。
- 使用一盏斜向太阳及一盏月面反弹IBL，全场共享；显式photometric模式关闭MuJoCo默认室内环境反射和头灯。太阳和IBL强度均80000，IBL像素已含月壤线性反照率×法线受光/π。`scripts/build_lunar_ibl.py`生成512像素六面体：黑天空、清晰地平线及4m重复、法线受光的月壤下半球，替代过宽的渐变；SHA与近似条件见`assets/lunar_ibl/manifest.json`。它是1.7m视点的共享平面近似，不包含周围人物或当前高度场的实时倒影；不把USD的1750及.53°太阳角声称逐项等价。SSAO开启、4096阴影图、bloom关闭。
- 当前安装的Studio C++ `ModelScene` 与公共 `mjrf` 在相同固定姿态审计中出图不同；项目内 `FilamentModelScene` 复用公共资源路径及native `mjv`诊断几何，用于原生窗口和静态审计。资源按模型加载创建，地形切换不重建；保留Studio UI和接触箭头等primitive叠加。文字标签和Flex调试层未在此适配中实现，本任务无Flex。

静态出图：`python scripts/audit_reference_render.py`（使用Playground解释器），只固定姿态渲染，不加载策略或推进物理。结果 `outputs/render_reference/reference_final.png` 和同名JSON明确记录4.0.1、零物理步及未做原生窗口动态验收。Classic不支持同等PBR/反射能力。

Studio界面使用项目内 `native_ui_bridge.py`，通过公开mjrf上传ImGui的交错顶点数据（position/UV/color）并重定位VtxOffset，修复本机原桥接的UV/color指针步长问题。只读draw-data读取器用已安装MuJoCo4.0.1 SDK/ImGui1.92.9b头文件及已有Torch的pybind11头编译，不改共享二进制、不安装依赖。当前工作区已构建；迁移/清理outputs后，用Playground解释器执行 `scripts/build_native_ui.py` 重建。`scripts/audit_native_ui.py` 只静态绘制界面，结果 `outputs/render_reference/ui_fixed.png`，不启动仿真或原生窗口。

面罩PBR校准参考[Filament官方材质模型](https://google.github.io/filament/main/filament.html)：金膜区域金属度1、感知粗糙度.04、金的法线入射反射色sRGB=(1,.85,.57)，金膜区域法线设平滑，使倒影呈镜面而非布料凹凸。当前MuJoCo默认将材质颜色按sRGB转线性，不重复gamma转换。`mjMaterial.reflectance`是平面反射混合量，不是Filament文档里的同名介电Fresnel参数，面罩设0，曲面反射走金属BRDF和IBL。本机4.0.1 `mjrfMaterial`未开放clearCoat/iridescence/透射参数，因此这是不透明金膜近似，不宣称薄膜光谱干涉或透明镀膜实现。

原始资源另核对了`~/myWorks/3d models/chinese-moon-spacesuit-empty-rigged-for-maya`：纹理包共19张，12张源图与项目缓存SHA一致；原包额外含Helmet_Opacity、Helmet_Refraction、烘焙bump和位移图，不能把它们自动当成金膜光学参数。解包只写入`outputs/original_spacesuit`，核对报告`source_audit.json`。最新零物理步静态图为`outputs/render_reference/visor_mirror_lunar.png`，此前`visor_pbr_*`仍有可见网格绑定错误，不作为修复后的证据。

可借鉴用户提供的[Three.js示例源码](https://github.com/mrdoob/three.js/blob/dev/examples/webgl_postprocessing_3dlut.html)：环境反射来自HDRI，LUT只做末端调色。`./run_playground.sh --viewer=studio --viz=skin --reflection-env royal`使用该示例同名Royal Esplanade的官方r180 HDR源图，离线转成线性RGB16F KTX六面体和GGX粗糙度mip；源URL/SHA及转换设置在`assets/reflection_reference/royal_esplanade.json`，脚本`prepare_reflection_environment.py`。这是参考环境模式，不把室内倒影冒充月面；默认`--reflection-env lunar`。也可传入自己的HDR预过滤KTX路径。KTX按上游pbr.xml以type=2d容器加载，实际六面体由KTX头定义，全场共享一次加载。参考近景`outputs/render_reference/visor_mirror_royal.png`可见完整弯曲建筑倒影。

衣服背光面的硬黑斑修复：此前PNG月面IBL只有镜面反射，且Royal转换KTX也没有`sh`元数据；MuJoCo仅在KTX含SH时调用Filament的`irradiance(3, ...)`，布料缺少月面反弹漫反射。两个环境现均保存线性HDR KTX与9个RGB辐照SH系数（包含余弦卷积及Lambert 1/π），无需额外头灯、复制光源或抬高曝光。月面反射仍保留原黑天空/地面分界，自阴影及太阳直射保留。CPU测试验证常量环境能量守恒、下半球照明方向和KTX元数据；固定姿态同相机前后图`suit_shadow_before.png`/`suit_shadow_irradiance.png`、另一侧`suit_shadow_fixed.png`，均0物理步。原始材质/模型/策略及训练逻辑未变，环境资产需重启窗口重新加载。

用户后续要求衣服更干净：`scripts/prepare_clean_cloth.py`从原始地图生成`assets/clean_cloth`显示派生图。原白布sRGB约213，低饱和明亮区域局部+22（上限245），保留底色中的缝线差异，红蓝/国旗/CMS标识颜色逐像素不变；白布和红织物金属度从原约.17改0、粗糙度至少.82，布料法线与平面法线混合强度.28，其他区域.55、头盔外壳法线.45。大褶皱几何与缝线保留，减弱原法线近乎掠射造成的黑色划痕。源图SHA和产物SHA均校验，原图不改；直接重指向已有共享texture资产，不增加纹理数量或复制每个人的资源，面罩镜面不变。静态同相机原图`cloth_source_comparison.png`、清洁版`cloth_clean_final.png`；`audit_reference_render.py --cloth-closeup --source-cloth`只用于静态A/B审计。
