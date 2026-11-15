# exolimb_moon_mujoco_playground

从 `../exolimb_moon_mjlab` 迁出的独立项目。项目目录已按用户要求修正为 `exolimb_moon_mujoco_playground`，Python 包名使用上游的 `mujoco_playground`。

使用本机当前 **MuJoCo Playground 1.0.1 主线：PyTorch + MuJoCo Warp + REAF**，不是 rss2025 JAX/MJX。新 `src/` 和入口没有 MJLab 依赖。原项目未修改，旧源代码留在 `reference/mjlab/` 供核对，未复制历史训练日志及检查点。

## 已迁移

- 默认 v15/detail3 月面任务；23 轴人体，固定头/腕，无额外根支撑；64 块 16m 原生高度场，保存原始 bank，避免运行时重新生成地形。
- 146 维 actor/critic 观测，保留原生 frame/touch/subtree 传感器及躯干 COM 局部速度；无稳定裕度或 LIPM 预测输入。
- 原奖励权重、动作 scale=.25/clip=2、关节到执行器映射、原生 PD、60Hz 控制/120Hz 物理、重力 1.635、10 秒回合、默认8192世界。
- 随机地形/安全出生点/朝向、全鞋底覆盖余量、1m 活动范围及60度跌倒终止；训练默认水平扰动，回放默认关闭。
- Playground 原生 REAF Device/TaskLayer 与 RSL-RL train/play 入口。新日志独立保存，检查点显式核对导出地形 SHA；不匹配需显式 `--terrain-transfer`。

## 环境与运行

共享环境已安装在 `~/dev-envs/robotics-playground-py312`，使用本机 `~/Github/mujoco_playground` 与 vendored REAF-sim 可编辑源码。Torch2.12/CUDA13/TorchVision及其依赖通过硬链接复用，缺失普通依赖优先从清华镜像安装；MuJoCo4开发版来自仓库指定的官方源。原 MJLab/Isaac 环境未修改。

`run.sh` 和 `train.sh` 已默认使用此解释器，无需设置环境变量。也可 `source ~/dev-envs/robotics-playground-py312/bin/activate`。版本、约束和检查记录见 `~/dev-envs/robotics-playground-py312.{freeze,constraints}.txt` 与 `~/dev-envs/diagnostics/playground/`。

```bash
./train.sh --overrides='{"max_iterations":300}'
./run.sh --checkpoint=/absolute/path/model_300.pt
# 仅在明确需要把旧策略用于当前地形时：
./run.sh --terrain-transfer --checkpoint=/absolute/path/model_500.pt
```

`./run.sh --help` 查看完整常用参数及示例，不加载仿真。推荐连字符写法，兼容旧名称：

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
./run.sh --viewer=classic --viz=physx --num-envs 4 --terrain-idx 12
```

运行中按 `[` / `]` 切换上一块/下一块地形（1与64循环），自动重置全部人物并复用已加载的高度场/渲染资源，当前编号打印到终端。暂停时也可切换，切换后保持暂停；Studio输入控件激活时不触发。

`--terrain-size=100` 可显式选择独立100×100m地形（1001×1001、0.1m采样）；100m不使用1–64编号或这两个地形切换键。100m资产来自已有Site20 DEM与相同细节算法的新裁剪，不拉伸旧地形；原资产与检查点不变，回放打印源/执行地形SHA及非精确迁移说明。训练默认仍为原64块16m bank。

Studio 和 classic 回放默认按墙钟 **1× 实时**限速（60Hz 控制对应每秒最多约 60 个控制步），暂停和重置后重新对时，不积攒追赶步。Studio 的 Speed 控件仍可调整目标倍率。计算或渲染不足时允许慢于实时，不跳物理步；训练不受此限速影响。

```bash
cd ~/myWorks/exolimb_moon_mujoco_playground
./run.sh
```

默认优先最新原生训练 run 的最大编号模型，没有训练结果时兜底已复制的 `logs/2026-11-15_import_mjlab_v15_groundsync1/model_200.pt`，单人、关闭扰动，直接打开 **开发版 MuJoCo Studio 原生 OpenGL 窗口**。不默认浏览器或独立simulate程序。

`src/studio_native.py` 是项目内对当前固定开发版的适配：传递原生窗口句柄、通过mjrf空target直接显示、同步ImPlot上下文；共享MuJoCo/REAF文件保持不变。已实际渲染超过120帧且继续策略回放。当前原生适配针对OpenGL；不宣称Vulkan已验证。日志 `outputs/playground_play/run_model200_studio.log`，进程记录 `outputs/playground_play/viewer.pid`。旧web及失败日志保留作审计。

## Studio 地形同步修复

已参考本机 MuJoCo Studio/rendering 技能，修复 mocap 地形在开发版 Filament 下不可见及reset模型字段未同步的问题。物理仍在原GPU mocap高度场执行；独立CPU显示副本将相同的64块原生hfield作为world下无碰撞渲染资源，专门TerrainSnapshot同步显示选择，不在每次reset重建渲染场景。默认相机拉近到5m。

已检查原生Studio在多次reset后的实际画面：`outputs/playground_play/studio_audit/scene_after_resets.png`。该历史验收使用局部16m原生地形；100m新地形目前仅通过CPU静态检查，尚无动态回放验收。CPU静态几何回归：`tests/test_display_terrain.py`。当前日志：`outputs/playground_play/run_model200_studio_terrain_snapshot.log`。设置`STUDIO_RENDER_AUDIT=1`可在诊断时保存同相机原生读回，正常运行不生成截图。

## 月面站立训练任务

回放的窗口与外观独立选择：`--viewer=classic|studio`、`--viz=physx|skin`，默认`studio + skin`。`physx`按用户命名表示物理代理几何显示，动力学后端仍为MuJoCo/Warp；`skin`显示完整航天服蒙皮和贴图。四种组合都复用相同策略、月面和物理，仅显示副本不同：

```bash
./run.sh --viewer=classic --viz=physx
./run.sh --viewer=classic --viz=skin
./run.sh --viewer=studio --viz=physx
./run.sh --viewer=studio --viz=skin --num-envs=8
```

多人体显示模型只保留一套高度场 bank 和一套场景灯光（1 盏 sun，头灯仍为场景单份），并共享人体静态 mesh、贴图和材质；仅姿态、骨骼绑定和蒙皮变形数据按人体独立保留。

Studio 和 classic 多环境回放默认在同一块地形上显示全部人物（独立物理 world，人物间不相互碰撞），所有 world 使用同一个固定地形块；不做八块地形平铺，启动日志打印 requested_worlds 和 simulated/displayed 数量。回放环境数支持 `./run.sh --viewer=studio --num-envs 8`（等价于`--nworld=8`，两者不要同时传）。回放默认全体共享本次会话首次抽取的固定月面块，Reset只重置人体的出生位置/朝向和状态；人物是独立物理 world，不互相碰撞。单世界需在 Reset 换地形时可用 `--env-overrides='{"resample_terrain":true,"shared_terrain":false}'`；多世界共享显示要求各 world 地形一致，不能关闭 shared_terrain。训练仍独立随机换地形。Studio的C/F/H及0–5显示快捷键可在Reset按钮取得焦点后继续使用，文本输入和正在操作的控件仍保留键盘输入。

经典viewer默认开启真实接触可视化：红点为接触位置，青色箭头为求解器接触力（长度0.003米/牛顿，方向由原生接触坐标系转换到世界系）。使用窗口的Contact point/Contact force显示开关控制。数据从选中world的Warp原生contacts/efc_force读取，非显示模型重新求解；额外同步仅用于经典回放，不影响训练路径。

经典交互窗口可选 `./run.sh --viewer=classic`，默认仍为开发版Studio（`--viewer=studio`）。经典窗口支持鼠标相机控制；双击选中身体后按住Ctrl拖拽鼠标，外力会施加到Playground实际仿真，和任务扰动相加。空格暂停/继续，R复位，N切换world（配合`--nworld=...`），Esc关闭。经典窗口的CPU模型仅用于显示/计算鼠标扰动力，策略与动力学仍由Playground执行；面板中的CPU控制量/物理选项编辑不回写GPU任务。两种窗口使用同样的检查点选择和地形显示同步。示例：`./run.sh --viewer=classic --checkpoint=/absolute/path/model_300.pt`。

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

日志写入 `logs/<上海时间>_wangyu_moon_playground_stand_v1_detail3/`，含 `training_config.json`（PPO、奖励、来源、代码SHA）、`terrain/`（实际bank状态、资产清单）、`terrain_manifest.json`、TensorBoard、`model_*.pt` 和最终状态。训练产生本项目检查点后，`./run.sh` 优先选最新原生训练run中编号最大的模型，避免同日import目录名遮住新结果；回放仍默认关闭扰动。

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

可借鉴用户提供的[Three.js示例源码](https://github.com/mrdoob/three.js/blob/dev/examples/webgl_postprocessing_3dlut.html)：环境反射来自HDRI，LUT只做末端调色。`./run.sh --viewer=studio --viz=skin --reflection-env royal`使用该示例同名Royal Esplanade的官方r180 HDR源图，离线转成线性RGB16F KTX六面体和GGX粗糙度mip；源URL/SHA及转换设置在`assets/reflection_reference/royal_esplanade.json`，脚本`prepare_reflection_environment.py`。这是参考环境模式，不把室内倒影冒充月面；默认`--reflection-env lunar`。也可传入自己的HDR预过滤KTX路径。KTX按上游pbr.xml以type=2d容器加载，实际六面体由KTX头定义，全场共享一次加载。参考近景`outputs/render_reference/visor_mirror_royal.png`可见完整弯曲建筑倒影。

衣服背光面的硬黑斑修复：此前PNG月面IBL只有镜面反射，且Royal转换KTX也没有`sh`元数据；MuJoCo仅在KTX含SH时调用Filament的`irradiance(3, ...)`，布料缺少月面反弹漫反射。两个环境现均保存线性HDR KTX与9个RGB辐照SH系数（包含余弦卷积及Lambert 1/π），无需额外头灯、复制光源或抬高曝光。月面反射仍保留原黑天空/地面分界，自阴影及太阳直射保留。CPU测试验证常量环境能量守恒、下半球照明方向和KTX元数据；固定姿态同相机前后图`suit_shadow_before.png`/`suit_shadow_irradiance.png`、另一侧`suit_shadow_fixed.png`，均0物理步。原始材质/模型/策略及训练逻辑未变，环境资产需重启窗口重新加载。

用户后续要求衣服更干净：`scripts/prepare_clean_cloth.py`从原始地图生成`assets/clean_cloth`显示派生图。原白布sRGB约213，低饱和明亮区域局部+22（上限245），保留底色中的缝线差异，红蓝/国旗/CMS标识颜色逐像素不变；白布和红织物金属度从原约.17改0、粗糙度至少.82，布料法线与平面法线混合强度.28，其他区域.55、头盔外壳法线.45。大褶皱几何与缝线保留，减弱原法线近乎掠射造成的黑色划痕。源图SHA和产物SHA均校验，原图不改；直接重指向已有共享texture资产，不增加纹理数量或复制每个人的资源，面罩镜面不变。静态同相机原图`cloth_source_comparison.png`、清洁版`cloth_clean_final.png`；`audit_reference_render.py --cloth-closeup --source-cloth`只用于静态A/B审计。
