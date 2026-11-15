# 项目约定

- 迁移自 ../exolimb_moon_mjlab，当前后端是 MuJoCo Playground 1.0.1 主线 PyTorch / MuJoCo Warp / REAF。目录已按用户最新要求修正为 exolimb_moon_mujoco_playground。
- 只授权代码和静态检查；未经新的明确请求，不启动训练、动态仿真、回放、评估或 GPU 基准。
- 遵守 ../AGENTS.md 的共享环境规则。Playground 依赖与 robotics-mjlab-py313 不兼容，不修改该环境，不拼接不同环境 site-packages。
- 新入口 src/、launch.py 不得导入 MJLab。scripts/export_source.py 是一次性源项目静态导出工具，明确例外。
- 原项目/原检查点不得更改。reference/mjlab 是历史代码快照，不是生产运行路径；其 AGENTS.md 仅适用于该子树。
- 默认 v15 月面、146 观测、23 动作、8192 世界、10 秒、60Hz 控制/120Hz 物理、重力 1.635。保留固定头/腕、原生传感器与动力学、无 LQR/课程/稳定裕度奖励和观测。
- 训练扰动默认开启，回放默认关闭。不同后端 MuJoCo 版本可能改变数值行为，不宣称策略或训练逐位等价。
- 以 assets/playground 中导出的模型、bank、参数及 SHA 为依据；不得静默重新生成月面或混用检查点。静态通过不代表 GPU/REAF 集成或站稳验证通过。

- 2026-11-15用户明确要求复制原MJLab已训练策略并run：已复制修正地面同步后的v15 groundsync1/model_200（原文件不动），放在logs/2026-11-15_import_mjlab_v15_groundsync1。源缺训练地形快照，terrain_manifest.json明确exact_replay=false、源/执行地形与模型SHA；这是对当前detail3的策略迁移，不冒充同地形训练结果。
- 本次真实Playground/Warp单世界无扰动GPU闭环5秒/300步通过，0失败复位、无NaN；模型严格加载，未训练。检查报告outputs/playground_play/check_model200.json。此有限测试不构成全地形稳定性结论或后续新测试的持续授权。
- 原生窗口初次运行曾出现Target not found及shutdown错误，随后临时使用web；该历史默认已被下一条用户最新要求覆盖。
- 本地moon_viewer.py薄适配将每世界geom_dataid/geom_pos同步到CPU显示model，再让上游做原生mj_forward。上游默认只复制size/color等，遗漏这两项会显示错误地形；不要因此回改GPU碰撞字段或共享REAF源码。

- 用户最新明确要求只用开发版 Studio 原生窗口：./run.sh现在默认--gfx=opengl、MUJOCO_GL=glfw；不自动切到浏览器或独立simulate二进制。studio_native.py在进程内局部替换NativeViewer（共享安装文件不变），将native_window句柄传给mjrf.ContextConfig，用原生RenderRequest的空target直接送窗口swapchain，补Python ImPlot独立模块set_implot_context。复用上游其余Studio UI、输入、模型同步及策略插件。当前固定支持OpenGL窗口；不把离屏渲染伪装成原生窗口。
- 已实际启动model_200的原生Studio，日志确认1/120帧直接显示到2728x1818桌面窗口，随后仍持续回合reset，无Target错误或ImPlot崩溃。日志outputs/playground_play/run_model200_studio.log，PID见viewer.pid；未训练、未修改原策略/共享依赖。原web进程已关闭。

- 2026-11-15用户要求查看Playground的Codex技能：Playground本身提供~/Github/mujoco_playground/AGENTS.md，Studio/rendering技能实际在~/Github/mujoco/doc/skills/{studio,rendering}/SKILL.md，已阅读并应用。重点为sim/viewer独立副本、Snapshot/Event通信、原生Filament资源复用，不把状态快照等同模型参数同步。
- 地形黑屏已复现并修复：CPU静态A/B显示原地形挂mocap时不可见，原样放world后可见；仅改body_weldid分类的试验不够，生产实现用独立display_spec重新编译。GPU spec/模型/mocap地形/奖励/策略完全不动。显示副本保留原nq/nv/nbody/nmocap，64个固定hfield渲染geom挂world（无碰撞），按名字映射人体geom，原生mj_forward仅更新CPU显示。
- 模拟镜像模型字段不会被Studio默认StateSnapshot发送；moon_studio_plugins的TerrainSender/Receiver用专门TerrainSnapshot传active_field，在viewer线程切换group0/5并刷新缓存可见性。每块显示hfield保持固定dataid，资源一次加载；禁止恢复每reset重建ModelScene的方案，该原型在多次reset后曾在native render中段错误。
- 默认相机距人物5m。STUDIO_RENDER_AUDIT=1才保存30/600帧同相机原生读回；正常启动无截图文件I/O。实际原生Studio跨多次reset地形已目视确认，证据outputs/playground_play/studio_audit/scene_after_resets.png。CPU回归test_display_terrain.py通过，比较3块×3原生ray及人体姿态，显示几何与原碰撞表面一致、源模型不被修改。

- 用户要求仿照MJLab月面站立提供任务，由用户自己训练。本轮不启动训练或GPU仿真。新增lunar_stand_training.py/任务WangyuMoonStandLunarTrainV1与scripts/train_stand.py，./train.sh默认从头8192环境/2000次更新，32步、保存25次、128x128ELU/归一化、LR3e-4、entropy.001、gamma.995等对齐原MJLab。物理/146obs/23actions/9项奖励保持；默认30N水平脉冲/零额外torque，可显式调幅或--nopushes。
- 新训练CLI用absl：--iterations/--num-envs、--init-from仅恢复网络/归一化并清优化器/迭代、--resume_path恢复本任务同MDP的网络/优化器/迭代，不沿用上游train.py默认重置critic的resume语义。--dry-run只CPU编译/打印，不创建run，不reset/step/PPO。
- 每次训练新建独立run，保存真实state.npz/资产元数据、任务/PPO/MDP签名和源代码SHA；拒绝覆盖现有run。新checkpoint_selection优先本项目已保存的native训练结果，再兜底最初import；不要按同日字符串排序使import永远遮住数字时间run。
- 此任务尚无新的GPU训练验收，不把早先model_200单世界回放当成8192批训练通过。CPU测试覆盖配置、兼容网络、扰动不加Z/脉冲保持、部分reset不影响其他world、native模型选择；已有dry-run默认及旧模型初始化路径通过。

- 2026-11-15 后续用户反复要求全面深入优化训练，本次已做必要的有界前台性能验证：8192环境各40轮PPO基线/优化对照，以及固定策略采样短测；全部结束，不留下训练后台进程。结果outputs/perf_deep/comparison.json，实际墙钟吞吐约+20.6%，非收敛结论。上述“尚无新的GPU训练验收”由此更新。
- ./train.sh默认MOON_FAST_TRAIN=1，展开为measurement graph、batched terrain、sparse reset、batched reward logs、rollout logger五开关；设置0回退，单项显式环境变量优先。普通回放不默认启用新增设备优化。共享第三方源码不改，physics CUDA Graph不重复捕获；测量graph只含纯张量，保留所有子步测量与主机NaN保护、输出clone和storage变化重捕获。随机数数量/顺序、物理与PPO参数不变。

- 训练效果审计发现并修复两项迁移错误：REAF单term观测引用output buffer，必须在LunarTrainingVecEnv._get_flat_obs克隆，PPO才不会在env.step后存错旧状态；FunctionalTerminationChecker要求TerminationResult枚举，布尔True等于TRUNCATE而非TERMINATE。moon_task.failed现返回0/TERMINATE，MDP_REVISION=lunar_stand_v2_terminal_fix，拒绝旧MDP直接resume。不能用性能等价测试替代transition生命周期/终止bootstrap正确性测试。诊断与短测在outputs/learning_diagnosis，不参与自动checkpoint选择。

- 用户最新要求回放默认按墙钟实时：Studio/classic 共用本地 playback_clock，以 monotonic 墙钟和实际模拟时间增量限速，默认1×；Studio沿用Speed百分比，暂停/reset重新对时，慢帧不突发追赶、不跳物理步。仅回放启用，不限制训练；未因此启动新回放。

- 最新用户明确要求 `--num-envs 8` 显示一块地形上的8个人，不要8块地形平铺。回放CLI默认 shared_terrain=true；MoonDevice仅在此开关下固定会话首次选中的地形供全部world使用（训练默认false及原随机流保持）。Studio/classic多world显示不加平铺偏移，只显示world_0的一块active hfield，其他world地形副本隐藏；sync_grid拒绝不同碰撞地形冒充共享显示。人物仍独立world，无人物间碰撞；鼠标力按人体映射回所属world。
- Studio启动时序修复：不在空模型的第0帧初始化相机/分组，而在真实模型加载时初始化。TerrainSnapshot带revision，Sender持续发送至可靠TerrainAck，模型替换通过TerrainRequest重取；提前到达空模型的快照缓存到PostModelEvent再应用，避免Reset清空待处理快照后永久丢失。CPU通道/丢包/晚ACK/单地形8人及CLI检查通过，未启动新的动态回放，不能宣称偶发黑屏已有原生窗口验收。

- 多人体显示资源已去重：仅world_0保留64个hfield资产与64个固定terrain geom；其他world附件在编译前移除地形及引用它的CPU显示传感器，不再保留隐藏副本。人体静态mesh/material/texture统一引用world_0并删除重复资产，8人编译后nmesh/nmat/ntex/hfield_data尺寸与单人相同。骨骼绑定/变形skin缓冲仍逐人独立；sync_grid保留显示模型编译后的geom_dataid，不从源编号覆盖共享资源索引。CPU检查覆盖8人资源数量、逐人独立蒙皮位移与鼠标力映射；未新启动态回放。

- 多人体复制遗漏的sun已修复：actor-only附件删除lights，场景仅world_0保留一盏sun，不能随人数叠加光照。CPU回归同时覆盖classic+physx四人和skin八人，nlight与单人均为1；原生MjvScene连续切换0/21/63块后都仅1个可见hfield、1sun+1headlight。未启动动态回放，不将静态可视场景检查冒充实际窗口验收。

- 用户要求多人随机均匀分散：shared_terrain且nworld>1时，playback_spawns从既有bank在±5m内筛选随机朝向/小扰动的安全放置，使用局部原生三角坡度覆盖界和原全脚底gap阈值，随机首点后逐次最大化与已选/仍运行人体的最小距离。部分reset只更新指定world，训练/单人保留原采样及随机流。保留1m活动范围、±6m终止界，默认多人相机18m。不生成/修改地形；CPU对64块×8人固定种子检查最小间距约4.26m，非动态站稳结论。

- 用户最新要求100×100m地形：./run.sh默认--terrain-size=100，可显式16回到原bank；训练默认16不变。新增assets/playground_100m独立1001×1001、0.1m采样资产，由已有Site20 DEM与原细节算法在满足源边界的新中心裁剪生成，不拉伸16m，也不修改原资产/原检查点。metadata/manifest记录来源、生成代码及SHA，运行时只校验加载，不自动生成。执行100m时明确打印源/执行terrain SHA及旧策略迁移、非exact replay说明。
- 100m配置物理/显示/高度观测统一half=50，单个hfield资源供多world共享；安全分散出生约±45m，终止边界±48m，1m活动范围保留；多人相机150m。Studio相机待真实模型的首个有效StateSnapshot再初始化，避免100m随机出生远离空模型原点。CPU原生编译/射线与张量采样及4人体共享资源检查通过；未启动新动态回放或训练。

- 最新用户要求覆盖100m回放默认：./run.sh默认--terrain-size=16；新增--terrain-idx 1–64，用户编号减1映射bank内部索引，省略仍随机，100m与编号参数互斥。固定编号通过moon_env/MoonDevice传递，重复/部分reset仍保持该块；训练默认参数签名不增加此键。100m资产及显式--terrain-size=100保留。
- 用户要求运行中键盘切换：Studio/classic用 [ / ] 在1–64循环前后切换并完整重置全部人物，保持暂停状态；复用原模型、高度场与渲染资源。MoonDevice.select_terrain清除首次固定patch/初始化mask后由正常env reset切换实际碰撞字段，不能仅改CPU显示。Studio用可靠TerrainCycle Event交给sim线程处理并dispatch ResetEvent，Viewer线程不运行physics/reset；活跃输入控件禁用快捷键。CPU检查覆盖1/64映射、参数越界/100m互斥、连续事件/首尾循环、多人碰撞地形和无模型/资产重建；未启动新动态回放。

- 用户要求结合原run_isaacsim.sh、实际USD及MuJoCo4.0.1 Filament技能提升显示。已核实runtime mj_versionString=4.0.1，完整参考本机doc/skills/{rendering,studio}及rendering.rst/reflect_pbr.xml。原USD通过既有Isaac环境pxr只读解析、未启动Kit；BaseColor与现有文件SHA一致，补齐原Helmet/Boot/Suit三套Metallic/Roughness/Normal，并恢复面罩原Helmet纹理绑定；手套原本共用Boot材质，不能凭名称另造贴图。
- 显示模型使用assets/suit_pbr与regolith_pbr本地缓存/SHA。月壤4m重复，转换参数记录来源；颜色sRGB、数据linear。显式光度太阳80000+月面低频IBL50000、头灯0，不使用内置室内IBL，面罩不能出现室内建筑反射。IBL为黑天空/灰地面近似，不冒充RTX周围几何完整反射；不同引擎强度单位、太阳角与BRDF不声称逐位一致。只改CPU显示，不改GPU场景/检查点/训练/源项目/共享安装。
- 本机native._viewer.ModelScene和公共mjrf的同姿态静态结果存在差异，且公共ModelDecorations.update绑定暴露raw struct类型不能直接接受Python wrapper。项目内filament_model_scene使用公共ModelScene长期保留资源，native mjv派生与复用primitive overlay池，供Studio原生适配及同类静态审计共同使用；文字label/Flex overlay未覆盖（本任务无Flex）。导入mujoco.studio只注册filament:打包shader provider，不创建窗口。单人显示同步不得用GPU light_*覆盖显示专用太阳/IBL；多人体共享纹理/材质/场景光源，保留场景级未加prefix的filament.*数值选项。
- outputs/render_reference/reference_final.png为固定姿态mjrf静态渲染（0物理步、非原生窗口动态验收）。未启动新回放、Kit或训练；不要将静态图说成动态稳定/原生窗口已验收。早先before/debug/pure等图片是排障历史，最终图以reference_final及JSON为准。

- 用户纠正仅Glass面罩有镀层，头盔外壳没有：已再次只读检查原USD九个Mesh的material binding（无GeomSubset）及skin UV。Helmet图集被多个部件共用，不等同于每个部件都应为金属；按用户要求skin_mat_0独立金色镀层，skin_mat_1外壳/框架强制metallic=0并解除金属度图，保留原颜色/法线/粗糙度。此前“面罩恢复同一Helmet材质”的约定由此覆盖。
- UI缺失定位到本机ImGui-Filament桥接VtxBuffer.Data按ImDrawVert步进而不是字节步进的UV/color偏移问题。用项目native_ui_bridge.py+只读draw_data.cc读取实际ImGui buffer，以公共mjrf设置正确交错attribute及U32重定位VtxOffset、clip/DPI、font纹理、blend_order和mesh/Renderable复用；Shader使用UI而非PBR。原生窗口与物理不重建，共享安装不改。编译器只用现有4.0.1 SDK(Imgui1.92.9b)/Torch pybind头，build_native_ui.py产物outputs/native_ui/_moon_draw_data*.so；不链接/复制另一套ImGui。ABI probe与>65535顶点/非零VtxOffset CPU测试通过，UI静态图ui_fixed.png完整显示文字控件；需原生窗口动态复核，不据静态demo宣称所有Studio面板已实测。

- 用户明确要求上网核对PBR与原始3D资源：查阅Filament官方材质模型及MuJoCo rendering文档；原始纹理rar只解包至outputs/original_spacesuit，19张图中现用12张SHA与原包一致，另有Helmet_Opacity/Refraction及烘焙bump/位移图，不能把它们冒充金膜光学参数。面罩改为金F0 sRGB=(1,.85,.57)、metallic=1、perceptual roughness=.08；MuJoCo reflectance是平面反射混合量，面罩设0，不将它当Filament dielectric Fresnel。4.0.1公开mjrfMaterial无clearCoat/iridescence/透射控制，因此当前是厚金膜PBR近似，无薄膜干涉或实时人物倒影。
- 新assets/lunar_ibl由scripts/build_lunar_ibl.py静态生成：黑天空/真实月壤albedo平面反弹，256px六面体，4m纹理重复、1.7m视点、albedo*cos45/pi，IBL与sun同强度80000；代替此前低频sigmoid渐变与IBL50000。全场只一份共享IBL，源及产物SHA校验，不能宣称当前高度场的精确倒影。固定姿态零物理步证据outputs/render_reference/visor_pbr_lunar.png，8项相关CPU检查通过；未启动回放、训练、Kit或原生动态验收。

- 用户指出面罩应为镜面后，逐skin静态显色定位到此前错误：可见正面是Suit_Shield（还含白色边框/附件），Glass是被遮住的内层。此前只把skin_mat_0改为金属并强制整个skin_mat_1非金属，导致可见面罩仍是金色漫反射。现按Helmet图集金色岛生成独立visor_color/metallic/roughness/normal，Shield使用skin_mat_visor，金膜metallic=1/roughness=.04/平滑法线，边框保持非金属；外壳材质将原金色岛重映射白色/roughness .45。转换脚本prepare_visor_materials.py，来源与产物SHA在assets/visor_pbr/manifest.json；不改原资源、UV、面、骨骼、skin数量或人体几何，仅新增共享材质/纹理。此条覆盖此前“仅Glass是可见面罩”的错误判断。
- 默认lunar IBL升级512px并包含月壤法线受光，仍是共享静态平面近似。用户Three.js 3DLUT示例已读取源码，实际反射由HDRI scene.environment产生；新增--reflection-env royal或KTX路径，仅Studio+skin支持，默认lunar。Royal来自Three.js官方r180同名HDR源，离线CPU转线性RGB16F KTX和GGX粗糙度mip，保留HDR>1，不应用LUT；source URL/SHA在assets/reflection_reference/royal_esplanade.json。KTX按上游pbr.xml用type=2d blob包装，六面体由文件头定义，不能强制mjTEXTURE_CUBE（本机编译器会错误要求3通道）。多人共用资源不变。
- 最新固定姿态零物理步证据visor_mirror_lunar.png（黑天空/月壤镜面）和visor_mirror_royal.png（完整建筑倒影），此前visor_pbr_*仍有正面绑定错误，不能作为已修复证据。31项相关CPU检查通过；未启动回放、训练、Kit或动态原生窗口。诊断棋盘/单mesh显色图只用于辨认反射通道与遮挡网格，不作为最终材质。

- 用户要求改善衣服阴影/暗区：根因为PNG月面IBL及原Royal转换KTX仅有反射，缺失漫反射辐照SH；ModelLights只有读取KTX SH后才调用Filament irradiance(3,sh)。现两个环境保存线性RGB16F KTX+9个RGB SH，按官方IndirectLight/CubemapSH多项式预缩放、余弦卷积和Lambert 1/pi。lunar.ktx替代生产PNG引用，PNG仅预览；源/产物SHA保持校验。衣服法线/几何/太阳强度/灯数量/曝光未变，未通过删除纹理或增加头灯掩盖黑斑。CPU验证常量辐射能量守恒、地面半球方向、HDR与SH元数据；静态固定姿态suit_shadow_before/irradiance同相机及另一侧suit_shadow_fixed证实暗部恢复连续皱褶。未启动动态回放、训练、Kit或改共享依赖。

- 用户继续要求消除衣服脏污观感：静态图检查原Suit白布albedo均匀约213，黑划痕主要来自过强法线，布料原金属度约43/255。新增prepare_clean_cloth.py与assets/clean_cloth派生显示图，白布低饱和区域局部+22 sRGB（上限245），红蓝/国旗/CMS色彩不改；白布/红织物metallic=0、roughness>=209/255，布料法线向平面混合.28、其他区域.55、头盔外壳法线.45。保留原几何、缝线、UV与骨骼及面罩镜面，不修改原资产/策略/共享环境。生产直接重指向既有texture文件，不增加纹理数量，多人继续共享一次加载；manifest校验源和派生SHA。静态同相机cloth_source_comparison.png/cloth_clean_final.png；新增白布金属度/粗糙度、法线倾角降低及彩色标识逐像素保真CPU检查。未启动回放、训练或Kit。此条覆盖此前“衣服法线未变”的显示外观约定。

- 2026-11-15 当前用户明确要求克隆到 Apple Silicon Mac 并用 CPU MuJoCo 运行。新增 run_cpu.sh / scripts/play_cpu.py，使用隔离 .venv-cpu、MuJoCo4.0.1 原生 mj_step 和 CPU Torch 策略；复用 moon_device_core.py 的原有张量逻辑，moon_device.py 保留 REAF 适配。原策略/资产不变，不安装或改动共享环境。CPU classic 窗口已启动并目视确认，5秒300控制步闭环无失败/非有限值。不把原生CPU回放当作Warp轨迹等价或训练收敛证据。

- 当前用户要求修复 CPU run.sh 对 --viewer studio --viz skin 的拒绝。CPU入口现支持 classic/studio × physx/skin，单人；cpu_viewer.py 用独立显示模型复用原skin/PBR和固定高度场bank，物理仍原生CPU mj_step。macOS Studio必须用普通Python主线程创建原生Window，classic仍用mjpython；studio_native.install(ui_bridge='native')使用安装包自带UI bridge（macOS已目视确认文字/控件正常），原REAF默认local桥接保留。不得将仅参数接受当作原生窗口验收。

- 用户要求 CPU --viewer studio --viz skin --num-envs 8：CpuWorlds 使用独立MjData、共享相同地形/物理模型和CPU批量策略，复用原安全分散出生与独立部分复位。CPU显示端复用build_grid/sync_grid，8人仅一套地形/纹理/网格/光源。单人限制由本条覆盖，原生CPU不改回Warp、不启动训练。

- 用户要求Mac键鼠施力：CPU显示端现在将Studio App计算的xfrc或classic原生扰动力按body/world映射回CpuWorlds.mouse_forces，并在每个物理子步加到任务外力；松手后清零，多人体只作用于选中world。Studio按R也接入复位队列。动态回归验证100N仅改变选中world、松手无残留，以及8人body映射；尚未声称已用真实鼠标拖拽验收。

- 用户要求恢复跟踪/退出跟踪/键盘切换当前人体。CPU classic/Studio共用CameraControls：T跟踪开关、N/P循环人体、A总览、Esc自由视角，Studio额外Shift+N。Studio用CpuApp优先处理，避免T透明/N岛/P接触分割/A自动连接冲突；文本编辑不拦截。选择人体映射到真实display body并清除旧鼠标外力；跟踪中切换保持跟踪，自由模式切换只对焦。Mac原生Studio实按T/N/P/Shift+N/A/Esc及状态栏已验收。

- 用户纠正必须使用原标点键位，覆盖刚新增的T/N/P方案：`,`上一人体、`.`下一人体并按原SharedMoonViewer语义进入近距跟踪，`/`开关跟踪，A原总览，Esc自由视角；`[`/`]`仍按原项目切换地形。T/N/P撤回自定义覆盖，不设相机别名。README/CLI/状态栏/测试统一。

- 用户要求各种显示配置交互统一，明确包含physx。CpuDisplay.handle_key共用标点键路由；update_mouse_interaction在classic/Studio均直接调用原生mjv_applyPerturbForce并统一选择/外力world映射，classic也同步鼠标选中的人体及HUD。修复主入口无条件启用SKIN标志。四组合各8人原生窗口集成检查通过（按键输入层注入，不是本轮真实OS按键验收），涵盖跟踪、切人、总览、地形键、选中world施力和松手清零；统一控制器覆盖单人/8人各显示组合。

- 用户报告skin双击错选/偏移：复现原Studio ux.Pick在29个可见skin点中24个错选/漏选；上游MakeSkin复用vertex_buffer且累加前未清零。新增skin_picking.NativeSkinPicker以原生mjv_updateScene/mjv_select获取正确变形顶点和body，再用mju_raySkin真实交点作为施力锚点；跟踪相机使用当前lookat的独立free副本，避免拾取推进平滑。仅当前CPU Studio进程的Python ux.Pick被替换，physx委托原函数，classic仍用原生mjv_select。单人/8人、多skin、遮挡、交点和相机不变测试通过；Studio原生窗口的App双击输入入口及实际localpos锚点集成检查通过，不冒充真实OS双击验收。

- 用户报告Studio间歇启动键鼠全失效但画面/steps继续，Classic正常；确认Studio用Command而Classic用Control。真实SDK诊断曾记录新Studio窗口120帧均无SDL键盘/鼠标焦点。新增studio_input：创建窗口前设置本进程SDL前台/首次点击穿透提示、禁用Control单击转右键；创建后仅一次RaiseWindow/SetWindowInputFocus，不持续抢焦点。新启动skin/physx集成日志focus=True。Studio Python鼠标输入接受Control或Command并在扰动时禁止同时移动相机；classic不改，两者可统一Control。原生窗口输入入口测试验证两修饰键产生相同外力，非真实OS组合拖拽验收；不承诺已排除所有偶发输入原因。

- 焦点请求进一步移到首个完整Studio渲染帧结束后（不是模型/纹理初始化之前），只执行一次；输入日志在SDL焦点变化时报告。skin/physx分别3秒正常CLI启动都观察到False→True。Classic未改输入后端；用户确认原Studio=Command/Classic=Control，当前统一Control且Studio兼容Command。

- 用户要求classic skin外观向Studio对齐，并补Studio双击选中高亮。classic_materials仅改classic显示spec：共享原RGB/UV/蒙皮，legacy RGB颜色空间/Phong高光与受支持的spot月面反弹补光，去掉不支持的KTX image light；源资产/CPU物理不改，PBR法线/环境镜面不能逐项等价。注意classic不支持mjLIGHT_POINT和IMAGE，补光必须SPOT；multiworld需复制headlight.active。SelectionHighlight在Filament renderable材质selected字段标当前人体黄色轮廓；按body_rootid只选一人，切换/取消清除旧轮廓，不改model.mat_rgba或共享贴图/物理。静态出图已目视确认黄色轮廓，8人真实Filament资源检查覆盖切换/取消/其他人体不高亮。

- 用户报告64人启动很慢/无法启动。分段计时确认CPU64环境<1秒，旧display_spec拼接/事后资产删除约34秒，compile约0.6秒、峰值约4GB。build_grid现先从actor-only小模板去除materials/textures/meshes，再attach并将新人体静态资产引用重定向world_0；保留独立skin/bones/全部64人。native dedup_assets作为最终保护；64人拼接实测约0.79秒，完整显示模型约1.34秒就绪，未降贴图/网格精度。启动进度分段输出。Studio+skin和classic+skin各64人原生窗口+6控制步短启动检查通过（384 transitions，无NaN/失败），不当作长时稳定或实时帧率验收。64人编译前共享资源回归及8人状态同步/部分复位回归通过。
