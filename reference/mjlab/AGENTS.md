# 本项目约定

- sim2sim首次用户运行现场：outputs/sim2sim_isaacsim/run-20261115-100323-c0cAXl/report.failure.json为NoneType.filter_count；startup /tmp/wangyu-run-init-pyumc7uf.log明确两Foot缺少PhysxContactReportAPI。RigidPrim(contact_filter_paths=...)并不自动启用报告，现由原生set_enabled_contact_tracking(True,threshold=0)在play/initialize_physics前启用并读回；初始化后预检刚体tensor和过滤视图、实际contact data读取，失败即终止。不要把这类缺失误诊为策略/碰撞参数问题。Kit的SimulationApp.close默认快退os._exit(0)会吞掉Python待打印Traceback，现异常在close前打印并存JSON traceback，close(exit_code=1)保留失败状态。只有CPU静态/桩回归，未新开Kit/PhysX，不能声称运行已通过。

- 新增独立`run_sim2sim.sh`用于146观测/23轴策略在Isaac Sim 6.1 CPU PhysX闭环执行，与静态`run_isaacsim.sh`分开。MJLab py313仅CPU导出原生模型/训练归一化MLP/生产features TorchScript；Isaac复用既有py312环境（uv --no-sync --offline），不改依赖。默认最新同任务checkpoint、60秒测试/10秒回合、单人体、无扰动/无LQR，原9块skin由原生场景更新驱动，不用参考项目静态rig冒充关节动画。打印来源/地形/关节映射，写report JSON+NPZ，启动默认静默进log，--verbose-init显示。未经新请求仍只做代码/静态检查，不自行启动Kit、PhysX或训练。
- sim2sim默认`--terrain trained`必须验证checkpoint旁bank.npz及SHA；当前最新groundsync1/model_200缺少快照，需用户显式`--terrain current --terrain-transfer`测试当前detail3，或`--terrain flat`作平地对照，不静默更换地形。所有旧invalid_terrain_training结论保留。仅导出所选16m真实碰撞块及同三角形高度查询，不做2km全图碰撞。与训练的随机reset不同，此测试固定所选patch/安全锚点/yaw，保守全鞋底间隙并保留1m边界/60度跌倒保护。
- sim2sim动作顺序由本机Entity.find_joints_by_actuator_names确认是自然JOINT序（不是XML actuator序），PD参数/限制按原生actuator_trnid映射。保持action clip2、scale.25、物理目标限幅，120Hz显式PD先限执行器力矩再加原关节被动阻尼，60Hz策略；无额外根支撑或控制器。现有原生MJCF导入器会将同体多hinge折叠为D6，因此USD显式串联23轴，12个小正质量辅助link共新增.0012kg/各惯量1e-6，其余人体质量/COM/主惯量保留；这与接触/限位/摩擦求解及显式积分差异必须记录，不能称与MuJoCo等价。
- sim2sim观测只把PhysX root/joint状态送给原生MuJoCo运动学/传感器（mj_kinematics/comPos/comVel/subtreeVel/sensorPos/sensorVel），禁止用MuJoCo动力学替代PhysX执行或生成假接触。root COM等于原点是已验证前提，world omega转换到freejoint body系；TOUCH载荷来自PhysX get_contact_force_data(dt=物理dt)的法向力、原生sole site体积筛选。原生subtree量包括原13个人体body、不包括USD辅助link；运行中另核对实际PhysX body frames，防止从错误关节坐标生成观测。肤动画用mjv_updateScene，避免3.12未暴露的mjv_updateActiveSkin替代函数导致每帧弃用警告。
- 静态sim2sim审计见scripts/audit_sim2sim.py及outputs/sim2sim_isaacsim/lunar_static_check：32姿态features误差2.39e-7、checkpoint/JIT动作误差0、native PD+passive力矩误差1.36e-14；Isaac env MuJoCo3.12与MJLab3.11 raw/obs/action对照误差0，USD转轴FK与原生误差6.07e-7，23动轴、9skins、仅3collider（双鞋+地面）。4项独立CPU单测覆盖native ray三角高度、旋转TOUCH体积、参数约束。均未启动Kit/PhysX动力学，不宣称跨引擎已站稳或渲染目视验收。

- 用户明确要求（2026-11-14）：尽量使用MuJoCo / MuJoCo Warp原生功能、函数、传感器和状态量。实现前先检查本机版本已有能力；原生能力满足需求时，不自行重复实现质量加权、运动学、动力学、接触或状态估计等计算。MJLab负责环境/训练组织，物理量优先取自底层原生接口。
- 使用原生量时核对含义、单位、坐标系、body/subtree范围、更新时机以及GPU后端支持；不能把尚未启用或未更新的字段当作有效结果。例：整个人体质心速度用Pelvis子树的原生subtreelinvel传感器，包含全部人体刚体，不用骨盆速度代替或自行逐体加权。确需自定义时，说明原生能力不足或后端不支持的具体原因，并保留明确的计算语义；不得以“原生优先”为由擅自修改共享依赖。
- 用户只让助手负责代码和静态检查；未经新的明确请求，不启动训练、仿真、回放、评估或GPU性能基准。已有历史短测不构成后续执行授权。
- 继续遵守父目录共享环境规则，复用 ~/dev-envs/robotics-mjlab-py313；不安装/升级GPU栈或修改共享依赖。
- 默认训练8192环境。性能方法参考zbot_rl_mjlab_R3与isaaclab-train-render-performance技能，但必须核对本机MJLab API与有效工作量；未实测不声称性能提升。
- 当前默认任务Wangyu-Moon-Stand-Push-v8；train.sh/run.sh无参数走新月面站立。旧月面v2显式stand-terrain-v2，旧平地显式stand-flat，资产预览显式--physics。
- run默认选对应实验最新非smoke运行中的最大已保存模型编号，可以是尚未完成的运行；打印任务/运行目录/检查点完整路径和关键运行参数。无结果明确报错，不兜底旧实验或旧物理资产。
- 当前默认人体23轴：腰3、髋6、膝2、踝4、肩6、肘2；新增腰Yaw和双踝Roll，头和手腕固定（旧20轴资产保留），浮动根不施加支撑力；物理用MuJoCo原生几何，蒙皮仅显示；仅鞋底与地面碰撞，双脚互碰排除。
- 月面站立使用9.81/6=1.635m/s²。保留源FBX初始姿态。随机位置/yaw需要按旋转后的双脚几何贴地，目标和跌倒高度必须参考出生点/局部地面，不能用世界绝对原点或海拔。
- 任务/观测/物理版本变化必须隔离日志与检查点；原跳跃和旧平地结果保留，不能把平地验证宣称为地形验证。

- 当前v4观测145维、动作23维，旧v3观测136维/动作20维保留（stand-v3）。v4增加joint_pose=-mean((q-q_initial)^2)，权重-1，使用原生关节状态与default_joint_pos，参考源FBX初始姿态，不使用动作目标代替关节偏离。
- 原v3及当前v4：直接使用原生frame、touch、subtreecom/subtreelinvel/subtreeangmom传感器。奖励主项为存活、骨盆线速度惩罚、人体总质心角动量惩罚与骨盆/躯干倾角惩罚；固定权重，不通过训练阶段改变正偏置伪造奖励变正。奖励/观测必须使用不同phase缓存，防止把sim.forward前的奖励状态复用作刷新后的观测。
- v3出生先核对实际MuJoCo碰撞高度场与采样数据一致，再筛选距边界至少20m的出生池；每批复位对旋转后双鞋底50点检查边界、有限值、至少14.5mm间隙和最多20cm离地间隙，不合格直接报错；偏离出生点1m即终止，不允许走出有限地形。未运行的8192环境只能称已加自检，不能声称实测通过。
- v3初始化参考zbot_exoarm_moon：有效锚点周围XY±1.25m、yaw±π、按旋转脚底点贴地并留15mm、清零速度；复位目标按实际出生点设置，禁止边界外无地面出生。
- 训练不增加额外[Rollout]终端行；本轮均值/奖励率/正奖励比例只写TensorBoard，保留原有固定console布局。
- 回放显式开启原生阴影及随人物平移的TRACK投影光，保证随机出生区域仍有局部脚底阴影；未显示验证不能声称画面已通过。
- 用户最新要求：不使用station_drift位置偏移奖励。保留出生位置偏移观测与1m活动范围终止保护；不得擅自恢复位置偏移惩罚。
- 标准RL项（力矩、关节速度、动作变化、关节限位等）优先检查并复用本机MJLab原生MDP函数，参考G1等标准任务的配置与数据语义，不照搬其机器人权重，不为已有功能另写重复实现。v4/v5新增joint_torques，直接用joint_torques_l2对23个原生actuator_force平方求和，单位传动比1时即Nm²；权重-2e-6，惩罚实际合成PD输出，包含LQR与RL贡献。
- stand-lqr-terrain/stand-v5为独立可选LQR+残差实验，默认v4不变；v5观测248维、23维残差，完整LQR权限时scale=.05rad，权限关闭时恢复.25rad，最终目标统一物理限幅。用户运行入口时，若缺少增益会先执行CPU平衡点搜索、原生FD和5个有界平地检查（包括真实源姿态+15mm出生间隙）；助手不能自行执行该准备脚本。训练期间不求逆动力学/导数/Riccati、不逐环境CPU循环。未实际准备/训练不能声称稳定或加速。浮动根角度差和角速度先转换至出生yaw局部世界系，再剔除世界yaw规范自由度，不能在倾斜参考姿态直接删body-z分量。

- 当前优先平地LQR：stand-lqr/stand-lqr-flat/stand-v7指向独立Wangyu-Moon-Stand-LQR-Flat-v7，日志wangyu_moon_stand_lqr_flat_v7（旧平地v6保留stand-v6）；旧月面v5保留stand-lqr-terrain/stand-v5。v6必须使用原生无限plane且摩擦[1,.01,.001]，从控制器qpos0和配套ctrl0复位、零速度，不加15mm间隙/随机地形出生。参考姿态可倾斜，出生heading仍为单位yaw坐标系，不能把参考roll/pitch当yaw去消掉。只完成代码/静态检查时不得宣称GPU平地LQR已有效。

- 平地v7纠正扰动时的门控参考切换：保持ctrl0平衡前馈，authority只缩放-Kdx，使用无时间延迟的C1 smoothstep权限映射，不回退源零目标、不盲目低通反馈。与v6隔离策略，保留相同物理资产和离线增益；只做数学/静态检查不能称扰动抖动已解决。

- 当前主线回到无LQR月面站立v8：train.sh/run.sh默认及stand/stand-terrain/stand-v8使用Wangyu-Moon-Stand-Push-v8，日志wangyu_moon_stand_push_v8；旧无扰动v4保留stand-v4。训练与回放均保留骨盆水平力脉冲：世界Fx/Fy各[-30,30]N，Fz和额外torque均0，持续0.10..0.20s，冷却2..4s独立计时、复位清零并重新等待。必须复用本机标准apply_body_impulse生命周期/原生wrench API，薄适配仅锁定Z采样为0，不另写逐环境调度；不改观测/奖励/地形出生检查、不加课程或LQR。未测不得声称收敛提速。

- 望宇FBX的Blender UV原样导出时，MuJoCo PNG纹理必须使用vflip=true；已用原FBX静态参考和MuJoCo A/B渲染确认。不能同时翻转UV的V和纹理，否则会重复翻转。skin.xml与静态visual场景、两个导出脚本保持一致。scripts/audit_wangyu_skin.py使用实际robot/挂载检查23轴、9块蒙皮、材质、权重、源姿态与纹理像素方向，静态渲染但不推进物理；旧validate_wangyu_skin.py包含历史20轴动态测试，不作为当前静态检查入口。仅改视觉资源无需重新训练，回放须重启以重新上传纹理。

- run v8支持--pushes/--no-pushes，默认关闭（用户最新要求），显式--pushes开启；仅本次回放/评估删除pelvis_push事件，不切换旧v4策略、不改变训练配置/观测。启动和评估报告必须标明状态，无扰动报告独立保存。

- 用户最新要求：joint_pose权重设0并从最新v8检查点续训100代；新主线v9，stand默认v9，stand-v8保留原实验。只改关节姿态奖励，不变观测/动力学/其他奖励，训练扰动开启、run扰动默认关闭。v9日志隔离到wangyu_moon_stand_push_v9。

- 最新主线v10（Wangyu-Moon-Stand-Support-v10，stand默认；stand-v9/v8保留）：取消连续gravity_tilt/joint_pose/质心角动量惩罚，骨盆速度权重-.5；以真实有载原生接触点投影凸包及LIPM捕获裕度近似构造capture_risk=-.15（2cm安全缓冲），无接触-.2；其他权重见support_task.REWARD_WEIGHTS单一来源。不得称为一般非平坦地形严格稳定裕度或ZMP。点/线/无接触不伪造正面积。每脚4接触槽，仅原生点。新增观测6维，总151，必须新训而非v8/v9直接续训。30秒回合，8192环境，run默认不扰动，训练默认30N/零额外力矩。只做CPU数学/配置/静态检查，不启动训练仿真。新日志wangyu_moon_stand_support_v10，自定义扰动另加幅度后缀。v10评估通过只定义为有限时长无失败复位；不把躯干竖直或固定站位当作稳定充分条件。

- 用户最新修正：v10保留cfg.rewards['joint_pose'].weight=-0.25。support_task.REWARD_WEIGHTS为实际统一来源，不能再次覆盖成0；v9字典也与其已有实际-.25保持一致。倾角和角动量权重仍0。

- 用户最新修正：v10保留质心角动量平方惩罚centroidal_angular_momentum=-0.002（原生Pelvis子树SUBTREEANGMOM），joint_pose=-0.25保持，gravity_tilt=0保持；以support_task.REWARD_WEIGHTS为准。

- run检查点选择包含同任务版本的自定义扰动幅度后缀目录；显式指定幅度且存在已保存匹配实验则优先匹配，否则选该版本最新非smoke运行最大编号模型。打印理由/实际目录，不跨版本回退。显式experiment-name/checkpoint-file优先。

- v10数值异常保护：standing_diagnostics复用本机NanGuard.detect_nans，加原生状态粗界限；pre-reward/post-forward fail-fast并保存原始.pt+.mjb，不裁剪异常奖励继续PPO，不隐藏异常复位。只增加诊断不改reward/obs含义，因此保留v10策略兼容性。无扰动静止指标需排除出生/最后外力（含qfrc_applied）后2秒；不要把恢复期间统计冒充静止抖动。用户只授权代码/静态检查，尚未运行此保护的真实仿真，勿称已解决数值发散或持续摆动。

- 新增独立v11鞋底接触对照：stand-contact/stand-v11 -> Wangyu-Moon-Stand-Support-Contact-v11，日志wangyu_moon_stand_support_contact_v11。默认stand仍v10。每只鞋底原box精确四等分，原始MuJoCo计算foot mass/ipos/iquat/inertia显式保留；新碰撞几何mass=0，几何并集不变，不加支撑力。接触传感器按Foot body聚合tiles，维持每脚4槽/总151观测与全部奖励/PD。保持joint_pose=-.25、角动量-.002。必须隔离日志并新训，不能直接resume v10。已静态检查物理不变量与实际robot/蒙皮attach，未仿真/训练，不得声称修复抖动或提升吞吐。根因线索是5m DEM上单个box/单个三角棱柱常仅一候选接触；hfield算法可从不同棱柱选多点，不能笼统说MuJoCo高度场永远只有一点。MjSpec编译后插入geoms可能导致名称查找索引过期，编辑时遍历元素name取对象，最终重新compile检验。

- v11原始异常现场证实pre_reward第12步/无外力时出现约-9m深度的远处虚假接触；不是CLI报错。stand-contact当前CCD对照profile原生根spec.option.ccd_tolerance=.001（本机MujocoCfg.apply不覆盖此字段），日志隔离到wangyu_moon_stand_support_contact_v11_ccd1e3。此为待用户运行验证的数值容差对照，尚未证明修复，勿关闭异常保护或用奖励裁剪掩盖。stand仍v10。MJLab ObservationManager构造时会在地形reset前探测term维数，guarded_observation在此阶段只返回形状，不对未初始化地形状态作异常判定；实际观测保持严格检查。

- 用户最新要求先回平面：默认stand/stand-plane/stand-v12 -> Wangyu-Moon-Stand-Support-Flat-v12，原生无限plane、不加载DEM、无LQR、恢复原单box鞋底；固定源姿态零速度，native mj_kinematics/mj_geomDistance静态确定15mm出生间隙。保留151观测、所有现有权重（joint_pose=-.25，H=-.002，tilt=0）、异常保护和无扰动静止诊断。8192环境/30秒回合。训练扰动默认仍30N/零力矩，先无扰动用--no-pushes；run默认不扰动。日志wangyu_moon_stand_support_flat_v12（变体加后缀）、报告/异常outputs/stand_support_flat_v12，必须新训/只续训v12。旧v10显式stand-terrain/stand-v10，旧v11显式stand-contact/stand-v11。只完成真实场景静态编译和CPU假复位测试，未训练仿真；不能声称平地已稳定。暂停局部地形改造，遵循用户当前平地目标。

- 稳定裕度审计见scripts/audit_support_margin.py及outputs/support_margin_audit/report.json。几何signed_margin通过3000随机SciPy oracle（f32~8e-8m）；**审计发现ContactSensor.data缓存跨pre/post-forward阶段未刷新（现已按下条contactsync1修复）**，仅support_mdp外层phase分离不足。无reset时新COM可配旧contact；真实静态快照能复现。后续应优先修正同阶段原生接触读取（或明确失效公共sensor缓存），不要再宣称当前裕度完整正确。仅验证脚本和文档改动，未改生产奖励/观测。物理仍是LIPM近似，force.norm门限不是法向门限，接触载荷子集不等于完整可用支撑区，无接触-.2是哨兵。审计仅CPU静态快照，无mj_step/策略/训练。

- 用户同意后已修复原生接触缓存时序：support_mdp.phase_contact_data在observation阶段用公共sensor.update(dt=0)刷新，无额外forward/时间推进。sensor必须stateless（air_time/history关闭）；reward阶段不额外update。保留151维/全部奖励/物理，旧v12可显式加载或resume；新v12日志基名wangyu_moon_stand_support_flat_v12_contactsync1，v10/v11也加contactsync1隔离。run默认不自动回退旧修复前目录，有需要显式checkpoint-file/experiment-name。已有5项真实ContactSensor CPU缓冲回归+6几何测试，静态audit调用生产函数；未训练/仿真或测GPU吞吐，勿称摆动已解决。此前审计的缓存错误已修复，但LIPM近似/载荷筛选/接触平均z定义的限制仍存在，未擅自改动。

- 用户最新要求：当前平地v12 episode_length_s=10秒（控制60Hz/600步）；不改旧地形任务30秒设置。独立evaluate继续使用--seconds指定时长。

- 用户最新要求增强平地运动惩罚：flat v12 pelvis_velocity=-4（8倍），centroidal_angular_momentum=-.01（5倍），其余项含joint_pose=-.25、10s、物理/观测保持。新日志基名wangyu_moon_stand_support_flat_v12_contactsync1_quiet1，支持旧v12显式resume隔离保存。未启动新训练。
- 用户明确授权本次回放验证：已完成model_300无扰动确定性64env×30s、zero PD同条件对照、以及10s回合的64env×30s诊断。不是对今后训练/仿真的持续授权。报告/轨迹outputs/stand_support_flat_v12/motion_check。nativeH/root速度确实受罚，当前策略残余运动，不能宣称改权重后已安静。MotionDiagnostics读取native缓冲/已有reward rate，ctrl必须按actuator_trnid映射到joint_names再与q比较；不能假设23轴的actuator顺序等于joint顺序。

- 用户最新替换项：当前平地v12删除pelvis_velocity奖励，改torso_velocity=-4.0（Torso刚体惯性COM世界线速度平方和），原生FRAMELINVEL/BODY传感器balance_torso_com_vel，避免测在腰转轴；不是XBODY或上身subtree速度。Torso速度由src/torso_velocity.py读取，shape151不变，无额外forward；H=-.01、joint_pose=-.25等保持。新日志基名wangyu_moon_stand_support_flat_v12_contactsync1_quiet1_torso1。添加torsospeed指标和motion-trace数据；已有v12可显式resume，新旧日志隔离。静态kinematics/comVel/sensorVel+jacBodyCom对照已验证，无动力学推进/训练。

- 用户确认加躯干角速度惩罚：flat v12 torso_angular_velocity=-.2，全三轴原生balance_torso_omega平方和；tilt仍0，不要求竖直。保留torso linear=-4、H=-.01、joint_pose=-.25、10s、151维及物理。新日志基名wangyu_moon_stand_support_flat_v12_contactsync1_quiet1_torso2；原v12可显式resume。记录torsobodyomega指标与trace，静态native雅可比和三轴测试验证；未新训练/仿真。

- 用户最新反馈：将稳定裕度capture_risk奖励权重设0后，训练/回放效果更好且不晃；已确认src/support_task.py为capture_risk=-0.0。保留用户这一设置，后续不得擅自恢复稳定裕度奖励。当前仅关闭奖励，151维观测与接触/裕度诊断仍保留；其他躯干线速度/角速度、H、姿态、力矩及airborne设置保持。此效果是用户反馈，未由助手新仿真独立复测，不宣称已证明单一因果。

- 用户最新要求从观测移除稳定裕度。新默认stand/stand-plane/stand-v13 -> Wangyu-Moon-Stand-Flat-v13，日志wangyu_moon_stand_flat_v13、outputs/stand_flat_v13。151→143：移除6维support扩展与旧base的2维LIPM预测偏移（95:97）。actor/critic都不输入这些代理量。capture_risk项删除，凸包/捕获裕度指标从v13训练/评估主路径移除；airborne直接复用原生接触掩码，无凸包。保留真实接触/载荷与基础COM等状态。平地、8192、10s、Torso线速-4/角速-.2、H-.01、pose-.25及其他权重不变。旧v12保留stand-v12，151维模型不能直接resume143维v13，必须新训，不暗中裁剪网络权重。guarded_observation支持自定义函数/维数并校验shape。仅CPU/静态/--help测试，未经新请求不仿真训练。

- 用户要求躯干速度/角速度直接观测，root系重力投影参考G1。v14（stand-plane/stand-v14，TASK Wangyu-Moon-Stand-Flat-v14，日志wangyu_moon_stand_flat_v14），146维。实际已有root gravity3（46:49）与Torso omega3（55:58），不重复追加；root gravity改用本机MJLab/G1标准projected_gravity，保留Torso局部omega×.5。新增3维Torso刚体COM线速度（143:146），native BODY framelinvel世界量经标准quat_apply_inverse转Torso系，÷.3。单位重力方向表示roll/pitch，不能测绝对yaw；不恢复倾角奖励。actor/critic同146维，稳定裕度仍完全排除。其余权重/10s/8192/物理不变。旧v13/12通过stand-v13/12保留，维数不同不能直接resume，需新训。

- 最新用户明确授权从刚才结果月面续训300代；已完成从v14的2026-11-14_21-56-39/model_500.pt恢复到v15的2026-11-14_22-18-13/model_799.pt（实际更新500..799共300，8192环境、10s、200N/50Nm）。本次授权不构成后续额外训练授权。默认stand/stand-terrain/stand-moon/stand-v15现在指向Wangyu-Moon-Stand-Lunar-v15，日志wangyu_moon_stand_lunar_v15；stand-plane/v14保留平地。使用64块16m真实Site20 DEM局部原生hfield，0.25m采样细分5m源数据，不虚构细节；每世界原生geom_dataid/geom_pos独立reset，形状XY一致，最大高差默认geom提供保守包围盒。MuJoCo编译会逐hfield重新归一化，必须逐块匹配zmin/span，不用共享竖直尺度。原生CPU射线对照、256假reset和8192真实出生检查通过，无数值guard/地形越界触发；不据此宣称接触全部正确或收敛。奖励/146维观测保持v14，无稳定裕度。首末50代奖励−1.55→−1.05，但躯干速度~.47m/s且频繁跌倒，未站稳；无独立确定性评估。完整来源、统计与地形bank见run/continuation.json、run/terrain及outputs/stand_lunar_v15/resume300.json。

- v15实际碰撞根因已确认：本机MJWarp smooth._geom_kernel跳过world-welded非mocap静态geom，geom_xpos仅make_data初始化；randomize geom_pos虽然CPU viewer/高度采样正确，GPU碰撞高度不更新。此前300代运行必须标为invalid_terrain_training，不能据其指标讨论有效月面学习。修正native_groundsync1将terrain挂在静止原生mocap body site20_ground下，仍0地形动态DOF，原生运动学刷新每世界geom位置；不手工改派生接触状态、不改共享库。CPU test检查mocap祖先，显式scripts/audit_local_lunar_runtime.py运行实际play配置64世界两轮reset，验证GPUgeom_xpos==model.geom_pos与下压30mm后全部接触；报告groundsync_audit.json。原model799无扰动16×5秒修正后测试1次失败，不能宣称站稳。未重新训练。未来v15训练独立日志wangyu_moon_stand_lunar_v15_groundsync1(+扰动后缀)，建议从原平地model500重新迁移；run在新结果存在前仍可选旧v15模型但基于continuation.json明确警告旧训练无效，不悄悄兜底其他任务。默认检查点选择包含groundsync1变体并优先匹配修正后幅度实验。

- 用户指出16m地形显示与原月面不同：已恢复play配置的完整原始Site20 2km×2km可视hfield（site20_full_visual），contype/conaffinity=0；不要再仅显示局部16m截块。全图位置每世界reset为[-source_center_x,-source_center_y,source_zmin-source_center_altitude]，与当前局部碰撞块同坐标；使用同一静止mocap父body，原生运动学同步显示位置。局部terrain碰撞几何在play时透明，训练不加载全图显示资源；不恢复旧巨大hfield碰撞。训练物理/观测/奖励不变，无须因显示变化重训。LocalSurface允许训练64/play65个hfield，visual_id可选。CPU静态ray验证4中心×3点完整可视DEM与局部高度误差<40微米；3个静态单测通过，本轮未动态回放/训练，不称已目视验收。全图可见不等于可任意走动，仍1m站立活动范围。

- 最新按用户要求参考~/Github/OmniLRS增加细节：lunar_details.py独立实现分层粗DEM+坐标种子泊松/间距拒绝小坑+变形径向坑碗/凸缘+多尺度平滑起伏+嵌入式凸起，无OmniLRS代码/资源复制、无Isaac依赖。参数assets/site20/detail_config.json；默认0.1m采样，R=.3..1m，深度=.075R，缘高=.3深度，凸起高.01..035m；附加高差实测生成bank约-.077..+.056m。这是程序生成，不是实测细节/岩石动力学/月壤标定模型。STEP=.1，源5m大形保留；加细后中央出生三角坡度<=15°再选64块，块中心互不覆盖10m显示孔，50脚点间隙检查保持。生成功能仅CPU初始化，reset GPU选择预生成hfield，不逐步生成。
- 完整2km显示改为源401²顶点/原三角面的非碰撞原生mesh，在64个源格点中心挖10m方孔，64个101²非碰撞细hfield填充；物理为64个161²16m hfield任选一个。细节在局部4..5m C1平滑归零，保留孔外粗DEM，避免粗hfield遮住坑底或与细表面z-fighting。所有显示geom同mocap父body，reset批量更新65个编译后geom_pos（保留原生mesh中心化/旋转偏置），物理/显示同高度源。训练仅64hfield，play128hfield+背景mesh，LocalSurface.visual_ids及visual_offsets[patch,65,3]；禁止恢复重叠粗hfield以填平坑底。
- 细节物理revision=native_groundsync1_detail1，train新实验wangyu_moon_stand_lunar_v15_groundsync1_detail1(+幅度后缀)，与旧地形日志隔离；检查点选择优先新细节同幅度及新细节已保存实验。LunarRunner继承StandRunner，只在有log_dir时保存terrain/metadata.json(配置SHA/来源/revision)+bank.npz(实际归一化高度及标定/中心)，不影响每步控制/奖励。旧v15无效300代不能充当细节地形训练，仍应从平地model500重新迁移。
- 本轮只静态CPU检查/原生渲染，无GPU动态/训练：4个中心细节与外圈原生ray高度、源外保持/重现性、256含65显示geom假reset通过；scripts/preview_lunar_detail.py生成outputs/stand_lunar_v15/detail/{standing,surface}.png及metadata.json，已经目视检查。脚本不调用mj_step。不要把之前native_groundsync1无细节GPU检查当成新细节已动态验证；后续显式scripts/audit_local_lunar_runtime.py支持全部显示geom同步检查，但本轮未执行。

- 用户澄清多人应该共享一张月面，随机位置/朝向，不是分屏：native月面回放num_envs>1接入shared_moon_viewer.SharedMoonViewer。GPU RL仍独立局部world，不改变8192训练；CPU显示用一个全局Site20场景+N份原生人体骨架/skin。SharedMoonModel把root平移[bank.centerXY,bank.altitude]回源地图坐标，yaw/关节/qvel不变，统一映射所有body wrench。地形仅1份，禁止原生_render_other_env_geoms追加重叠skin/地形；shared CPU display model不修改sim.mj_model，也不被其每世界geom_pos覆盖。
- MuJoCo 3.11 attach没有为skin.name/material正确命名空间化：逐人显式env_i/skin名，保留各自bonebody绑定；所有skin共享env_0材质，删除重复材质/纹理，16人=144skin、3纹理、一张地形，不复制纹理16份。显示模型几何不参与碰撞，原生mj_forward仅为显示/选取，不mj_step；训练碰撞仍按原GPU模型。鼠标使用显示body->(env,source body)映射及原生mj_applyFT，只写目标world qfrc_applied，结束/关闭清理该鼠标通道；外部随机wrench及紫色力矩图保留。
- A切换全图总览，按逗号/句号切换近距跟随者，滚轮缩放；总览有带编号位置标记，真实2km分布近景不保证所有人体同时入镜。Shared viewer setup恢复启动日志后打印布局/按键；初始化构建输出仍进startup log。回放cfg reset distinct_patches=True，<=64个world完整/部分reset都排除其他world正在用的中心避免出生重叠，训练False且不变。初始XY/yaw仍随机，既非4x4排布，也非4x4视口。
- 静态检查：native CPU回归3项验证独立skin骨骼、所有刚体global offset、原生ray显示地面vs局部采样、mj_applyFT广义力只映射目标人；16人原生静态渲染outputs/stand_lunar_v15/shared_view/{overview,tracked}.png已目视检查，144 skins/3 textures、一张地图。preview_shared_moon.py仅CPU构建+GL渲染不推进物理。新交互窗口和GPU动态本轮未跑，不声称运行时全面验收；未启动训练。

- 用户截图出现巨大的蓝色遮挡物，触发路径用户不清楚，不能仅凭截图声称已确认是蒙皮或marker根因。已移除shared view所有按视距缩放的实心定位球，统一add_overview_labels用原生mjGEOM_LABEL纯文字；只在_overview且FREE相机且distance>=20m时绘制，近景/跟随不显示。_setup_camera自身清_overview及user_scn.ngeom，消除旧装饰残留；preview_shared_moon复用同helper，避免静态预览和实测显示逻辑分叉。4个CPU/native回归通过，包括overview->鼠标zoom(不经过action)->native拾取tracking->跟随清理；16人静态预览已重生成并检查。截图所示交互现场未独立复现，不宣称全部运行时问题已解决；本轮无训练/动态回放。

- 用户最新要求：不需要位置球，也不需要编号文字。已删除add_overview_labels及全部人体位置装饰绘制，总览/近景均不显示这两类标记；preview脚本同步删除。保留A全图总览、逗号/句号跟随切换、原生状态HUD和既有力/力矩可视化，不擅自恢复人体定位标记。

- 最新蓝球根因已复现，前面猜测位置标记的诊断错误：本机mujoco.mjVISSTRING第19项Auto Connect默认快捷键也是A，passive UI先处理原生快捷键，再执行自定义总览回调。打开mjVIS_AUTOCONNECT会按原生vis.rgba.connect=[.2,.2,.8,1]绘制蓝色辅助几何，16人静态原生渲染可复现近景蓝色整屏遮挡，删除user_scn位置球/文字并不能解决。SharedMoonViewer在setup、A handler、每帧_update_debug_visualizers时关闭viewer.opt的AUTOCONNECT（后两处持viewer.lock）；SharedMoonModel额外将纯显示model的visual.rgba.connect alpha=0，阻止原生UI先切flag的一帧瞬态蓝影，不改GPU物理。A总览和逗号/句号跟随保留，无定位球/编号恢复。
- 回归5项通过，新增模拟passive先切原生A标志再处理总览，和真实mjv_updateScene在AUTOCONNECT强开时无不透明蓝色连接几何。16人固定姿态原生GL对照outputs/stand_lunar_v15/autoconnect_repro/{before,after}.png已目视检查：before蓝色遮挡，after同样原生flag强开仍正常人体；audit.json记录条件。没有启动动态回放/训练。后续自定义按键先查本机mjVISSTRING及原生UI绑定，不能把所有显示问题都归因于蒙皮/位置标记。

- 2026-11-15用户要求对照OmniLRS重新检查生成缺陷：新增scripts/audit_lunar_generation.py与outputs/terrain_generation_audit/report.json，记录本机OmniLRS commit 7fd43573772a445a04a38b113bb4d91e6018505e、生产源代码SHA/配置。只CPU生成与静态mj_forward/碰撞，未mj_step、GPU仿真或训练，未修改默认生成/物理/奖励。关键证据：256出生姿态无初始接触穿透，但原生静态下降找首次接触(精度0.049mm)测得12例实际全鞋底竖直间隙<14.5mm，最小11.304mm。原50点检查仅保证样点15mm，不保证整个鞋底，不能再把50点自检宣称完整15mm安全间隙。mj_geomDistance在这些非凸hfield给出负值而原生collision无穿透，不能直接据该API判定穿地；审计将其列为不采用的诊断，实际采用原生接触检测/静态下降。
- 同次生成审计：最窄坑缘sigma=.048m/FWHM=.113m，即.1m网格1.13个间隔；最小凸起短径仅3.2间隔，25点高度观测步距.25m更稀。212个非间距拒绝候选里148个(69.8%)因细节局部坡度被整块丢弃，原粗DEM坡度拒绝0，出生筛选与地貌生成耦合会偏向平缓样本。64×10m插块只覆盖2km×2km地图面积的.16%(地图为2km×2km即4e6m²)，核心满幅区域.1024%；不同crop种子对同一全局点示例相差11.1mm，但当前插块不重叠，不能将此说成现有显示/碰撞接缝错位。max(abs(x),abs(y))淡出在内部对角线非C1，外边界仍平滑归零。base只保持5m原三角折面，单一坑族/统一radius抽样/平滑嵌入凸起不等同OmniLRS多尺度形貌、邻块数据库与插值重建。OmniLRS默认HR DEM .025m、collider .05m，不应未经8192世界原生接触预算评估直接照搬。
- 另确认LunarRunner虽保存terrain/metadata.json和bank.npz，当前回放仍按现有代码/参数重新生成，没有加载或核对训练bank，参数/算法更新可能静默改变回放地形，应优先补地形指纹一致性。当前CPU静态降低20mm后接触最多43；实际cfg GPU nconmax128/njmax512，并非stand_task最底层16/96，不要误报容量不足或将CPU计数当GPU计数。已有3个显示/碰撞/假reset静态单测通过，新细节GPU动态仍未验证。

- 用户同意优先修前三项：已实现detail2（native_groundsync1_detail2，训练新日志wangyu_moon_stand_lunar_v15_groundsync1_detail2+幅度后缀），actor/critic146、8192、10s及奖励不变。LocalSurface用native mj_kinematics的实际box鞋底rest几何构建<=.025m网格；sole_sampling返回点和覆盖半径。每块求[-2,2]区域native hfield三角面的坡度Lipschitz界，出生高度额外加bound*cover+2e-6，assert所有脚点在2m认证区域且rest鞋底水平。这是有保守余量的覆盖界，不再以50个点假称全鞋底安全。原生CPU静态256姿态最小实际竖直间隙~18.14mm，无初始穿透、0例低于14.5mm，未进行GPU动态验证。
- detail2保留.1m网格，坑缘sigma>=1.5网格间隔(FWHM>=3.53格)，外缘紧支撑过渡>=3格；凸起radius改.4..6m使短径>=6.4格。代价/收益未实测，不声称训练提速；出生筛选偏差/全球细节连续性/基底5m折面问题仍未在这轮修复。
- 新terrain_provenance.py保存bank.npz及metadata(实际数组+配置/源/代码/REVISION fingerprint、archive SHA256)。play_stand对月面检查点加载前校验，LunarRunner.load同样校验月面续训，不匹配/缺失/损坏默认拒绝；显式--terrain-transfer才能把旧策略用于当前地形，打印not exact replay。training_options解析该flag并设置env，只允许显式迁移，不静默换地形；明确--resume-from flat_v14恢复仍支持并打印迁移说明。checkpoint_selection包含detail2并优先选择其已保存结果，旧300代仍无效。新测试覆盖快照匹配/缺失/修改/损坏和最小特征宽度，CPU原生高度/假reset检查通过。audit脚本当前输出detail2_verification.json，旧report.json保留原审计，不覆盖历史结论。
- 用户随后明确渲染分工：不在MJLab追求视觉渲染，训练/策略执行用MJLab，完成后轨迹回放到Isaac Sim做照片级场景，视觉参考OmniLRS environment=largescale。已撤回本轮试加的lunar_appearance.py/8k或2k月壤纹理拷贝/材质/太阳光修改，保留既有基本skin、阴影、共享月面调试和物理前三项修复。不要继续在MJLab堆视觉资源；不修改OmniLRS或共享依赖。推荐导出原生世界body姿态、关节、时间/reset标记及相同地形快照，再在Isaac Sim还原姿态渲染；导出/Isaac导入代码本轮尚未实现。直接在PhysX跑策略属于另一个sim2sim适配任务，不能将纯轨迹渲染等同跨引擎闭环验证。

- 用户要求参考~/myWorks/zbot_exoarm_moon提供run_isaacsim.sh看场景：已新增该启动脚本、scripts/export_isaacsim_scene.py与preview_isaacsim.py。复用现有IsaacLab uv项目~/nvidia/isaac-lab-3.0.0(.venv cp312)，uv --no-sync --offline，env -u PYTHONPATH避免MjLab py313污染，不安装依赖、不改参考项目/OmniLRS。--help在导入Isaac/启动Kit之前由argparse处理；--dry-run只检查环境/缓存。CPU exporter在MjLab py313中编译真实play场景，仅native mj_kinematics与ray，导出65个可视地形mesh/hfield顶面到outputs/isaacsim_preview/terrain.npz(813665 vertices/11MB)及json；读取native mesh变换/三角拓扑，非另一份月面。ray探测临时将透明physics collider alpha置1仅在CPU模型，因为原生mj_ray会跳过全透明geom，不改GPU或导出面。
- Isaac预览为独立静态USD场景：当前地形mesh、源地图偏移/指纹、参考项目LunarRegolith4k四贴图OmniPBR(OmniLRS材质参数)与DistantLight1750/angle.53，原始rigged Wangyu USD可选，不加载checkpoint、不声称已接入训练轨迹。--focus terrain(默认)/astronaut，--quality realtime/path，--headless需finite --frames，--snapshot需warmup后>=10帧，--export-usd保留外部资源引用。默认交互窗口由用户自行./run_isaacsim.sh启动。本轮只有脚本help/dry-run/py313和py312语法及CPU静态地形导出验证，未启动Isaac GUI、训练或物理，不称照片级效果已验收。后续真实动作回放仍需轨迹导出/rig绑定接口。

- 最新继续修网格为native_groundsync1_detail3，训练日志wangyu_moon_stand_lunar_v15_groundsync1_detail3(+幅度后缀)，146维/8192/10s/奖励不变，旧模型按既有指纹校验仅显式迁移。CPU RectBivariateSpline连续重建5m DEM（MuJoCo没有低分辨率DEM样条重建API），10cm原生hfield仍负责碰撞；插值不等于新增实测数据。lunar_details删除active_half_extent/fade_end配置及方形淡出，采用全局噪声+10m规范单元确定性元数据，邻单元padding覆盖特征支持及拒绝半径，使用候选priority Matérn式拒绝，查询顺序/裁剪中心不改变同一世界点。缓存元数据；跨>32m稀疏查询按单元分组，避免全局O(N²)拒绝，不建2km全10cm场。
- bank不再根据细节坡度淘汰整块；先保留64个相隔>=18m的地形，再用13x13三角坡度最大滤波筛各块内13x13出生候选（.6m保守脚部包络），reset随机选候选+XY±.025/yaw。所有64块均有安全点，共9029个，最少20个。LocalSurface验证native rest sole半径<=.5m，全鞋底坡度覆盖余量/2m认证区不变；snapshot FIELDS包含spawn_xy/spawn_counts。CPU256出生无穿透、0例小于14.5mm，最小约17.8mm。最大接触/约束仍不作GPU验证结论。
- 全局mesh最初独立显示hfield方案在整图边缘ray暴露底面命中；改为一体焊接mesh。背景Delaunay加入全部细块边缘点，检查每边仅一个背景相邻面，再将fine网格三角面用同边界vertex索引拼入；一个可视MESH，无可视hfield侧壁/底面。训练64原生hfield资源；play也是64hfield+1mesh，LocalSurface.visual_ids仅1个/visual_offsets[64,1,3]，不再128hfield/65显示geom。SharedMoonModel仍一张地图+独立人体skin，不改sim模型；Isaac exporter现在1个mesh约1819169顶点，需用新导出覆盖旧65面NPZ。远处粗LOD，只在64训练区域细化，但全局细节函数可按需查询；不得宣称整图高分辨率或已实现动态clipmap。
- 精确位于MESH顶点/边的native mj_ray实测可漏判，周围±1mm均有面时不能据孤立精确边ray认为几何裂缝。用焊接拓扑检查+每块四边内/外1mm各一次，共512 ray，细侧最大高度误差8.50e-6m，跨2mm最大高度变化.000521m。全局重叠查询示例差0，移除旧fade对角导数折线；新审计detail3_verification.json保留旧report/detail2报告。source高度/边界/假reset、全局场3项、共享body/skin/鼠标原生力5项、快照和CLI等静态测试通过；未启动GPU仿真/训练/IsaacGUI。后续多尺度坑族、独立岩块、远景高分辨率LOD仍是剩余工作。不要在MJLab重新增加照片级视觉资源。
