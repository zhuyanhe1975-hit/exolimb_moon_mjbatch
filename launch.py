"""Register the local task before delegating to Playground's native RL CLI."""
import argparse,os,sys,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))

def main():
    if len(sys.argv)>1 and sys.argv[1]=='train':
        sys.argv=[str(ROOT/'scripts/train_stand.py'),*sys.argv[2:]]
        runpy.run_path(str(ROOT/'scripts/train_stand.py'),run_name='__main__')
        return
    p=argparse.ArgumentParser(
        description='月面策略回放：默认16×16米地形、Studio 原生 OpenGL + skin，1 个世界，关闭扰动。',
        allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例（run_playground.sh 已自动添加 play）：
  ./run.sh --backend playground --device cuda:0
  ./run_playground.sh --checkpoint=/absolute/path/model_300.pt --num-envs=8
  ./run_playground.sh --viewer=classic --viz=physx --terrain-idx 12 --num-envs 4
  ./run_playground.sh --env-overrides='{"resample_terrain":true}'
  ./run_playground.sh --zero-actions

运行中按 [ / ] 切换上一块/下一块16米地形（1–64循环），并重置全部人物。

默认优先最新原生训练检查点，再兜底导入模型。帮助不加载仿真运行时。
连字符参数兼容旧下划线名称；其他上游参数原样透传。
环境变量：PLAYGROUND_PYTHON 指定解释器，MUJOCO_GL 默认 glfw。
""")
    p.add_argument('mode',choices=('train','play','check'))
    p.add_argument('--terrain-size',type=int,choices=(16,100),default=16,help='回放地形边长，默认16米；100米需显式选择')
    p.add_argument('--terrain-idx',type=int,default=None,metavar='1–64',help='固定选择16米bank中的地形编号（1–64）；省略则随机选择')
    p.add_argument('--terrain-transfer',action='store_true',help='允许检查点向当前地形迁移，不转换模型格式')
    p.add_argument('--viewer',choices=('studio','classic'),default='studio',help='Playback window; default: developer Studio.')
    p.add_argument('--reflection-env',default='lunar',metavar='lunar|royal|FILE.ktx',help='Studio PBR反射环境：默认月面；royal使用Three.js示例同源HDRI，或指定KTX立方体')
    p.add_argument('--viz',choices=('physx','skin'),default='skin',help='physx: physics proxy geometry; skin: astronaut suit (default). Both use MuJoCo physics.')
    p.add_argument('--num-envs','--num_envs',type=int,default=None,help='回放人数（默认 1）；多世界在一块共享地形上显示，兼容 --nworld。')
    forwarded={
        'resume_path': (('--checkpoint','--resume-path','--resume_path'),'检查点路径；默认自动选择'),
        'env_overrides': (('--env-overrides','--env_overrides'),'任务 JSON，默认 pushes=false、resample_terrain=false'),
        'env_id': (('--env-id','--env_id'),'初始显示的 world 编号，默认 0'),
        'device': (('--device',),'计算设备，例如 cuda:0、0、cpu；默认上游自动选择'),
        'gfx': (('--gfx',),'Studio 图形后端，默认 opengl；当前原生适配支持 OpenGL'),
        'hot_reload': (('--hot-reload','--hot_reload'),'热重载模式，例如 plugins、all；默认关闭'),
        'wp_kernel_cache_dir': (('--wp-kernel-cache-dir','--wp_kernel_cache_dir'),'Warp 缓存目录，默认 /tmp/wp_kernel_cache'),
        'env': (('--env',),'任务名称，默认本项目月面回放任务'),
        'port': (('--port',),'上游 web 端口，原生窗口不使用'),
    }
    group=p.add_argument_group('回放参数（推荐连字符名称，兼容旧名称）')
    for name,(aliases,help_text) in forwarded.items():
        group.add_argument(*aliases,dest=name,default=argparse.SUPPRESS,
                           type=int if name in ('env_id','port') else str,help=help_text)
    group.add_argument('--zero-actions','--zero_actions',dest='zero_actions',
                       nargs='?',const='true',choices=('true','false'),
                       default=argparse.SUPPRESS,help='使用零动作，不加载策略；默认 false')
    args,rest=p.parse_known_args()
    if args.mode=='play' and args.reflection_env!='lunar' and (args.viewer!='studio' or args.viz!='skin'):
        p.error('--reflection-env requires --viewer=studio --viz=skin')
    if args.terrain_idx is not None:
        if not 1<=args.terrain_idx<=64:p.error('--terrain-idx must be in 1–64')
        if args.terrain_size!=16:p.error('--terrain-idx selects the 16m bank; use --terrain-size=16')
    for name in (*forwarded,'zero_actions'):
        if hasattr(args,name):rest.append('--'+name+'='+str(getattr(args,name)))
    if args.num_envs is not None:
        if args.num_envs<=0:p.error('--num-envs must be positive')
        if any(a=='--nworld' or a.startswith('--nworld=') for a in rest):
            p.error('Use either --num-envs or --nworld, not both')
        rest.append('--nworld='+str(args.num_envs))
    if args.mode=='check':
        runpy.run_path(str(ROOT/'scripts/static_check.py'),run_name='__main__');return
    if args.mode=='play' and getattr(args, 'resume_path', None):
        from provenance import training_env_config
        if training_env_config(Path(args.resume_path).resolve().parent):
            p.error('该检查点来自独立训练任务。请用 ./run.sh --backend cpu --checkpoint PATH 回放')
    try:
        import moon_task
        moon_task.register()
    except ModuleNotFoundError as e:
        raise SystemExit(f'Missing Playground runtime: {e}. Set PLAYGROUND_PYTHON to a compatible isolated environment; see README.md.') from e
    from provenance import verify_assets,verify_checkpoint,save
    verify_assets()
    def option(name):
        for i,a in enumerate(rest):
            if a.startswith(name+'='):return a.split('=',1)[1]
            if a==name and i+1<len(rest):return rest[i+1]
        return None
    checkpoint=option('--resume_path')
    if args.mode=='play' and not checkpoint and option('--zero_actions') != 'true' and '--zero_actions' not in rest:
        from checkpoint_selection import latest_checkpoint
        try:selected,reason=latest_checkpoint(ROOT/'logs')
        except FileNotFoundError as error:raise SystemExit(str(error)) from error
        checkpoint=str(selected)
        print('[Checkpoint selection] '+reason,flush=True)
        rest.append('--resume_path='+checkpoint)
    if checkpoint:
        verify_checkpoint(checkpoint,args.terrain_transfer,args.terrain_size)
        print('[Checkpoint] '+str(Path(checkpoint).resolve()),flush=True)
    if args.mode=='train':
        logdir=option('--logdir')
        if not logdir:
            from datetime import datetime
            from zoneinfo import ZoneInfo
            logdir=str(ROOT/'logs'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S'))
            rest.append('--logdir='+logdir)
        save(logdir)
    if args.mode=='play':
        os.environ['MOON_VIZ']=args.viz
        os.environ['MOON_VIEWER']=args.viewer
        os.environ['MOON_REFLECTION_ENV']=args.reflection_env
        os.environ['MOON_FILAMENT_SCENE']='1' if args.viz=='skin' else '0'
        print(f'[Playback] viewer={args.viewer}; viz={args.viz}; requested_worlds={option("--nworld") or 1}',flush=True)
        from moon_viewer import install
        install()
        if args.viewer=='classic':
            from classic_viewer import install as install_classic
            install_classic()
        elif option('--gfx') is None:rest.append('--gfx=opengl')
        if args.viewer=='studio' and option('--gfx') in ('','opengl'):
            from studio_native import install as install_studio
            install_studio()
    module='mujoco_playground.rl.rsl_rl.'+args.mode
    if not any(a.startswith('--env=') or a=='--env' for a in rest):rest.insert(0,'--env='+moon_task.TASK_ID)
    if args.mode=='play':
        import json
        overrides={'pushes':False,'resample_terrain':False,'shared_terrain':True,'terrain_size':args.terrain_size}
        if args.terrain_idx is not None:overrides['terrain_idx']=args.terrain_idx
        supplied=option('--env_overrides')
        if supplied is not None:
            decoded=json.loads(supplied)
            if decoded.get('terrain_size',args.terrain_size)!=args.terrain_size:
                p.error('Set terrain size with --terrain-size, not a conflicting --env-overrides value')
            if 'terrain_idx' in decoded and decoded['terrain_idx']!=args.terrain_idx:
                p.error('Set terrain index with --terrain-idx, not --env-overrides')
            overrides.update(decoded)
            for i,a in enumerate(rest):
                if a.startswith('--env_overrides='):
                    del rest[i];break
                if a=='--env_overrides':
                    del rest[i:i+2];break
        rest.append('--env_overrides='+json.dumps(overrides))
    sys.argv=[module,*rest]
    runpy.run_module(module,run_name='__main__')

if __name__=='__main__':main()
