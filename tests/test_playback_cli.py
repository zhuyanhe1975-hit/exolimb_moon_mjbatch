"""CLI forwarding only, with all runtime entry points replaced by stubs."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import json
import pytest


@pytest.mark.parametrize('terrain_idx',[None,1,64])
@pytest.mark.parametrize('viewer',['studio','classic'])
@pytest.mark.parametrize('world_flag',['--num-envs','--num_envs','--nworld'])
def test_eight_world_cli_forwarding(monkeypatch,viewer,world_flag,terrain_idx):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('moon_launch_cli_test',root/'launch.py')
    launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
    for name,module in {
        'moon_task':SimpleNamespace(register=lambda:None,TASK_ID='test'),
        'provenance':SimpleNamespace(verify_assets=lambda:None,verify_checkpoint=lambda *a:None,save=lambda *a:None),
        'moon_viewer':SimpleNamespace(install=lambda:None),
        'classic_viewer':SimpleNamespace(install=lambda:None),
        'studio_native':SimpleNamespace(install=lambda:None),
    }.items():monkeypatch.setitem(sys.modules,name,module)
    for name in ('MOON_VIEWER','MOON_VIZ','MOON_FILAMENT_SCENE'):monkeypatch.setenv(name,'')
    captured=[]
    monkeypatch.setattr(launch.runpy,'run_module',lambda *a,**k:captured.extend(sys.argv))
    monkeypatch.setattr(sys,'argv',['launch.py','play','--viewer',viewer,world_flag,'8','--zero-actions'])
    if terrain_idx is not None:sys.argv.extend(['--terrain-idx',str(terrain_idx)])
    reference_env=viewer=='studio' and terrain_idx==1 and world_flag=='--num-envs'
    if reference_env:sys.argv.extend(['--reflection-env','royal'])
    launch.main()
    import os
    assert os.environ['MOON_REFLECTION_ENV']==('royal' if reference_env else 'lunar')
    assert '--nworld=8' in captured or ('--nworld' in captured and captured[captured.index('--nworld')+1]=='8')
    overrides=json.loads(next(a.split('=',1)[1] for a in captured if a.startswith('--env_overrides=')))
    expected={'pushes':False,'resample_terrain':False,'shared_terrain':True,'terrain_size':16}
    if terrain_idx is not None:expected['terrain_idx']=terrain_idx
    assert overrides==expected


@pytest.mark.parametrize('args',[['--terrain-idx','0'],['--terrain-idx','65'],['--terrain-size','100','--terrain-idx','1']])
def test_invalid_terrain_selection_fails_before_runtime_import(args):
    import subprocess
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,str(root/'launch.py'),'play',*args],capture_output=True,text=True)
    assert result.returncode==2 and '--terrain-idx' in result.stderr
