"""Local immutable asset manifest and checkpoint terrain compatibility."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ASSET=ROOT/'assets/playground'

def training_env_config(run_dir):
    try:
        env = json.loads((run_dir / 'config.json').read_text())['env']
        if isinstance(env, dict) and env.get('backend') in ('native_mujoco_cpu', 'mujoco_warp_cuda'):
            return env
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return {}

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify_assets():
    manifest=json.loads((ASSET/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        path=ROOT/name
        if not path.is_file() or digest(path)!=expected:raise ValueError(f'Migration asset changed: {path}')
    return digest(ASSET/'state.npz')
def verify_checkpoint(path,allow_transfer=False,execution_terrain_size=16):
    path=Path(path).resolve();snapshot=path.parent/'terrain_manifest.json'
    match=snapshot.is_file() and json.loads(snapshot.read_text()).get('state_sha256')==verify_assets()
    if not match and not allow_transfer:raise ValueError('Checkpoint terrain snapshot missing/mismatched. Explicit --terrain-transfer is required.')
    if execution_terrain_size==100:
        from large_terrain import bank_data,ASSET as LARGE_ASSET
        bank_data() # validates source and generated asset SHAs without regeneration
        if match and json.loads(snapshot.read_text()).get('exact_replay') is False:
            if digest(path)!=json.loads(snapshot.read_text()).get('policy_sha256'):
                raise ValueError('Imported policy SHA256 mismatch')
        print('[Terrain transfer] legacy policy -> 100x100m; not exact replay; '
              'source_state_sha256='+verify_assets()+'; execution_state_sha256='+digest(LARGE_ASSET/'state.npz'),flush=True)
        return
    if match and json.loads(snapshot.read_text()).get('exact_replay') is False:
        meta=json.loads(snapshot.read_text())
        if digest(path)!=meta.get('policy_sha256'):raise ValueError('Imported policy SHA256 mismatch')
        print('[Terrain transfer] imported MJLab policy -> current detail3; source terrain snapshot unavailable; not exact replay',flush=True)
    if not match:print('[Terrain transfer] current exported terrain; not exact replay',flush=True)
def save(logdir):
    path=Path(logdir);path.mkdir(parents=True,exist_ok=True)
    (path/'terrain_manifest.json').write_text(json.dumps({'state_sha256':verify_assets()},indent=2)+'\n')
