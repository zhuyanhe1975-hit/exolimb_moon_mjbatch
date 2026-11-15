"""Build only the local ImGui draw-data reader, with the existing SDK/torch headers."""
from pathlib import Path
import sysconfig,subprocess,torch
ROOT=Path(__file__).resolve().parents[1];sdk=Path.home()/'Github/mujoco-4.0.1'
out=ROOT/'outputs/native_ui';out.mkdir(parents=True,exist_ok=True)
subprocess.run(['g++','-std=c++20','-O2','-shared','-fPIC','-fvisibility=hidden',
 str(ROOT/'src/native_ui/draw_data.cc'),'-I'+str(sdk/'include'),
 '-I'+str(Path(torch.__file__).parent/'include'),'-I'+sysconfig.get_paths()['include'],
 '-o',str(out/('_moon_draw_data'+sysconfig.get_config_var('EXT_SUFFIX')))],check=True)
print('Built local draw-data reader:',out)
