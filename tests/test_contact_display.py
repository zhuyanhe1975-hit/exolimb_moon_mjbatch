import sys
from pathlib import Path
import mujoco
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from contact_display import draw_contacts


def test_contact_overlay_uses_native_solver_force_and_switches():
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="2 2 .1"/><body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    assert data.ncon>0
    scene=mujoco.MjvScene(model,maxgeom=100)
    draw_contacts(model,data,scene)
    assert scene.ngeom==2*data.ncon
    np.testing.assert_allclose(scene.geoms[0].pos,data.contact[0].pos,atol=1e-6)
    force=np.zeros(6);mujoco.mj_contactForce(model,data,0,force)
    np.testing.assert_allclose(scene.geoms[1].size[2],.003*np.linalg.norm(force[:3]),rtol=1e-5)
    draw_contacts(model,data,scene,points=False,forces=False)
    assert scene.ngeom==0
    draw_contacts(model,data,scene,points=True,forces=False)
    assert scene.ngeom==data.ncon
