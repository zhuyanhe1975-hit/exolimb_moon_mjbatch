from pathlib import Path
from types import SimpleNamespace
import sys
import mujoco
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from moon_studio_plugins import ContactSender,ContactReceiver,ContactOptionsSnapshot


def test_contact_snapshot_and_studio_flags_without_erasing_other_overlays():
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="2 2 .1"/><body pos="0 0 .09"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    sender=ContactSender(SimpleNamespace(mj_model=model,impl='cpp',data=SimpleNamespace(obj=data)),{'world':0})
    packets=[];sender._handle=SimpleNamespace(send_to_viewer=packets.append)
    sender.on_options(ContactOptionsSnapshot(True))
    assert len(packets)==1 and len(packets[0].positions)==data.ncon
    reference=np.zeros(6);mujoco.mj_contactForce(model,data,0,reference)
    np.testing.assert_allclose(packets[0].forces[0],data.contact[0].frame.reshape(3,3).T@reference[:3])
    options=mujoco.MjvOption();other=mujoco.MjvGeom();requests=[]
    receiver=ContactReceiver();receiver._viewer=SimpleNamespace(vis_options=options,extra_geoms=[other],send_to_sim=requests.append)
    receiver.on_contacts(packets[0])
    for points,forces in ((True,False),(False,True),(True,True),(False,False)):
        options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]=points
        options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]=forces
        receiver.on_update(None)
        assert receiver.viewer.extra_geoms[0] is other
        assert len(receiver.viewer.extra_geoms)==1+data.ncon*(int(points)+int(forces))
    assert requests[-1].enabled is False
