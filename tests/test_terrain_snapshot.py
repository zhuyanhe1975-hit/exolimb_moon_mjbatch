"""CPU-only terrain visibility lifecycle regression; no viewer or stepping."""
from pathlib import Path
from types import SimpleNamespace
import sys
import mujoco
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from moon_studio_plugins import TerrainReceiver,TerrainSnapshot


def test_terrain_snapshot_restores_static_and_survives_model_replacement():
    geoms=''.join(f'<geom name="{("terrain" if i==0 else f"display_terrain_{i:02d}")}" type="hfield" hfield="h{i}" group="5"/>' for i in range(64))
    assets='<asset>'+''.join(f'<hfield name="h{i}" size="1 1 1 1" nrow="2" ncol="2" elevation="0 0 0 0"/>' for i in range(64))+'</asset>'
    xml='<mujoco>'+assets+'<worldbody>'+geoms+'</worldbody></mujoco>'
    model=mujoco.MjModel.from_xml_string(xml)
    options=mujoco.MjvOption();options.flags[mujoco.mjtVisFlag.mjVIS_STATIC]=0
    receiver=TerrainReceiver();receiver._viewer=SimpleNamespace(model=model,vis_options=options)
    receiver.on_terrain(TerrainSnapshot(21))
    assert options.flags[mujoco.mjtVisFlag.mjVIS_STATIC]
    assert options.geomgroup[0] and not options.geomgroup[5]
    np.testing.assert_array_equal(np.flatnonzero(model.geom_group==0),[21])
    receiver.viewer.model=mujoco.MjModel.from_xml_string(xml)
    receiver.on_model(None)
    np.testing.assert_array_equal(np.flatnonzero(receiver.viewer.model.geom_group==0),[21])
    assert receiver.viewer._moon_terrain_visibility_changed


def test_terrain_delivery_retries_dropped_snapshot_and_waits_for_model():
    from mujoco.studio.launch_thread import make_thread_endpoints
    from moon_studio_plugins import TerrainSender,TerrainRequest,TerrainAck
    viewer_endpoint,sim_endpoint=make_thread_endpoints()
    spec=mujoco.MjSpec()
    for i in range(64):
        spec.add_hfield(name=f'h{i}',nrow=2,ncol=2,size=[1,1,1,1],userdata=[0,0,0,0])
        spec.worldbody.add_geom(name='terrain' if i==0 else f'display_terrain_{i:02d}',
            type=mujoco.mjtGeom.mjGEOM_HFIELD,hfieldname=f'h{i}',group=0 if i==21 else 5)
    source=spec.compile()
    sender=TerrainSender();sender._handle=SimpleNamespace(send_to_viewer=sim_endpoint.send_to_viewer)
    event=SimpleNamespace(model=source)
    sender.on_post_step(event)
    viewer_endpoint.get_sim_snapshots() # Studio Reset can drain pending snapshots.
    sender.on_post_step(event)
    packet=viewer_endpoint.get_sim_snapshots()[0]
    receiver=TerrainReceiver();receiver._viewer=SimpleNamespace(
        model=mujoco.MjSpec().compile(),vis_options=mujoco.MjvOption(),
        send_to_sim=viewer_endpoint.send_to_sim)
    receiver.on_terrain(packet) # snapshot races ahead of ModelEvent
    assert not sim_endpoint.get_viewer_events() # no premature acknowledgement
    receiver.viewer.model=source
    receiver.on_model(None) # apply cached snapshot and request current state
    events=sim_endpoint.get_viewer_events()
    assert isinstance(events[0],TerrainAck) and isinstance(events[1],TerrainRequest)
    sender.on_ack(events[0]);sender.on_post_step(event)
    assert not viewer_endpoint.get_sim_snapshots()
    sender.on_request(events[1]);sender.on_post_step(event)
    replacement=viewer_endpoint.get_sim_snapshots()[0]
    assert replacement.revision>packet.revision
    sender.on_ack(events[0]) # late ACK must not stop the new revision
    sender.on_post_step(event)
    assert viewer_endpoint.get_sim_snapshots()[0].revision==replacement.revision
    receiver.on_terrain(replacement)
    sender.on_ack(sim_endpoint.get_viewer_events()[0]);sender.on_post_step(event)
    assert not viewer_endpoint.get_sim_snapshots()
    np.testing.assert_array_equal(np.flatnonzero(source.geom_group==0),[21])


def test_camera_initialization_waits_for_valid_state_of_current_model():
    receiver=TerrainReceiver();receiver._viewer=SimpleNamespace(model=mujoco.MjSpec().compile())
    receiver.on_state(SimpleNamespace(state_sig=int(mujoco.mjtState.mjSTATE_QPOS),state=np.zeros(0)))
    assert not hasattr(receiver.viewer,'_moon_camera_state_model_id')
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
    receiver.viewer.model=model
    receiver.on_state(SimpleNamespace(state_sig=int(mujoco.mjtState.mjSTATE_QPOS),state=np.zeros(1)))
    assert not hasattr(receiver.viewer,'_moon_camera_state_model_id')
    receiver.on_state(SimpleNamespace(state_sig=int(mujoco.mjtState.mjSTATE_QPOS),state=model.qpos0.copy()))
    assert receiver.viewer._moon_camera_state_model_id==id(model)
