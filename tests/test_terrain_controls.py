"""Keyboard control and reliable reset dispatch; no runtime simulation."""
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from moon_device import MoonDevice
from terrain_controls import select_relative_terrain,TerrainCycle,TerrainControlReceiver,TerrainControlSender
from mujoco.studio import messages


def test_selection_wraps_and_repeated_events_reset_all_worlds():
    device=MoonDevice(shared_terrain=True,terrain_idx=64)
    device._terrain_initialized=torch.ones(4,dtype=torch.bool)
    device.ground=SimpleNamespace(patch_ids=torch.full((4,),63))
    device._shared_patch=torch.tensor(63)
    env=SimpleNamespace(device_layer=SimpleNamespace(_devices=[device]))
    controls=TerrainControlReceiver(env);events=[]
    controls._handle=SimpleNamespace(dispatch=events.append)
    controls.on_cycle(TerrainCycle(1))
    assert device.terrain_idx==0 and not device._terrain_initialized.any()
    assert device._shared_patch is None
    controls.on_cycle(TerrainCycle(-1))
    assert device.terrain_idx==63
    controls.on_cycle(TerrainCycle(-1))
    assert device.terrain_idx==62
    assert len(events)==3 and all(isinstance(e,messages.ResetEvent) for e in events)
    device.terrain_size=100
    assert not select_relative_terrain(env,1)
    assert device.terrain_idx==62


def test_studio_terrain_keys_preserve_widget_input(monkeypatch):
    from mujoco.studio.dear_imgui import dear_imgui as imgui
    control=TerrainControlSender();events=[]
    control._viewer=SimpleNamespace(send_to_sim=events.append)
    monkeypatch.setattr(imgui,'GetIO',lambda:SimpleNamespace(WantCaptureKeyboard=True))
    active=[False]
    monkeypatch.setattr(imgui,'IsAnyItemActive',lambda:active[0])
    monkeypatch.setattr(imgui,'IsKeyChordPressed',lambda key:key==imgui.Key.RightBracket)
    control.on_update(None)
    assert len(events)==1 and events[0].delta==1
    active[0]=True
    control.on_update(None)
    assert len(events)==1
