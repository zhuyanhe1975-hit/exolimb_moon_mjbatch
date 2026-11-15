"""Playback-only terrain selection; policy/environment reset stays with viewer."""
from dataclasses import dataclass
from mujoco.studio import messages,plugin


def select_relative_terrain(environment,delta):
    device=next(dev for dev in environment.device_layer._devices if hasattr(dev,'select_terrain'))
    if device.terrain_size!=16:
        print('[Terrain] keyboard selection is available for the 16m bank only',flush=True)
        return False
    current=device.terrain_idx
    if current is None:current=int(device.ground.patch_ids[0].item())
    index=(current+delta)%64+1
    device.select_terrain(index)
    print(f'[Terrain] selected {index}/64; resetting all people',flush=True)
    return True


@dataclass
class TerrainCycle(messages.Event):
    delta:int


class TerrainControlSender(plugin.ViewerPlugin):
    @messages.handler
    def on_update(self,event:messages.UpdateEvent):
        from mujoco.studio.dear_imgui import dear_imgui as imgui
        io=imgui.GetIO()
        if getattr(io,'WantTextInput',False) or imgui.IsAnyItemActive():return
        # Reliable events preserve several quick key presses in their order.
        if imgui.IsKeyChordPressed(imgui.Key.LeftBracket):self.viewer.send_to_sim(TerrainCycle(-1))
        if imgui.IsKeyChordPressed(imgui.Key.RightBracket):self.viewer.send_to_sim(TerrainCycle(1))


class TerrainControlReceiver(plugin.SimPlugin):
    def __init__(self,environment):
        super().__init__()
        self.environment=environment

    @messages.handler
    def on_cycle(self,event:TerrainCycle):
        if select_relative_terrain(self.environment,event.delta):
            # REAF's ResetEvent handler resets policy state/environment on its
            # next StepEvent, including while paused. Never step from the UI.
            self.handle.dispatch(messages.ResetEvent())
