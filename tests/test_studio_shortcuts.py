import sys
from pathlib import Path
import mujoco
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def test_display_shortcuts_after_reset_focus_preserve_text_capture(monkeypatch):
    from mujoco.studio.app import inputs
    from studio_shortcuts import install
    imgui=inputs.imgui;context=imgui.CreateContext()
    try:
        io=SimpleNamespace(WantCaptureKeyboard=True,WantTextInput=False);options=mujoco.MjvOption()
        monkeypatch.setattr(imgui,'GetIO',lambda:io)
        monkeypatch.setattr(inputs,'handle_vis_options_keyboard_events',inputs.handle_vis_options_keyboard_events)
        monkeypatch.setattr(imgui,'IsAnyItemActive',lambda:False)
        key=[imgui.Key.C]
        monkeypatch.setattr(imgui,'IsKeyChordPressed',lambda k:k==key[0])
        install()
        for pressed,flag in ((imgui.Key.C,mujoco.mjtVisFlag.mjVIS_CONTACTPOINT),
                             (imgui.Key.F,mujoco.mjtVisFlag.mjVIS_CONTACTFORCE),
                             (imgui.Key.H,mujoco.mjtVisFlag.mjVIS_CONVEXHULL)):
            key[0]=pressed;options.flags[flag]=0
            for _ in range(2):
                io.WantCaptureKeyboard=True;io.WantTextInput=False
                assert inputs.handle_vis_options_keyboard_events(options,False)
                assert io.WantCaptureKeyboard
            assert options.flags[flag]==0
            io.WantTextInput=True
            assert not inputs.handle_vis_options_keyboard_events(options,False)
            assert options.flags[flag]==0
    finally:imgui.DestroyContext(context)


def test_shortcuts_with_real_studio_io_missing_text_hint(monkeypatch):
    from mujoco.studio.app import inputs
    from studio_shortcuts import install
    imgui=inputs.imgui;context=imgui.CreateContext()
    try:
        real_io=imgui.GetIO()
        # Only force the capture branch; every other field uses the real binding.
        class CapturedIO:
            WantCaptureKeyboard=True
            def __getattr__(self,name):return getattr(real_io,name)
        io=CapturedIO()
        monkeypatch.setattr(imgui,'GetIO',lambda:io)
        monkeypatch.setattr(inputs,'handle_vis_options_keyboard_events',lambda *a:False)
        active=[False]
        monkeypatch.setattr(imgui,'IsAnyItemActive',lambda:active[0])
        monkeypatch.setattr(imgui,'IsKeyChordPressed',lambda key:key==imgui.Key.C)
        install()
        options=mujoco.MjvOption();flag=mujoco.mjtVisFlag.mjVIS_CONTACTPOINT
        options.flags[flag]=0
        assert inputs.handle_vis_options_keyboard_events(options)
        assert options.flags[flag]==1
        # An active text field or other widget must keep its keyboard input.
        active[0]=True
        assert not inputs.handle_vis_options_keyboard_events(options)
        assert options.flags[flag]==1
    finally:imgui.DestroyContext(context)
