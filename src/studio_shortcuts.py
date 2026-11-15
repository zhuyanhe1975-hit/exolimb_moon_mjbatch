"""Keep display shortcuts usable after Studio's Reset button takes focus."""


def install():
    import mujoco
    from mujoco.studio.app import inputs
    if getattr(inputs.handle_vis_options_keyboard_events,'_moon_focus_fix',False):return
    original=inputs.handle_vis_options_keyboard_events
    def shortcuts(options,is_freecam_wasd=False):
        imgui=inputs.imgui;io=imgui.GetIO()
        if not io.WantCaptureKeyboard:
            return original(options,is_freecam_wasd)
        # Button navigation capture should not disable these display shortcuts.
        # Do not override text editing, active widgets, or ImGui's read-only IO.
        # The pinned Studio IO binding omits WantTextInput. Active InputText
        # widgets are covered by IsAnyItemActive; use the extra hint if exposed.
        if getattr(io,'WantTextInput',False) or imgui.IsAnyItemActive():return False
        pressed=imgui.IsKeyChordPressed
        for key,flag in ((imgui.Key.C,mujoco.mjtVisFlag.mjVIS_CONTACTPOINT),
                         (imgui.Key.F,mujoco.mjtVisFlag.mjVIS_CONTACTFORCE),
                         (imgui.Key.H,mujoco.mjtVisFlag.mjVIS_CONVEXHULL),
                         (imgui.Key.Semicolon,mujoco.mjtVisFlag.mjVIS_SKIN)):
            if pressed(key):
                options.flags[flag]^=1
                return True
        for i in range(6):
            key=getattr(imgui.Key,'N'+str(i))
            if pressed(int(key)|int(imgui.Key.Shift)):
                options.sitegroup[i]^=1
                return True
            if pressed(key):
                options.geomgroup[i]^=1
                return True
        return False
    shortcuts._moon_focus_fix=True
    inputs.handle_vis_options_keyboard_events=shortcuts
