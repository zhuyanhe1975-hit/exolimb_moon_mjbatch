"""Process-local Mac Studio focus setup and modifier compatibility."""
from contextlib import contextmanager
import ctypes
import sys


def sdl_api():
    from mujoco.studio import _studio
    api=ctypes.CDLL(_studio.__file__)
    for name,restype,argtypes in (
        ('SDL_SetHint',ctypes.c_int,[ctypes.c_char_p,ctypes.c_char_p]),
        ('SDL_GetWindowFromID',ctypes.c_void_p,[ctypes.c_uint32]),
        ('SDL_GetWindowTitle',ctypes.c_char_p,[ctypes.c_void_p]),
        ('SDL_GetKeyboardFocus',ctypes.c_void_p,[]),
        ('SDL_GetMouseFocus',ctypes.c_void_p,[]),
        ('SDL_RaiseWindow',None,[ctypes.c_void_p]),
        ('SDL_SetWindowInputFocus',ctypes.c_int,[ctypes.c_void_p])):
        function=getattr(api,name);function.restype=restype;function.argtypes=argtypes
    return api


def configure_mac_input():
    """Set only this process's SDL hints, before its Studio window is created."""
    if sys.platform!='darwin':return None
    api=sdl_api()
    api.SDL_SetHint(b'SDL_MAC_BACKGROUND_APP',b'0')
    api.SDL_SetHint(b'SDL_MOUSE_FOCUS_CLICKTHROUGH',b'1')
    api.SDL_SetHint(b'SDL_MAC_CTRL_CLICK_EMULATE_RIGHT_CLICK',b'0')
    return api


class StudioFocus:
    def __init__(self,api,title):
        self.api=api;self.window=None
        if api is not None:
            for identifier in range(1,257):
                window=api.SDL_GetWindowFromID(identifier)
                if window and api.SDL_GetWindowTitle(window)==title.encode():
                    self.window=window
    def acquire(self):
        # One launch-time request, never repeatedly steal focus after Cmd+Tab.
        if self.window:
            self.api.SDL_RaiseWindow(self.window)
            self.api.SDL_SetWindowInputFocus(self.window)
    def keyboard_focused(self):
        if self.api is None:return None
        return bool(self.window and self.api.SDL_GetKeyboardFocus()==self.window)


@contextmanager
def mac_control_or_command(imgui):
    """Python Studio mouse handlers accept either physical Mac modifier."""
    if sys.platform!='darwin':
        yield
        return
    original=imgui.GetIO
    class IOProxy:
        def __init__(self,io):self._io=io
        @property
        def KeyCtrl(self):return self._io.KeyCtrl or self._io.KeySuper
        def __getattr__(self,name):return getattr(self._io,name)
    imgui.GetIO=lambda:IOProxy(original())
    try:yield
    finally:imgui.GetIO=original
