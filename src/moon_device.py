"""REAF adapter; native CPU replay uses MoonDeviceCore directly."""
import torch
from reaf.simulation import torch as device
from moon_device_core import MoonDeviceCore, ASSET, META

class MoonDevice(MoonDeviceCore, device.Device):
    pass
