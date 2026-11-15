from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from playback_clock import PlaybackClock,studio_clock_plugin


class FakeTime:
    t=0.
    def now(self):return self.t
    def sleep(self,dt):self.t+=dt


def test_realtime_includes_work_and_never_accumulates_catchup():
    time=FakeTime();clock=PlaybackClock(now=time.now,sleep=time.sleep)
    for _ in range(600):
        time.t+=.003
        clock.wait(1/60)
    assert time.t==pytest.approx(10.)
    time.t+=1.
    clock.wait(1/60)
    late=time.t
    clock.wait(1/60)
    assert time.t==pytest.approx(late+1/60)
    clock.reset();clock.wait(1/60,50)
    assert time.t==pytest.approx(late+1/60+1/30)


def test_studio_pause_reset_and_speed_without_simulation():
    time=FakeTime();plugin=studio_clock_plugin()
    plugin.clock=PlaybackClock(now=time.now,sleep=time.sleep)
    def tick(t):plugin.on_post_step(SimpleNamespace(data=SimpleNamespace(time=t)))
    tick(0.)
    for i in range(1,61):tick(i/60)
    assert time.t==pytest.approx(1.)
    time.t+=20.;tick(1.) # paused
    plugin.on_control(SimpleNamespace(speed=50.))
    tick(1.+1/60)
    assert time.t==pytest.approx(21.+1/30)
    tick(0.) # episode reset
    tick(1/60)
    assert time.t==pytest.approx(21.+2/30)
