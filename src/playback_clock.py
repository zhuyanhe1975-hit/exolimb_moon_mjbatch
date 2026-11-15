"""Wall-clock playback pacing, independent of physics and episode time origins."""
import time


class PlaybackClock:
    def __init__(self,*,now=time.monotonic,sleep=time.sleep):
        self._now=now
        self._sleep=sleep
        self.reset()

    def reset(self):
        self._deadline=self._now()

    def wait(self,simulation_dt,speed=100.):
        if simulation_dt<=0 or speed<=0:
            self.reset()
            return
        self._deadline+=simulation_dt*100./speed
        remaining=self._deadline-self._now()
        if remaining>0:self._sleep(remaining)
        else:
            # A slow frame must not cause a burst of accelerated catch-up steps.
            self.reset()


def studio_clock_plugin():
    from mujoco.studio import messages,plugin

    class StudioPlaybackClock(plugin.SimPlugin):
        def __init__(self):
            super().__init__()
            self.clock=PlaybackClock()
            self.speed=100.
            self._previous_time=None

        @messages.handler
        def on_control(self,event:messages.StepControlSnapshot):
            self.speed=event.speed
            self.clock.reset()

        @messages.handler
        def on_post_step(self,event:messages.PostStepEvent):
            current=float(event.data.time)
            previous=self._previous_time
            self._previous_time=current
            if previous is None or current<=previous:
                # Initial reset, pause, single-step idle or episode time reset.
                self.clock.reset()
                return
            self.clock.wait(current-previous,self.speed)

    return StudioPlaybackClock()
