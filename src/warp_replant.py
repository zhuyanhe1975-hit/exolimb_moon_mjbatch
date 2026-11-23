"""Device-resident MuJoCo Warp rollout for Wangyu-Replant-v1."""
import os

import mujoco_warp as mjw
import torch
import warp as wp
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from wangyu_replant import CONTACT_GEOMS, DT, TASK, Recovery


@wp.kernel
def _support_polygon(slots: wp.array4d[float], com: wp.array2d[float],
                     candidate: wp.array[bool], inside: wp.array[bool]):
    """Test every supporting line of the loaded contacts, with CPU's 1 mm tolerance."""
    w = wp.tid()
    inside[w] = False
    if not candidate[w]:
        return
    noncollinear = bool(False)
    for i in range(16):
        if slots[w,i//4,i%4,0] == 0.0 or wp.abs(slots[w,i//4,i%4,1]) <= 1.0:
            continue
        x = slots[w,i//4,i%4,4]
        y = slots[w,i//4,i%4,5]
        for j in range(i+1,16):
            if slots[w,j//4,j%4,0] == 0.0 or wp.abs(slots[w,j//4,j%4,1]) <= 1.0:
                continue
            dx = slots[w,j//4,j%4,4]-x
            dy = slots[w,j//4,j%4,5]-y
            length = wp.sqrt(dx*dx+dy*dy)
            if length < 1.e-8:
                continue
            low = float(0)
            high = float(0)
            for k in range(16):
                if slots[w,k//4,k%4,0] > 0.0 and wp.abs(slots[w,k//4,k%4,1]) > 1.0:
                    cross = dx*(slots[w,k//4,k%4,5]-y)-dy*(slots[w,k//4,k%4,4]-x)
                    low = wp.min(low,cross)
                    high = wp.max(high,cross)
            eps = 1.e-7*length
            if high > eps or low < -eps:
                noncollinear = True
            signed = dx*(com[w,1]-y)-dy*(com[w,0]-x)
            if (low >= -eps and signed < -.001*length) or (high <= eps and signed > .001*length):
                return
    inside[w] = noncollinear


class WarpReplantEnv(VecEnv):
    max_episode_length = 36*60

    def __init__(self, model, count):
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.num_envs, self.num_actions = count, model.nu
        self.cfg = dict(task=TASK, physics_backend='mujoco_warp', device=str(self.device))
        # One static host template, shared by all episodes and worlds.
        initial = Recovery(model)
        self.root, self.torso, self.tips = initial.root, initial.torso, initial.tips
        self.weight = float(initial.weight)
        def upload(value):
            return torch.as_tensor(value, dtype=torch.float32, device=self.device)
        self.q0 = upload(initial.d.qpos)
        self.ctrl0 = upload(initial.d.ctrl)
        self.obs0 = upload(initial.observe())
        self.potential0 = float(initial.potential())
        self.lo, self.hi = upload(initial.limits).T
        self.phase = torch.zeros(count,dtype=torch.long,device=self.device)
        self.hold = torch.zeros_like(self.phase)
        self.episode_length_buf = torch.zeros_like(self.phase)
        self.anchor = torch.zeros(count,2,2,device=self.device)
        self.previous = torch.full((count,),self.potential0,device=self.device)
        self.phase_ids = torch.arange(4,device=self.device)
        self.inside = torch.zeros(count,dtype=torch.bool,device=self.device)
        self.done = torch.ones_like(self.inside)
        self.obs = self.obs0.expand(count,-1)
        for i,name in enumerate(CONTACT_GEOMS):
            sensor = model.sensor(f'replant/{name}')
            if int(sensor.adr[0]) != i*28 or int(sensor.dim[0]) != 28:
                raise ValueError('Unexpected replant contact sensor layout')

        wp.init()
        self.warp_device = wp.get_device(str(self.device))
        self.mask = wp.from_torch(self.done,dtype=wp.bool)
        self.graphs = os.environ.get('MOON_WARP_GRAPH','1') == '1'
        with wp.ScopedDevice(self.warp_device), wp.ScopedStream(self.stream()):
            self.model = mjw.put_model(model)
            self.data = mjw.put_data(model,initial.d,nworld=count,nconmax=64,njmax=256)
            for name in ('qpos','qvel','ctrl','geom_xpos','subtree_com','xmat','xpos','sensordata'):
                setattr(self,name,wp.to_torch(getattr(self.data,name)))
            if self.graphs:
                stream = wp.Stream(self.warp_device)
                stream.wait_stream(self.stream())
                with wp.ScopedStream(stream):
                    with wp.ScopedCapture() as physics:
                        self.physics()
                    with wp.ScopedCapture() as reset:
                        mjw.reset_data(self.model,self.data,self.mask)
                self.stream().wait_stream(stream)
                self.physics_graph, self.reset_graph = physics.graph, reset.graph

    def stream(self):
        return wp.stream_from_torch(torch.cuda.current_stream(self.device))

    def physics(self):
        mjw.step(self.model,self.data)
        mjw.step(self.model,self.data)
        mjw.forward(self.model,self.data)

    def get_observations(self):
        return TensorDict({'policy':self.obs},batch_size=[self.num_envs])

    def step(self, actions):
        self.ctrl.copy_(self.lo+(actions.clamp(-1,1)+1)*.5*(self.hi-self.lo))
        with wp.ScopedStream(self.stream()):
            if self.graphs:
                wp.capture_launch(self.physics_graph)
            else:
                self.physics()
        self.episode_length_buf += 1
        slots = self.sensordata.reshape(self.num_envs,6,4,7)
        load = (slots[...,1].abs()*(slots[...,0]>0)).sum(-1)
        tips = self.geom_xpos[:,self.tips]
        com = self.subtree_com[:,self.root,:2]
        span = tips[:,1,:2]-tips[:,0,:2]
        fraction = ((com-tips[:,0,:2])*span).sum(-1)/span.square().sum(-1).clamp_min(1e-9)
        near = (com-tips[:,0,:2]-fraction.clamp(0,1)[:,None]*span).norm(dim=-1)<=.10
        loaded = load[:,4:].amin(-1)>=10
        upright = self.xmat[:,self.torso,2,2]
        height = self.xpos[:,self.root,2]
        speed = self.qvel[:,:6].norm(dim=-1)
        kneel = ((self.phase==0) & (upright>.65) & (height<.7) & (speed<.45)
                 & ((load[:,:4]>5).sum(-1)>=3) & (load[:,:4].sum(-1)>.65*self.weight))
        wp.launch(_support_polygon,self.num_envs,
                  inputs=[wp.from_torch(slots),wp.from_torch(com),wp.from_torch(kneel),
                          wp.from_torch(self.inside)],device=self.warp_device,stream=self.stream())
        standing = (loaded & near & (load[:,:2].amin(-1)>=5) & (load[:,2:4].amax(-1)<5)
                    & (height>.70) & (upright>.90) & (speed<.40))
        self.hold = torch.where(self.inside | ((self.phase==3)&standing),self.hold+1,0)
        knelt = (self.phase==0)&(self.hold>=12)&loaded
        lift = (self.phase==1)&(load[:,4:].amax(-1)<1)&(tips[:,:,2].amin(-1)>.06)
        replant = ((self.phase==2)&loaded&near
                   & ((tips[:,:,:2]-self.anchor).norm(dim=-1).amin(-1)>=.15))
        self.anchor = torch.where((knelt|((self.phase==1)&loaded))[:,None,None],tips[:,:,:2],self.anchor)
        advanced = knelt|lift|replant
        self.phase += advanced.long()
        self.hold = torch.where(advanced,0,self.hold)
        success = (self.phase==3)&(self.hold>=60)
        finite = self.qpos.isfinite().all(-1)&self.qvel.isfinite().all(-1)
        failed = ~finite | (self.qpos[:,:2].norm(dim=-1)>3)
        timeout = (self.episode_length_buf>=self.max_episode_length)&~(success|failed)
        done = success|failed|timeout
        potential = height+.5*upright
        reward = 4*(potential-self.previous)+8*advanced+50*success-5*failed
        reward = torch.where(finite,reward,-5.)
        obs = torch.cat((self.qpos[:,2:],self.qvel,load/self.weight,tips.flatten(1),
                         self.anchor.flatten(1),self.phase[:,None]==self.phase_ids,
                         self.hold[:,None]*DT,self.episode_length_buf[:,None]*DT),dim=-1)
        self.obs = torch.where(done[:,None],self.obs0,obs)
        self.previous = torch.where(done,self.potential0,potential)
        self.done.copy_(done)
        with wp.ScopedStream(self.stream()):
            if self.graphs:
                wp.capture_launch(self.reset_graph)
            else:
                mjw.reset_data(self.model,self.data,self.mask)
        self.qpos.copy_(torch.where(done[:,None],self.q0,self.qpos))
        self.ctrl.copy_(torch.where(done[:,None],self.ctrl0,self.ctrl))
        self.phase.masked_fill_(done,0)
        self.hold.masked_fill_(done,0)
        self.anchor.masked_fill_(done[:,None,None],0)
        self.episode_length_buf.masked_fill_(done,0)
        return self.get_observations(),reward,done,{'time_outs':timeout}
