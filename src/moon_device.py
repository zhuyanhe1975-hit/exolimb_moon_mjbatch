"""REAF device for Wangyu. No MJLab runtime imports or dependency."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import numpy as np
import torch
from dm_env import specs
from reaf.simulation import torch as device
from balance_features import SENSORS,features,inverse_rotate
from measurement_terrain import measurement_terrain
import local_lunar_terrain as terrain

ASSET=Path(__file__).resolve().parents[1]/'assets/playground'
META=json.loads((ASSET/'metadata.json').read_text())

class MoonDevice(device.Device):
    def __init__(self,name='wangyu',pushes=True,push_force=30.,push_torque=0.,resample_terrain=True,shared_terrain=False,terrain_size=16,terrain_idx=None):
        self._name=name;self.pushes=pushes
        self.push_force=float(push_force);self.push_torque=float(push_torque)
        self.resample_terrain=resample_terrain
        if terrain_idx is not None and (type(terrain_idx) is not int or not 1<=terrain_idx<=64 or terrain_size!=16):
            raise ValueError('terrain_idx must be a 1–64 integer selecting the 16m bank')
        self.terrain_idx=None if terrain_idx is None else terrain_idx-1
        self.terrain_size=terrain_size
        self.shared_terrain=shared_terrain
        self._shared_patch=None
    def select_terrain(self,index):
        """Pin the next full playback reset to a user-facing 1-based bank index."""
        if self.terrain_size!=16 or type(index) is not int or not 1<=index<=64:
            raise ValueError('Terrain selection requires a 16m bank index in 1–64')
        self.terrain_idx=index-1
        self._shared_patch=None
        self._terrain_initialized.zero_()

    @property
    def name(self):return self._name
    def initialize(self,simulator):
        m=simulator.mj_model;self.m=m
        self.dev=simulator.data.qpos.device;self.n=simulator.data.qpos.shape[0]
        def t(a,dtype=torch.float32):return torch.as_tensor(a,device=self.dev,dtype=dtype)
        self.q0=t(m.qpos0);self.qadr=t(m.jnt_qposadr[1:],torch.long);self.vadr=t(m.jnt_dofadr[1:],torch.long)
        self.aorder=t(m.actuator_trnid[:,0]-1,torch.long)
        self.limits=t(m.jnt_range[1:]);self.root=m.body('robot/Pelvis').id
        self.default_q=self.q0[self.qadr]
        self.action_lo=self.limits[:,0].clamp_min(-.5)
        self.action_hi=self.limits[:,1].clamp_max(.5)
        self.contact_indices=t([[int(m.sensor('contacts/robot/'+side+'_sole').adr[0])+i
            for i in range(32)] for side in ('left','right')],torch.long)
        self.indices=t([int(m.sensor('robot/balance_'+name).adr[0])+i for name,_,_,_,width in SENSORS for i in range(width)],torch.long)
        self.veladr=int(m.sensor('robot/balance_torso_com_vel').adr[0])
        # Load the exported bank, rather than silently regenerate changed terrain.
        if self.terrain_size==100:
            from large_terrain import bank_data
            bank=bank_data()
        else:
            with np.load(ASSET/'state.npz') as z:bank={k:z[k].copy() for k in ('normalized','zmin','height_range','spawn_xy','spawn_counts','centers','altitudes')}
        if self.terrain_size==16:terrain.bank_data=lambda:bank
        self.ground=terrain.LocalSurface(SimpleNamespace(device=self.dev,num_envs=self.n,sim=SimpleNamespace(mj_model=m)),bank=bank)
        self._terrain_initialized=torch.zeros(self.n,dtype=torch.bool,device=self.dev)
        self.weight=float(m.body_subtreemass[self.root])*abs(float(m.opt.gravity[2]))
        self.action=torch.zeros(self.n,23,device=self.dev);self.previous=self.action.clone();self.pending=self.action.clone()
        self.target=torch.zeros(self.n,3,device=self.dev)
        self.heading=torch.zeros(self.n,4,device=self.dev);self.heading[:,0]=1
        self.wait=torch.zeros(self.n,device=self.dev);self.remaining=self.wait.clone();self.wrench=torch.zeros(self.n,6,device=self.dev)
        self._measurement_graph=None
        self._measurement_storage=None
        self.use_measurement_graph=self.dev.type=='cuda' and os.environ.get('MOON_MEASUREMENT_GRAPH','0')=='1'
    def substep_commands_spec(self):return {'wangyu/action':specs.Array((23,),np.float32)}
    def substep_measurements_spec(self):
        return {'wangyu/state':specs.Array((146,),np.float32),'wangyu/rewards':specs.Array((9,),np.float32),'wangyu/failed':specs.Array((),np.bool_)}
    def reset(self,simulator,env_mask=None):
        mask=torch.ones(self.n,dtype=torch.bool,device=self.dev) if env_mask is None else env_mask
        # Masked reset preserves the state of worlds that are still running.
        g=self.ground
        chosen=g.spawnable_ids[torch.randint(len(g.spawnable_ids),(self.n,),device=self.dev)]
        if self.terrain_idx is not None:chosen=torch.full_like(chosen,self.terrain_idx)
        if self.shared_terrain:
            # Playback worlds share one fixed collision surface for this session.
            if self._shared_patch is None:self._shared_patch=chosen[0].clone()
            chosen=self._shared_patch.expand_as(chosen)
        terrain_mask=mask if self.resample_terrain else mask&~self._terrain_initialized
        g.patch_ids[:]=torch.where(terrain_mask,chosen,g.patch_ids)
        self._terrain_initialized|=mask
        p=g.patch_ids
        simulator.model.geom_dataid[:,g.geom_id]=g.native_ids[p]
        simulator.model.geom_pos[:,g.geom_id,2]=g.zmin[p]
        idx=(torch.rand(self.n,device=self.dev)*g.spawn_counts[p]).long()
        xy=g.spawn_xy[p,idx]+(2*torch.rand(self.n,2,device=self.dev)-1)*.025
        yaw=(2*torch.rand(self.n,device=self.dev)-1)*torch.pi
        distributed=(self.shared_terrain and self.n>1) or self.terrain_size==100
        padding=g.clearance_padding[p]
        if distributed:
            from playback_spawns import distributed_spawns
            ids=mask.nonzero(as_tuple=True)[0]
            if len(ids):
                if not hasattr(self,'_playback_sole_cover'):
                    self._playback_sole_cover=terrain.sole_sampling(self.m)[1]
                placements,headings,margins=distributed_spawns(
                    g.meta,int(p[ids[0]].item()),g.sole_points.cpu().numpy(),
                    self._playback_sole_cover,len(ids),
                    simulator.data.qpos[~mask,:2].cpu().numpy(),
                    int(torch.randint(2**31-1,(1,),device=self.dev).item()))
                xy[ids]=torch.as_tensor(placements,device=self.dev,dtype=xy.dtype)
                yaw[ids]=torch.as_tensor(headings,device=self.dev,dtype=yaw.dtype)
                padding=padding.clone()
                padding[ids]=torch.as_tensor(margins,device=self.dev,dtype=padding.dtype)
        c=yaw.cos();s=yaw.sin();foot=g.sole_points
        # Preserve full-batch random draws and their order, but evaluate the
        # 208-point sole covering bound only for worlds that actually reset.
        # Full resets keep the fixed-size path and need no nonzero() sync.
        sparse=env_mask is not None and os.environ.get('MOON_SPARSE_RESET','0')=='1'
        ids=mask.nonzero(as_tuple=True)[0] if sparse else None
        active_xy=xy if ids is None else xy[ids]
        active_c=c if ids is None else c[ids];active_s=s if ids is None else s[ids]
        active_p=p if ids is None else p[ids]
        points=torch.stack((active_xy[:,None,0]+active_c[:,None]*foot[None,:,0]-active_s[:,None]*foot[None,:,1],active_xy[:,None,1]+active_s[:,None]*foot[None,:,0]+active_c[:,None]*foot[None,:,1]),-1)
        heights=g.height(points,env_ids=ids)
        active_z=(heights-foot[None,:,2]).max(-1).values+.015+(padding if ids is None else padding[ids])
        gaps=active_z[:,None]+foot[None,:,2]-heights
        valid=torch.isfinite(gaps).all(-1)&(points.abs()<=(g.half-2.25 if distributed else 2.)).all((1,2))&(gaps.min(-1).values>=.0145)&(gaps.max(-1).values<=.20)
        if not bool((valid if sparse else valid|~mask).all()):raise ValueError("Invalid whole-sole terrain spawn")
        if sparse:
            z=torch.zeros_like(yaw);z[ids]=active_z
        else:z=active_z
        q=self.q0.repeat(self.n,1);q[:,:2]=xy;q[:,2]=z;q[:,3:7]=0;q[:,3]=(yaw/2).cos();q[:,6]=(yaw/2).sin()
        simulator.data.qpos[:]=torch.where(mask[:,None],q,simulator.data.qpos[:])
        simulator.data.qvel[:]=torch.where(mask[:,None],0.,simulator.data.qvel[:])
        self.target[:]=torch.where(mask[:,None],q[:,:3],self.target)
        self.heading[:]=torch.where(mask[:,None],q[:,3:7],self.heading)
        for a in (self.action,self.previous,self.pending,self.wrench):a[mask]=0
        self.wait[:]=torch.where(mask,2+2*torch.rand(self.n,device=self.dev),self.wait)
        self.remaining[mask]=0
        simulator.data.xfrc_applied[:,self.root]=torch.where(mask[:,None],0.,simulator.data.xfrc_applied[:,self.root])
        simulator.data.ctrl[:]=torch.where(mask[:,None],self.q0[self.qadr][self.aorder],simulator.data.ctrl[:])
    def set_substep_commands(self,simulator,consumed_substep_commands):
        self.pending[:]=consumed_substep_commands['wangyu/action'].clamp(-2,2)
        # Preserve source scale -> default offset -> target clip ordering.
        target=(self.default_q+.25*self.pending).maximum(self.action_lo).minimum(self.action_hi)
        simulator.data.ctrl[:]=target[:,self.aorder]
        if self.pushes:
            dt=META['physics_dt'];self.wait-=dt;self.remaining-=dt
            start=(self.wait<=0)&(self.remaining<=0)
            force=torch.zeros_like(self.wrench)
            force[:,:2]=(2*torch.rand(self.n,2,device=self.dev)-1)*self.push_force
            torque=self.push_torque;force[:,3:]=(2*torch.rand(self.n,3,device=self.dev)-1)*torque
            self.wrench[:]=torch.where(start[:,None],force,self.wrench)
            self.remaining[:]=torch.where(start,.1+.1*torch.rand(self.n,device=self.dev),self.remaining)
            self.wait[:]=torch.where(start,self.remaining+2+2*torch.rand(self.n,device=self.dev),self.wait)
            simulator.data.xfrc_applied[:,self.root]=torch.where((self.remaining>0)[:,None],self.wrench,0.)
    def on_step(self,simulator):
        self.previous[:]=self.action;self.action[:]=self.pending
    def get_measurements(self,simulator):
        d=simulator.data
        # Only pure tensor measurement math is captured, never Warp physics,
        # reset, RNG, or the host-visible numerical guard. Detect reallocation.
        if self.use_measurement_graph:
            tensors=(d.qpos[:],d.qvel[:],d.sensordata[:],d.actuator_force[:],
                self.action,self.previous,self.target,self.heading,self.ground.patch_ids)
            storage=tuple((t.data_ptr(),tuple(t.shape),t.stride(),t.dtype) for t in tensors)
            if storage!=self._measurement_storage:
                from tensor_graph import FixedTensorGraph
                self._measurement_graph=FixedTensorGraph(lambda _:self._measurement_values(simulator),d.qpos[:,:1])
                self._measurement_storage=storage
            values=self._measurement_graph(d.qpos[:,:1])
        else:
            values=self._measurement_values(simulator)
        if not bool(values[:,156].bool().all()):
            out=ASSET.parents[1]/'outputs/numerical_guard';out.mkdir(parents=True,exist_ok=True)
            from time import time_ns
            torch.save({'qpos':d.qpos[:].detach().cpu(),'qvel':d.qvel[:].detach().cpu(),'sensordata':d.sensordata[:].detach().cpu(),'action':self.action.detach().cpu()},out/f'{time_ns()}.pt')
            raise FloatingPointError('Invalid native Wangyu state; raw snapshot saved; refusing reward masking')
        return {'wangyu/state':values[:,:146],'wangyu/rewards':values[:,146:155],'wangyu/failed':values[:,155].bool()}

    def _measurement_values(self,simulator):
        d=simulator.data;native=d.sensordata[:,self.indices];g=self.ground
        delta=d.qpos[:,:3]-self.target
        drift=inverse_rotate(self.heading,delta)[:,:2]
        raw=torch.cat((d.qpos[:,self.qadr]-self.q0[self.qadr],d.qvel[:,self.vadr],self.action,self.previous,native,drift),-1)
        if os.environ.get('MOON_BATCHED_TERRAIN','0')=='1':
            heights,scan,root_height=measurement_terrain(g,native[:,:7],native[:,29:31],native[:,36:38],d.qpos[:,:2])
        else:
            xy=torch.stack((native[:,:2],native[:,29:31],native[:,36:38]),1)
            heights,scan,root_height=g.height(xy),g.scan(native[:,:7]),g.height(d.qpos[:,:2])
        state,penalties=features(torch.cat((raw,heights,scan),-1),self.weight,abs(float(self.m.opt.gravity[2])),standing=True)
        torso=d.sensordata[:,self.veladr:self.veladr+3]
        tq=native[:,13:17]
        obs=torch.cat((state,inverse_rotate(tq,torso)/.3),-1)
        # Native contact sensors from moon_scene; no stale Python cache.
        contact=d.sensordata[:,self.contact_indices].reshape(self.n,2,4,8)
        valid=(contact[...,0]>0)&(contact[...,1:4].norm(dim=-1)>.5)&(contact[...,4]<=.002)&torch.isfinite(contact[...,5:8]).all(-1)
        airborne=~valid.any(dim=(1,2))
        rates=torch.stack((torch.ones_like(penalties[:,0]),penalties[:,0],penalties[:,1],penalties[:,2],penalties[:,3],d.actuator_force[:].square().sum(-1),airborne.float(),torso.square().sum(-1),native[:,17:20].square().sum(-1)),-1)
        finite=torch.isfinite(obs).all(-1)&torch.isfinite(rates).all(-1)&torch.isfinite(d.qvel[:]).all(-1)
        # Fail fast before REAF's reward provider can nan_to_num invalid rewards.
        bounded=(d.qvel[:].abs()<1e4).all(-1)&(d.qpos[:].abs()<1e6).all(-1)
        rootg=obs[:,46:49];torsog=obs[:,49:52]
        failed=(rootg[:,2]>-.5)|(torsog[:,2]>-.5)|((d.qpos[:,:2]-self.target[:,:2]).square().sum(-1)>1)|((d.qpos[:,:2].abs()>self.ground.half-2).any(-1))|((d.qpos[:,2]-root_height)<.70*self.q0[2])
        rates[:,0]=(~failed).float()
        return torch.cat((obs,rates,failed[:,None].float(),(finite&bounded)[:,None].float()),-1)
