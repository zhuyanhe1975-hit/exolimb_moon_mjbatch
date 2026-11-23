"""Wangyu prone recovery on a lunar-gravity plane, with two free ground tips."""
import numpy as np
import mujoco
import torch
from tensordict import TensorDict
from rsl_rl.env import VecEnv
from scipy.spatial import ConvexHull, QhullError

from moon_scene import WANGYU_EVA_PROFILE, scene_spec

TASK = 'Wangyu-Replant-v1'
SIDES = ('left', 'right')
DT = 1 / 60


CONTACT_GEOMS = tuple(f'robot/{side}_{part}' for part in ('sole', 'shin') for side in SIDES) + ('left_tip', 'right_tip')


def build_model(*, training_sensors=False):
    spec = scene_spec(geometry='fitted', mass_profile=WANGYU_EVA_PROFILE,
                      suit_ankle_roll=True, suit_ankle_pitch=True, suit_major_joints=True)
    ground = spec.geom('terrain')
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.hfieldname = ''
    ground.pos = [0, 0, 0]
    ground.size = [8, 8, .1]
    ground.contype, ground.conaffinity = 1, 2
    for geom in spec.geoms:
        if geom.name.startswith('robot/'):
            geom.contype, geom.conaffinity = 2, 1
    for side, sign in zip(SIDES, (1, -1)):
        mount = spec.body('robot/Torso').add_body(name=f'{side}_rod', pos=[-.24, sign*.38, .30])
        shaft = mount.add_body(name=f'{side}_extension', pos=[0, 0, -.54])
        mount.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0,0,0,0,0,-.54],
                       size=[.025,0,0], mass=1.3, contype=0, conaffinity=0)
        shaft.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, name=f'{side}_tip', size=[.04,0,0],
                       mass=.8, contype=2, conaffinity=1, friction=[1,.01,.001])
        shaft.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0,0,0,0,0,.55],
                       size=[.015,0,0], mass=.4, contype=0, conaffinity=0)
        for body, suffix, kind, axis, limits, force, kp, kd in (
            (mount,'pivot',mujoco.mjtJoint.mjJNT_HINGE,[0,1,0],[-1.65,1.65],200,180,18),
            (shaft,'slide',mujoco.mjtJoint.mjJNT_SLIDE,[0,0,-1],[0,.85],500,1200,60)):
            name = f'{side}_{suffix}'
            body.add_joint(name=name, type=kind, axis=axis, limited=True, range=limits, damping=[2,0,0])
            act = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                                    forcelimited=True, forcerange=[-force,force])
            act.set_to_position(kp=kp, kv=kd)
    if training_sensors:
        # Plane-box contacts have at most four points, capsules two, spheres one.
        # Keep each physical contact position and local normal load for the hull test.
        for sensor in list(spec.sensors):
            spec.delete(sensor)
        for geom in CONTACT_GEOMS:
            spec.add_sensor(name=f'replant/{geom}', type=mujoco.mjtSensor.mjSENS_CONTACT,
                            objtype=mujoco.mjtObj.mjOBJ_GEOM, objname=geom,
                            reftype=mujoco.mjtObj.mjOBJ_GEOM, refname='terrain',
                            intprm=[19, 2, 4])
    return spec.compile()


class Recovery:
    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.root = model.body('robot/Pelvis').id
        self.torso = model.body('robot/Torso').id
        self.ground = model.geom('terrain').id
        self.support = [model.geom(f'robot/{s}_{part}').id for part in ('sole','shin') for s in SIDES]
        self.tips = [model.geom(f'{s}_tip').id for s in SIDES]
        self.contacts = self.support + self.tips
        joints = model.actuator_trnid[:,0]
        self.addr = model.jnt_qposadr[joints]
        self.limits = model.jnt_range[joints].copy()
        self.weight = model.body_subtreemass[self.root] * 1.635
        self.reset()

    def reset(self):
        m, d = self.m, self.d
        mujoco.mj_resetData(m,d)
        d.qpos[2] = 2
        angle = np.deg2rad(74.9)
        d.qpos[3:7] = [np.cos(angle/2),0,np.sin(angle/2),0]
        for s in SIDES:
            for joint, val in [('ShoulderPitch',-.7),('Elbow',-.8),('Knee',.25)]:
                j = m.joint('robot/'+s.title()+joint)
                d.qpos[j.qposadr[0]] = np.clip(val,*j.range)
            d.qpos[m.joint(s+'_pivot').qposadr[0]] = -.75
        mujoco.mj_forward(m,d)
        gaps = [mujoco.mj_geomDistance(m,d,self.ground,i,10,None)
                for i in range(m.ngeom) if m.geom_contype[i] == 2]
        d.qpos[2] -= min(gaps) - .001
        d.ctrl[:] = d.qpos[self.addr]
        mujoco.mj_forward(m,d)
        self.phase = self.hold = self.steps = 0
        self.anchor = np.zeros((2,2))
        self.load = np.zeros(6)
        self.previous = self.potential()

    def upright(self):
        return self.d.xmat[self.torso,8]

    def potential(self):
        return self.d.xpos[self.root,2] + .5*self.upright()

    def observe(self):
        d = self.d
        tips = d.geom_xpos[self.tips]
        return np.r_[d.qpos[2:], d.qvel, self.load/self.weight, tips.ravel(),
                     self.anchor.ravel(), np.eye(4)[self.phase], self.hold*DT,
                     self.steps*DT].astype(np.float32)

    def step(self, action):
        m,d = self.m,self.d
        lo, hi = self.limits.T
        d.ctrl[:] = lo + (np.clip(action,-1,1)+1)*.5*(hi-lo)
        mujoco.mj_step(m,d,nstep=2)
        mujoco.mj_forward(m,d)
        self.steps += 1
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            return -5., False, True
        self.load.fill(0)
        points = []
        wrench = np.empty(6)
        for k,c in enumerate(d.contact):
            if self.ground not in c.geom:
                continue
            other = c.geom[1] if c.geom[0]==self.ground else c.geom[0]
            if other in self.contacts:
                i = self.contacts.index(other)
                mujoco.mj_contactForce(m,d,k,wrench)
                self.load[i] += wrench[0]
                if i < 4 and wrench[0] > 1:
                    points.append(c.pos[:2].copy())
        tips = d.geom_xpos[self.tips]
        com = d.subtree_com[self.root,:2]
        span = tips[1,:2]-tips[0,:2]
        nearest = tips[0,:2]+span*np.clip((com-tips[0,:2])@span/max(span@span,1e-9),0,1)
        near = np.linalg.norm(com-nearest) <= .10
        loaded = min(self.load[4:]) >= 10
        phase = self.phase
        if phase == 0:
            stable = (self.upright()>.65 and d.xpos[self.root,2]<.7
                      and np.count_nonzero(self.load[:4]>5)>=3 and self.load[:4].sum()>.65*self.weight
                      and np.linalg.norm(d.qvel[:6])<.45 and len(points)>=3)
            if stable:
                try:
                    planes = ConvexHull(points).equations
                    stable = np.all(planes[:,:2]@com + planes[:,2] <= .001)
                except QhullError:
                    stable = False
            self.hold = self.hold+1 if stable else 0
            if self.hold>=12 and loaded:
                self.anchor[:] = tips[:,:2]
                self.phase = 1
        elif phase == 1:
            if loaded:
                self.anchor[:] = tips[:,:2]
            if max(self.load[4:])<1 and min(tips[:,2])>.06:
                self.phase = 2
        elif phase == 2:
            if loaded and near and min(np.linalg.norm(tips[:,:2]-self.anchor,axis=1))>=.15:
                self.phase = 3
        else:
            standing = (loaded and near and min(self.load[:2])>=5 and max(self.load[2:4])<5
                        and d.xpos[self.root,2]>.70 and self.upright()>.90
                        and np.linalg.norm(d.qvel[:6])<.40)
            self.hold = self.hold+1 if standing else 0
        advanced = self.phase != phase
        if advanced:
            self.hold = 0
        success = self.phase==3 and self.hold>=60
        potential = self.potential()
        reward = 4*(potential-self.previous) + 8*advanced + 50*success
        self.previous = potential
        failed = np.linalg.norm(d.qpos[:2])>3
        return reward-5*failed, success, failed


class ReplantEnv(VecEnv):
    device = 'cpu'
    max_episode_length = 36*60

    def __init__(self, model, count):
        self.num_actions, self.num_envs = model.nu, count
        self.worlds = [Recovery(model) for _ in range(count)]
        self.episode_length_buf = torch.zeros(count,dtype=torch.long)
        self.cfg = dict(task=TASK)

    def get_observations(self):
        obs = torch.from_numpy(np.stack([w.observe() for w in self.worlds]))
        return TensorDict({'policy':obs},batch_size=[self.num_envs])

    def step(self, actions):
        rewards, dones, timeouts = [], [], []
        for w,a in zip(self.worlds,actions.detach().cpu().numpy()):
            r,success,failed = w.step(a)
            timeout = w.steps>=self.max_episode_length and not (success or failed)
            done = success or failed or timeout
            rewards.append(r)
            dones.append(done)
            timeouts.append(timeout)
            if done:
                w.reset()
        self.episode_length_buf[:] = torch.tensor([w.steps for w in self.worlds])
        return (self.get_observations(), torch.tensor(rewards,dtype=torch.float32),
                torch.tensor(dones), {'time_outs':torch.tensor(timeouts)})
