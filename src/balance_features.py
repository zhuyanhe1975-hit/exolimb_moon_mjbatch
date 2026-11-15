"""V4 balance signals from native MuJoCo sensors; separate pre/post-forward caches."""
from functools import partial
import math
import os
import torch
JOINT_DIM=23
OBS_DIM=145
STATE_DIM=4*JOINT_DIM
# Name, native sensor type suffix, native object type suffix, object name, dimension.
SENSORS=(
    ('root_pos','FRAMEPOS','SITE','balance_root',3),
    ('root_quat','FRAMEQUAT','SITE','balance_root',4),
    ('root_vel','FRAMELINVEL','SITE','balance_root',3),
    ('root_omega','FRAMEANGVEL','SITE','balance_root',3),
    ('torso_quat','FRAMEQUAT','SITE','balance_torso',4),
    ('torso_omega','FRAMEANGVEL','SITE','balance_torso',3),
    ('com_pos','SUBTREECOM','BODY','Pelvis',3),
    ('com_vel','SUBTREELINVEL','BODY','Pelvis',3),
    ('momentum','SUBTREEANGMOM','BODY','Pelvis',3),
    ('left_pos','FRAMEPOS','SITE','balance_left_sole',3),
    ('left_vel','FRAMELINVEL','SITE','balance_left_sole',3),
    ('left_load','TOUCH','SITE','balance_left_sole',1),
    ('right_pos','FRAMEPOS','SITE','balance_right_sole',3),
    ('right_vel','FRAMELINVEL','SITE','balance_right_sole',3),
    ('right_load','TOUCH','SITE','balance_right_sole',1),
)
SENSOR_DIM=sum(item[4] for item in SENSORS)
# Output columns after the actor observation. Penalties are POSITIVE physical quantities.
NAMES=('pelvis_speed_squared','momentum_squared','tilt_squared','action_change',
       'drift_squared','pelvis_speed','com_speed','momentum_norm','root_tilt_deg',
       'torso_tilt_deg','predicted_support_offset','double_contact','joint_pose_squared')
INDEX={name:OBS_DIM+i for i,name in enumerate(NAMES)}


def inverse_rotate(q,v):
    xyz=q[:,1:];cross=torch.cross(xyz,v,dim=-1)
    return v-2*q[:,:1]*cross+2*torch.cross(xyz,cross,dim=-1)


def features(x,weight,gravity,*,standing=False):
    q,qd,action,previous=x[:,:JOINT_DIM],x[:,JOINT_DIM:2*JOINT_DIM],x[:,2*JOINT_DIM:3*JOINT_DIM],x[:,3*JOINT_DIM:STATE_DIM]
    values={};offset=STATE_DIM
    for name,_,_,_,width in SENSORS:
        values[name]=x[:,offset:offset+width];offset+=width
    drift=x[:,offset:offset+2];ground=x[:,offset+2:offset+5];scan=x[:,offset+5:offset+30]
    root_q=values['root_quat'];torso_q=values['torso_quat']
    w,a,b,z=root_q.unbind(-1)
    c=1-2*(b*b+z*z);s=2*(w*z+a*b)
    norm=(c*c+s*s).sqrt().clamp_min(1e-6);c=c/norm;s=s/norm
    def yaw(v):
        return torch.stack((c*v[:,0]+s*v[:,1],-s*v[:,0]+c*v[:,1],v[:,2]),-1)
    down=torch.zeros_like(values['root_pos']);down[:,2]=-1
    rg=inverse_rotate(root_q,down);tg=inverse_rotate(torso_q,down)
    rw=inverse_rotate(root_q,values['root_omega'])
    tw=inverse_rotate(torso_q,values['torso_omega'])
    # Exact direction derivative in the rotating body frame, not finite differences.
    rgdot=-torch.cross(rw,rg,dim=-1);tgdot=-torch.cross(tw,tg,dim=-1)
    feet=torch.stack((values['left_pos'],values['right_pos']),dim=1)
    loads=torch.cat((values['left_load'],values['right_load']),-1).clamp_min(0)/weight
    contact=loads>.02
    load_sum=loads.sum(-1,keepdim=True)
    weighted=(feet*loads[:,:,None]).sum(1)/load_sum.clamp_min(.02)
    support=torch.where((load_sum>.02),weighted,feet.mean(1))
    com=values['com_pos'];vcom=yaw(values['com_vel']);relative=yaw(com-support)
    # LIPM look-ahead is a diagnostic/observation proxy, NOT a native support margin.
    # Native sensors provide COM position/velocity; no custom dynamics are applied.
    if not standing:
        horizon=(relative[:,2].clamp(.2,1.5)/gravity).sqrt()
        predicted=relative[:,:2]+horizon[:,None]*vcom[:,:2]
    foot_relative=torch.cat((yaw(values['left_pos']-com),yaw(values['right_pos']-com)),-1)
    foot_vel=torch.cat((yaw(values['left_vel']),yaw(values['right_vel'])),-1)
    clearance=torch.stack((values['left_pos'][:,2]-ground[:,1],values['right_pos'][:,2]-ground[:,2]),-1)
    root_clearance=values['root_pos'][:,2:3]-ground[:,:1]
    obs=torch.cat((q,.25*qd,rg,tg,.5*rw,.5*tw,
                   yaw(values['root_vel'])/.3,vcom/.3,relative,
                   yaw(values['momentum'])/10.,foot_relative,foot_vel/.5,
                   contact.float(),loads.clamp_max(2),clearance/.1,
                   drift/.3,root_clearance,rgdot[:,:2],tgdot[:,:2],
                   *(() if standing else (predicted/.2,)),
                   action,scan),-1)
    root_tilt=torch.atan2(rg[:,:2].norm(dim=-1),-rg[:,2])
    torso_tilt=torch.atan2(tg[:,:2].norm(dim=-1),-tg[:,2])
    speed2=values['root_vel'].square().sum(-1)
    h2=values['momentum'].square().sum(-1)
    if standing:
        # The lunar task removes the two LIPM columns and consumes only these
        # four diagnostics. Avoid computing unused norms/support predictions.
        return obs,torch.stack((h2,.5*(root_tilt.square()+torso_tilt.square()),
            (action-previous).square().mean(-1),q.square().mean(-1)),-1)
    diagnostics=torch.stack((speed2,h2,.5*(root_tilt.square()+torso_tilt.square()),
        (action-previous).square().mean(-1),drift.square().sum(-1),
        speed2.sqrt(),values['com_vel'].norm(dim=-1),h2.sqrt(),
        root_tilt*(180/math.pi),torso_tilt*(180/math.pi),predicted.norm(dim=-1),
        contact.all(-1).float(),q.square().mean(-1)),-1)
    return torch.cat((obs,diagnostics),-1)


