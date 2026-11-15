"""One terrain lookup for the standing observation and termination heights."""
import torch


def measurement_terrain(ground,root,left_xy,right_xy,qpos_xy):
    """Return native root/foot heights, the 25-point scan, and qpos height.

    The native sensor root and qpos root can represent different kinematic
    phases. Query both separately; only the scan center shares the sensor root.
    Keep the scan rotation/order/arithmetic identical to Surface.scan().
    """
    q=root[:,3:7];w,x,y,z=q.unbind(-1)
    c=1-2*(y*y+z*z);s=2*(w*z+x*y)
    norm=torch.sqrt(c*c+s*s).clamp_min(1e-6);c=c/norm;s=s/norm
    local=ground.scan_points
    xx=root[:,None,0]+c[:,None]*local[None,:,0]-s[:,None]*local[None,:,1]
    yy=root[:,None,1]+s[:,None]*local[None,:,0]+c[:,None]*local[None,:,1]
    sites=torch.stack((root[:,:2],left_xy,right_xy),1)
    xy=torch.cat((sites,torch.stack((xx,yy),-1),qpos_xy[:,None,:]),1)
    heights=ground.height(xy)
    scan=(5*(heights[:,3:28]-heights[:,:1])).clamp(-5.,5.)
    return heights[:,:3],scan,heights[:,28]
