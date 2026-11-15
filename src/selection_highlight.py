"""Renderable-local selection feedback; shared textures/material assets stay intact."""
import mujoco
import numpy as np


class SelectionHighlight:
    def __init__(self,model,content):
        self.model,self.content=model,content
        self.previous=set()
        self.items={}
        for i in range(model.ngeom):
            root=int(model.body_rootid[model.geom_bodyid[i]])
            if root:self.items[(mujoco.mjtObj.mjOBJ_GEOM,i)]=root
        for i in range(model.nskin):
            adr=int(model.skin_boneadr[i]);num=int(model.skin_bonenum[i])
            roots=np.unique(model.body_rootid[model.skin_bonebodyid[adr:adr+num]])
            if len(roots)==1 and roots[0]:self.items[(mujoco.mjtObj.mjOBJ_SKIN,i)]=int(roots[0])

    def update(self,body):
        root=int(self.model.body_rootid[body]) if 0<body<self.model.nbody else 0
        current={key for key,value in self.items.items() if root and value==root}
        for kind,index in self.previous|current:
            renderable=self.content.get_renderable(kind,index)
            material=renderable.get_material()
            material.selected=(kind,index) in current
            renderable.set_material(material)
        self.previous=current
