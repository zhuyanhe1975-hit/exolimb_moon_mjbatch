"""Classic OpenGL approximation of the shared Studio astronaut presentation."""
import mujoco


def adapt(spec):
    # RGB/UV, geometry, bones, and clean-cloth assets remain shared. Legacy
    # OpenGL has no Filament IBL/normal/metallic-roughness pipeline.
    # The legacy framebuffer does not apply Filament's output transfer. Use
    # the display PNG RGB values directly for the legacy color pipeline.
    rgb_names={material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] for material in spec.materials}
    for texture in spec.textures:
        if texture.name in rgb_names:
            texture.colorspace=mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR
    sun=spec.light('sun')
    sun.intensity=0
    sun.ambient=[.10,.10,.10]
    sun.diffuse=[.85,.85,.85]
    sun.specular=[.65,.65,.65]
    bounce=spec.light('lunar_bounce')
    bounce.type=mujoco.mjtLightType.mjLIGHT_SPOT
    bounce.texture=''
    bounce.intensity=0
    bounce.pos=[0,0,-20]
    bounce.dir=[0,0,1]
    bounce.cutoff=90
    bounce.exponent=0
    bounce.ambient=[.17,.165,.16]
    bounce.diffuse=[.10,.095,.085]
    bounce.specular=[0,0,0]
    bounce.attenuation=[1,0,0]
    bounce.castshadow=False
    # Encoded KTX is meaningful only to Filament and otherwise produces a
    # skipped-upload warning in the classic renderer.
    spec.delete(spec.texture('lunar_bounce_ibl'))
    spec.visual.headlight.active=0
    for material in spec.materials:
        material.reflectance=0
        material.specular=.06
        material.shininess=.12
    spec.material('skin_mat_1').specular=.15
    spec.material('skin_mat_1').shininess=.45
    for name in ('skin_mat_0','skin_mat_visor'):
        spec.material(name).specular=.8
        spec.material(name).shininess=.95
    ground=spec.material('filament_lunar_regolith')
    ground.specular=.03
    ground.shininess=.05
    ground.texrepeat=[.5,.5]
    return spec
