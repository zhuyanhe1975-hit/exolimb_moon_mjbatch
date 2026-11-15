from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from moon_scene import scene_spec
from moon_viewer import display_spec


def test_native_skin_preserves_physics_and_tracks_bones(monkeypatch):
    monkeypatch.setenv('MOON_FILAMENT_SCENE','1')
    spec=scene_spec();source=spec.compile();display=display_spec(SimpleNamespace(mj_spec=spec)).compile()
    assert source.nskin==0 and display.nskin==9 and display.nskinvert==43786
    assert (source.nq,source.nv,source.nbody)==(display.nq,display.nv,display.nbody)
    robot=[i for i in range(source.nbody) if source.body(i).name.startswith('robot/')]
    np.testing.assert_array_equal(source.body_mass[robot],display.body_mass[robot])
    np.testing.assert_array_equal(source.hfield_data,display.hfield_data)
    assert all(display.body(int(i)).name.startswith('robot/') for i in display.skin_bonebodyid)
    data=mujoco.MjData(display);mujoco.mj_forward(display,data)
    scene=mujoco.MjvScene(display,maxgeom=300);option=mujoco.MjvOption();camera=mujoco.MjvCamera()
    mujoco.mjv_updateScene(display,data,option,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
    before=scene.skinvert.copy()
    data.qpos[0]+=.5;mujoco.mj_forward(display,data)
    mujoco.mjv_updateScene(display,data,option,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
    np.testing.assert_allclose((scene.skinvert-before).reshape(-1,3),np.tile([.5,0,0],(display.nskinvert,1)),atol=2e-6)


def test_usd_pbr_maps_physical_lighting_and_no_default_room_ibl(monkeypatch):
    monkeypatch.setenv('MOON_FILAMENT_SCENE','1')
    model=display_spec(SimpleNamespace(mj_spec=scene_spec())).compile()
    for i,part in ((1,'Helmet'),(2,'Boot'),(3,'Suit')):
        material=model.material(f'skin_mat_{i}').id
        for role,suffix in ((mujoco.mjtTextureRole.mjTEXROLE_METALLIC,'Metallic'),
                            (mujoco.mjtTextureRole.mjTEXROLE_ROUGHNESS,'Roughness'),
                            (mujoco.mjtTextureRole.mjTEXROLE_NORMAL,'Normal')):
            if i==1 and suffix=='Metallic':
                assert model.mat_texid[material,role]==-1
                assert model.mat_metallic[material]==0
                continue
            tid=int(model.mat_texid[material,role])
            expected='shell_roughness' if i==1 and suffix=='Roughness' else f'pbr_{part}_{suffix}'
            assert model.texture(tid).name==expected
            assert model.tex_colorspace[tid]==mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR
    visor=model.material('skin_mat_0').id
    assert model.mat_metallic[visor]==1
    # Measured gold F0 from Filament's sRGB metal reference table. Keeping the
    # old dark diffuse gold would attenuate the conductor reflections twice.
    np.testing.assert_allclose(model.mat_rgba[visor],[1.,.85,.57,1])
    np.testing.assert_allclose(model.mat_roughness[visor],.04)
    assert model.mat_reflectance[visor]==0  # planar mirror control, not Fresnel
    assert model.mat_texid[visor,mujoco.mjtTextureRole.mjTEXROLE_RGB]==-1
    # Glass is occluded: the visible face is Shield, sharing a mesh with its
    # white frame. A material test on Glass alone previously missed this bug.
    visible=model.skin_matid[model.skin('skin_Suit_Shield_0').id]
    assert model.material(int(visible)).name=='skin_mat_visor'
    assert model.mat_metallic[visible]==1
    mask=int(model.mat_texid[visible,mujoco.mjtTextureRole.mjTEXROLE_METALLIC])
    assert model.texture(mask).name=='visor_metallic'
    assert model.mat_texid[visible,mujoco.mjtTextureRole.mjTEXROLE_RGB]==model.texture('visor_color').id
    shell=model.skin_matid[model.skin('skin_Suit_Detail_0').id]
    assert model.mat_metallic[shell]==0
    assert model.mat_texid[shell,mujoco.mjtTextureRole.mjTEXROLE_RGB]==model.texture('shell_color').id
    ground=model.material('filament_lunar_regolith').id
    for role,name in ((mujoco.mjtTextureRole.mjTEXROLE_RGB,'regolith_diffuse'),
                      (mujoco.mjtTextureRole.mjTEXROLE_NORMAL,'regolith_normal'),
                      (mujoco.mjtTextureRole.mjTEXROLE_ORM,'regolith_orm')):
        assert model.mat_texid[ground,role]==model.texture(name).id
    np.testing.assert_allclose(model.mat_texrepeat[ground],[.25,.25])
    assert model.light_intensity[model.light('sun').id]>0 # physical mode disables built-in room
    assert model.light_type[model.light('lunar_bounce').id]==mujoco.mjtLightType.mjLIGHT_IMAGE
    assert model.light_texid[model.light('lunar_bounce').id]==model.texture('lunar_bounce_ibl').id
    tid=model.texture('lunar_bounce_ibl').id
    assert model.tex_colorspace[tid]==mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR
    # KTX must contain diffuse SH, not just a reflection-only PNG cube.
    import struct
    adr=int(model.tex_adr[tid]);size=int(model.tex_width[tid])
    raw=model.tex_data[adr:adr+size].tobytes()
    header=struct.unpack('<13I',raw[12:64])
    assert header[10]==6 and header[12]>0 and raw[68:71]==b'sh\0'
    np.testing.assert_allclose(model.light_intensity[model.light('lunar_bounce').id],
                               model.light_intensity[model.light('sun').id])
    assert model.numeric('filament.fallback.head_light_intensity').data[0]==0
