"""Refresh retained Studio visibility after classic-style geom group changes."""
import copy


def refresh_terrain_visibility(viewer):
    """Drive both cached terrain groups through on -> off before final sync.

    Classic evaluates geom_group every frame. Filament only changes membership
    when MjvOption changes, so moving a geom from group 0 to 5 alone leaves its
    old renderable visible. Force both transitions using the *new* group map.
    These syncs happen before rendering; no intermediate terrain is presented.
    NativeViewer.sync then applies the user's actual options as usual.
    """
    options=copy.deepcopy(viewer.vis_options)
    for visible in (1,0):
        options.geomgroup[0]=visible
        options.geomgroup[5]=visible
        viewer._model_scene.sync(
            viewer._window.GetWidth(),viewer._window.GetHeight(),
            viewer.model,viewer.data,viewer.perturb,viewer.camera,
            options,viewer.extra_geoms)
