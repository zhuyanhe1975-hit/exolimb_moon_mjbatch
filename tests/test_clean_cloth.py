"""Asset calibration preserves badges while reducing fabric dirt-like relief."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from reference_materials import verify_clean_cloth


def test_clean_fabric_keeps_colored_badges_and_is_dielectric():
    asset=verify_clean_cloth()
    # Subsample maps to keep this CPU asset audit small.
    def load(path):return np.array(Image.open(path).convert('RGB'))[::16,::16]
    source=load(ROOT/'assets/wangyu_skin/Suit_BaseColor.png')
    color=load(asset/'Suit_BaseColor.png')
    rgb=source.astype(float);spread=rgb.max(-1)-rgb.min(-1);level=rgb.mean(-1)
    white=(spread<18)&(level>185)
    colored=spread>40
    assert white.any() and colored.any()
    np.testing.assert_array_equal(color[colored],source[colored])
    assert color[white].mean()>source[white].mean()+10
    assert not load(asset/'Suit_Metallic.png')[white].any()
    assert load(asset/'Suit_Roughness.png')[white].min()>=209
    original_n=load(ROOT/'assets/suit_pbr/Suit_Normal.png').astype(float)/127.5-1
    clean_n=load(asset/'Suit_Normal.png').astype(float)/127.5-1
    assert clean_n[white,2].min()>.8  # no grazing/back-facing wrinkle strokes
    assert np.abs(clean_n[white,:2]).mean()<np.abs(original_n[white,:2]).mean()
