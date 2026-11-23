#!/usr/bin/env python3
"""Generate SHA-locked 18-vertex 3D skin-fitted flat-sole boot convex hulls.

Produces assets/boot_hull_18v/{fit.json,manifest.json} directly from
assets/wangyu_skin/skin.xml (skin_Boot_0 below-ankle vertices in LeftFoot/RightFoot
bind frames):
  - 8 adaptive outer vertices on the true rubber outsole (z <= z_min + 3.5 cm),
    strictly coplanar at z = z_min (flat support face, no tread noise).
  - 10 greedy 3D skin-envelope upper vertices (+2 mm radial fit, capped at ankle z <= 0)
    matching the sloped toe cap, medial/lateral welt, and tapered ankle cuff.
"""
from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
SKIN_PATH = ROOT / 'assets/wangyu_skin/skin.xml'
OUT_DIR = ROOT / 'assets/boot_hull_18v'
REVISION = 'boot_hull_fit3d_18v_v1'
K_BOT = 8
K_UPPER = 10
SOLE_Z_CUT_M = 0.035
UPPER_INFLATE_M = 0.002


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simplify_outer_2d(pts_2d: np.ndarray, k: int) -> np.ndarray:
    """Reduce a 2D convex hull to k outer vertices minimizing added polygon area."""
    ch = ConvexHull(pts_2d)
    poly = pts_2d[ch.vertices]
    while len(poly) > k:
        n = len(poly)
        best_area = float('inf')
        best_poly = None
        for i in range(n):
            p0, p1, p2, p3 = poly[(i - 1) % n], poly[i], poly[(i + 1) % n], poly[(i + 2) % n]
            d1, d2 = p1 - p0, p2 - p3
            det = d1[0] * (-d2[1]) - d1[1] * (-d2[0])
            if abs(det) < 1e-10:
                continue
            rhs = p3 - p0
            t = (rhs[0] * (-d2[1]) - rhs[1] * (-d2[0])) / det
            s = (d1[0] * rhs[1] - d1[1] * rhs[0]) / det
            if t < 1.0 - 1e-9 or s < 1.0 - 1e-9:
                continue
            q = p0 + t * d1
            area = 0.5 * abs((q[0] - p1[0]) * (p2[1] - p1[1]) - (q[1] - p1[1]) * (p2[0] - p1[0]))
            if area < best_area:
                best_area = area
                best_poly = (np.vstack([poly[:i], q[None], poly[i + 2:]])
                             if i < n - 1 else np.vstack([q[None], poly[1:n - 1]]))
        if best_poly is None:
            break
        poly = best_poly
    return poly


def build_fit3d_18v(pts3d: np.ndarray) -> dict:
    zb = float(pts3d[:, 2].min())
    zt = float(pts3d[:, 2].max())
    sole_xy = simplify_outer_2d(pts3d[pts3d[:, 2] <= zb + SOLE_Z_CUT_M, :2], K_BOT)
    bot_v = np.column_stack([sole_xy, np.full(len(sole_xy), zb)])

    upper_pts = pts3d[pts3d[:, 2] > zb + 0.020]
    verts = np.vstack([bot_v, upper_pts[np.argmax(upper_pts[:, 2])][None]])
    for _ in range(K_UPPER - 1):
        ch = ConvexHull(verts)
        dist = (upper_pts @ ch.equations[:, :3].T + ch.equations[:, 3]).max(axis=1)
        nxt = int(np.argmax(dist))
        if dist[nxt] <= 1e-5:
            break
        verts = np.vstack([verts, upper_pts[nxt:nxt + 1]])

    u = verts[len(bot_v):].copy()
    c = u.mean(axis=0)
    d = u - c
    nrm = np.linalg.norm(d, axis=1, keepdims=True).clip(1e-6)
    u = u + (d / nrm) * UPPER_INFLATE_M
    u[:, 2] = np.minimum(u[:, 2], zt)
    verts = np.vstack([bot_v, u])

    ch = ConvexHull(verts)
    remap = {int(o): n for n, o in enumerate(ch.vertices)}
    hv = verts[ch.vertices]
    hf = np.vectorize(remap.get)(ch.simplices).astype(np.int32)
    center = hv.mean(axis=0)
    for idx in range(len(hf)):
        a, b, c_ = hv[hf[idx]]
        if np.dot(np.cross(b - a, c_ - a), a - center) < 0:
            hf[idx, 1], hf[idx, 2] = hf[idx, 2], hf[idx, 1]

    return {
        'vertices': hv.tolist(),
        'faces': hf.tolist(),
        'vertex_count': int(len(hv)),
        'triangle_count': int(len(hf)),
        'flat_bottom_z': zb,
        'volume_m3': float(ch.volume),
        'sole_vertex_count': int(np.sum(hv[:, 2] <= zb + 1e-6)),
    }


def main():
    root = ET.parse(SKIN_PATH).getroot()
    boot_skin = next(s for s in root.findall('./deformable/skin') if s.get('name') == 'skin_Boot_0')
    all_v = np.fromstring(boot_skin.get('vertex'), sep=' ').reshape(-1, 3)

    bind_pos = {}
    for bone in boot_skin.findall('./bone'):
        bname = bone.get('body')
        if bname in ('LeftFoot', 'RightFoot'):
            bind_pos[bname] = np.fromstring(bone.get('bindpos'), sep=' ')

    feet = {}
    for side, bname, sign in (('left', 'LeftFoot', 1), ('right', 'RightFoot', -1)):
        bind = bind_pos[bname]
        pts = all_v[(all_v[:, 1] * sign > 0) & (all_v[:, 2] <= bind[2])] - bind
        hull = build_fit3d_18v(pts)
        assert hull['vertex_count'] == K_BOT + K_UPPER, hull['vertex_count']
        assert hull['sole_vertex_count'] == K_BOT, hull['sole_vertex_count']
        feet[side] = {
            'bind_position': bind.tolist(),
            'fit_vertex_count': int(len(pts)),
            'hull': hull,
        }

    payload = {
        'revision': REVISION,
        'source_sha256': sha256_file(SKIN_PATH),
        'generator_sha256': sha256_file(Path(__file__)),
        'method': (f'{K_BOT}-vertex adaptive outer outsole polygon (z<=z_min+{SOLE_Z_CUT_M*100:.1f}cm) '
                   f'at z_min + {K_UPPER} greedy 3D skin upper vertices (+{UPPER_INFLATE_M*1000:.0f}mm); '
                   'flat support face, no tread noise.'),
        'feet': feet,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fit_path = OUT_DIR / 'fit.json'
    fit_path.write_text(json.dumps(payload, indent=2) + '\n')

    manifest = {
        'revision': REVISION,
        'files': {
            'assets/wangyu_skin/skin.xml': sha256_file(SKIN_PATH),
            'scripts/prepare_boot_hull_18v.py': sha256_file(Path(__file__)),
            'assets/boot_hull_18v/fit.json': sha256_file(fit_path),
        },
    }
    (OUT_DIR / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for side in ('left', 'right'):
        h = feet[side]['hull']
        print(f'[boot_hull_18v] {side}: {h["vertex_count"]}v/{h["triangle_count"]}f, '
              f'sole_v={h["sole_vertex_count"]}, vol={h["volume_m3"]*1e6:.0f}cm3, z0={h["flat_bottom_z"]:.5f}')


if __name__ == '__main__':
    main()
