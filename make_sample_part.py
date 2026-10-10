"""Generate a demo L-bracket STL (watertight, mm) so the pipeline can be exercised.

Profile (XY, mm): a 120 x 12 base with a 2.5 mm thin vertical wall -> produces
  * one sharp internal (concave) corner at x=2.5, y=12   -> sharp_internal_radius hotspot
  * an abrupt section change across y=12 -> section_change hotspot
  * a 2.5 mm wall against a 12 mm median                 -> thin_wall hotspot
Extruded 60 mm along Z.

Usage:  python make_sample_part.py [output.stl]
"""
import sys
from pathlib import Path

import trimesh
from shapely.geometry import Polygon


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "parts/bracket.stl")
    out.parent.mkdir(parents=True, exist_ok=True)

    profile = Polygon([(0, 0), (120, 0), (120, 12), (2.5, 12), (2.5, 72), (0, 72)])
    mesh = trimesh.creation.extrude_polygon(profile, height=60.0)

    if not mesh.is_watertight:
        print("ERROR: generated mesh is not watertight")
        return 1
    mesh.export(out)

    vol = mesh.volume
    print(f"wrote {out}  ({len(mesh.faces)} triangles, watertight={mesh.is_watertight})")
    print(f"  volume      = {vol:.1f} mm^3")
    print(f"  bbox        = {[round(float(x), 2) for x in mesh.extents]} mm")
    print(f"  mass (steel)= {vol * 1e-9 * 7850.0:.4f} kg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
