#!/usr/bin/env python3
"""
Mechanical CAD Companion AI - LangGraph backend
================================================

Hub-and-spoke StateGraph (matches the "LangGraph CAD Orchestrator Flow" diagram):

    START -> orchestrator --+--> geometry_extractor  --+
                            +--> compliance_checker  --+--> back to orchestrator
                            +--> stress_reviewer     --+
                            +--> human_review (HITL interrupt)
                            +--> reporter -> END
                            +--> failure_handler -> END   (tool error or rejection)

The orchestrator is *deterministic Python* (no LLM routing). LLMs are used only
for (a) tool-call argument passing, (b) rulebook interpretation, (c) prose.
Every number in state comes from a Python tool; LLM-reported numbers are
overwritten or rejected by validators.

Install (Python >= 3.10):
    pip install langgraph langchain-core langchain-openai langchain-anthropic \
                langchain-chroma langchain-community langchain-text-splitters \
                trimesh numpy scipy rtree networkx pydantic pypdf
    pip install cadquery          # only needed for .step / .stp files

Env:
    LLM_PROVIDER=anthropic|openai   (default anthropic)
    ANTHROPIC_MODEL / OPENAI_MODEL  (optional overrides)
    OPENAI_API_KEY                  (embeddings always use OpenAI; also LLM if provider=openai)
    ANTHROPIC_API_KEY               (if provider=anthropic)

Usage:
    python cad_companion.py ingest ./rulebook/baja_sae_2026.pdf
    python cad_companion.py run ./parts/bracket.stl --units mm --material aisi_4130 \
           --part "suspension mounting bracket"
"""
import argparse
import functools
import json
import logging
import math
import os
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
import trimesh
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

try:  # optional: lets you keep keys in a local .env instead of the shell
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

log = logging.getLogger("cad_companion")


# =============================================================================
# 0. Settings, constants, LLM factory
# =============================================================================
@dataclass(frozen=True)
class Settings:
    llm_provider: str
    openai_model: str
    anthropic_model: str
    gemini_model: str
    embedding_provider: str
    embedding_model: str
    gemini_embedding_model: str
    rulebook_db_dir: str
    rulebook_collection: str
    reports_dir: str

    @classmethod
    def from_env(cls) -> "Settings":
        llm_provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
        return cls(
            llm_provider=llm_provider,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            # Embeddings must come from the same model at ingest and query time.
            # With the Gemini free tier, keep both on Gemini (no OpenAI credits needed).
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "gemini" if llm_provider == "gemini" else "openai"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            gemini_embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
            rulebook_db_dir=os.getenv("RULEBOOK_DB_DIR", "./.rulebook_chroma"),
            rulebook_collection=os.getenv("RULEBOOK_COLLECTION", "baja_sae_rulebook"),
            reports_dir=os.getenv("REPORTS_DIR", "./reports"),
        )


UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}

# Handbook densities, kg/m^3. Used ONLY when the caller names a material.
MATERIAL_DENSITY_KG_M3 = {
    "aisi_4130": 7850.0,
    "aisi_1018": 7870.0,
    "al_6061_t6": 2700.0,
    "al_7075_t6": 2810.0,
    "ti_6al_4v": 4430.0,
}

MAX_STRESS_TOOL_ROUNDS = 6


@functools.lru_cache(maxsize=1)
def get_llm():
    s = Settings.from_env()
    if s.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=s.openai_model, temperature=0)
    if s.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=s.gemini_model, temperature=0, max_retries=4)
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=s.anthropic_model, temperature=0, max_tokens=4096)


class CADToolError(RuntimeError):
    """Raised by any CAD tool. Never swallowed: it puts the graph in a failure state."""


# =============================================================================
# 1. State
# =============================================================================
class AgentState(TypedDict):
    # --- keys required by the spec ------------------------------------------
    cad_file_path: str
    extracted_properties: dict      # mass, volume, bounding box, CoG (+ provenance)
    compliance_status: list[dict]   # rule checked, pass/fail, reasoning (+ audit fields)
    stress_hotspots: list[dict]     # location, estimated severity, geometric reason (+ id)
    final_report: str
    current_step: str
    # --- operational keys ---------------------------------------------------
    units: str                      # units of an STL (STEP is read as mm)
    material: str | None            # key of MATERIAL_DENSITY_KG_M3, supplied by the user
    density_kg_m3: float | None     # explicit override, supplied by the user
    part_description: str
    completed: list[str]            # which spokes have finished (drives the router)
    error: str | None               # set => graph routes to failure_handler
    stress_notes: str               # qualitative LLM note on hotspots (advisory)
    fea_job: dict                   # status of the (stubbed) external FEA job
    human_review: dict              # engineer's decision at the HITL gate
    verdict: str
    report_path: str | None


def make_initial_state(
    cad_file_path: str,
    units: str = "mm",
    material: str | None = None,
    density_kg_m3: float | None = None,
    part_description: str = "",
) -> AgentState:
    return AgentState(
        cad_file_path=cad_file_path,
        extracted_properties={},
        compliance_status=[],
        stress_hotspots=[],
        final_report="",
        current_step="init",
        units=units,
        material=material,
        density_kg_m3=density_kg_m3,
        part_description=part_description,
        completed=[],
        error=None,
        stress_notes="",
        fea_job={},
        human_review={},
        verdict="PENDING",
        report_path=None,
    )


# =============================================================================
# 2. Pydantic tool schemas + CAD tools
# =============================================================================
# ---- 2a. Geometry extraction -------------------------------------------------
class ExtractGeometryInput(BaseModel):
    """Input for extract_geometry. The pipeline overrides every field with trusted state values."""

    file_path: str = Field(description="Path to the .stl / .step / .stp file.")
    units: Literal["mm", "cm", "m", "in"] = Field("mm", description="Model units (STL only).")
    material: str | None = Field(None, description="Material key supplied by the user, or null.")
    density_kg_m3: float | None = Field(None, gt=0, description="Explicit density, or null.")


class GeometryProperties(BaseModel):
    source_format: Literal["stl", "step"]
    volume_source: Literal["b-rep", "mesh"]
    volume_mm3: float
    surface_area_mm2: float
    bounding_box_xyz_mm: list[float] = Field(min_length=3, max_length=3)
    bbox_min_mm: list[float] = Field(min_length=3, max_length=3)
    bbox_max_mm: list[float] = Field(min_length=3, max_length=3)
    center_of_gravity_mm: list[float] = Field(min_length=3, max_length=3)
    material: str | None
    density_kg_m3: float | None
    mass_kg: float | None = Field(description="null when no material/density was supplied")
    triangle_count: int
    is_watertight: bool
    warnings: list[str] = []


def _load_stl(path: Path, units: str):
    loaded = trimesh.load(str(path), file_type="stl", force="mesh")
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise CADToolError(f"STL contains no triangles: {path}")
    mesh = loaded.copy()
    mesh.apply_scale(UNIT_TO_MM[units])
    return mesh, {"format": "stl", "exact_volume_mm3": None, "exact_com_mm": None, "warnings": []}


def _load_step(path: Path):
    try:
        import cadquery as cq
    except ImportError as exc:
        raise CADToolError("cadquery is required to read STEP files (pip install cadquery)") from exc
    wp = cq.importers.importStep(str(path))
    solids = wp.vals()
    if not solids:
        raise CADToolError(f"STEP file contains no shapes: {path}")
    shape = solids[0] if len(solids) == 1 else cq.Compound.makeCompound(solids)
    verts, tris = shape.tessellate(0.05, 0.1)  # linear (mm) / angular (rad) tolerance
    if not tris:
        raise CADToolError("STEP tessellation produced no triangles")
    mesh = trimesh.Trimesh(
        vertices=np.array([[v.x, v.y, v.z] for v in verts], dtype=float),
        faces=np.array(tris, dtype=np.int64),
        process=True,
    )
    com = shape.Center()
    return mesh, {
        "format": "step",
        "exact_volume_mm3": float(shape.Volume()),
        "exact_com_mm": [float(com.x), float(com.y), float(com.z)],
        "warnings": ["STEP files are interpreted as millimetres."],
    }


_MESH_CACHE: dict[tuple, tuple[trimesh.Trimesh, dict]] = {}


def load_mesh(file_path: str, units: str) -> tuple[trimesh.Trimesh, dict]:
    """Load + validate a CAD file as a watertight mesh in mm. Cached per (path, mtime, units)."""
    path = Path(file_path)
    if not path.is_file():
        raise CADToolError(f"CAD file not found: {file_path}")
    key = (str(path.resolve()), path.stat().st_mtime_ns, units)
    if key in _MESH_CACHE:
        return _MESH_CACHE[key]

    ext = path.suffix.lower()
    if ext == ".stl":
        mesh, meta = _load_stl(path, units)
    elif ext in (".step", ".stp"):
        mesh, meta = _load_step(path)
    else:
        raise CADToolError(f"Unsupported CAD format '{ext}' (supported: .stl, .step, .stp)")

    if not mesh.is_watertight:
        raise CADToolError(
            "Mesh is not watertight; volume, mass and centre of gravity would be meaningless. "
            "Repair the model in your CAD package and re-export."
        )
    if not mesh.is_winding_consistent:
        mesh.fix_normals()
        meta["warnings"].append("Face winding was inconsistent and was re-oriented.")
    if mesh.volume < 0:
        mesh.invert()
        meta["warnings"].append("Mesh normals pointed inward and were inverted.")
    if mesh.volume <= 0:
        raise CADToolError("Mesh has zero or negative volume.")

    if len(_MESH_CACHE) >= 4:
        _MESH_CACHE.clear()
    _MESH_CACHE[key] = (mesh, meta)
    return mesh, meta


def extract_geometry(
    file_path: str,
    units: str = "mm",
    material: str | None = None,
    density_kg_m3: float | None = None,
) -> dict:
    mesh, meta = load_mesh(file_path, units)
    warnings = list(meta["warnings"])

    if meta["exact_volume_mm3"] is not None:
        volume, volume_source = meta["exact_volume_mm3"], "b-rep"
        cog = meta["exact_com_mm"]
    else:
        volume, volume_source = float(mesh.volume), "mesh"
        cog = [float(c) for c in mesh.center_mass]  # uniform-density assumption

    density, mat_key = None, None
    if density_kg_m3 is not None:
        density, mat_key = float(density_kg_m3), material or "user_density"
    elif material:
        mat_key = material.strip().lower().replace("-", "_").replace(" ", "_")
        if mat_key not in MATERIAL_DENSITY_KG_M3:
            raise CADToolError(
                f"Unknown material '{material}'. Known: {sorted(MATERIAL_DENSITY_KG_M3)} "
                "- or pass density_kg_m3 explicitly."
            )
        density = MATERIAL_DENSITY_KG_M3[mat_key]
    else:
        warnings.append("No material/density supplied: mass_kg is null and weight rules cannot be verified.")

    mass_kg = volume * 1e-9 * density if density is not None else None  # mm^3 -> m^3
    lo, hi = mesh.bounds
    return GeometryProperties(
        source_format=meta["format"],
        volume_source=volume_source,
        volume_mm3=volume,
        surface_area_mm2=float(mesh.area),
        bounding_box_xyz_mm=[float(x) for x in (hi - lo)],
        bbox_min_mm=[float(x) for x in lo],
        bbox_max_mm=[float(x) for x in hi],
        center_of_gravity_mm=[float(x) for x in cog],
        material=mat_key,
        density_kg_m3=density,
        mass_kg=mass_kg,
        triangle_count=int(len(mesh.faces)),
        is_watertight=bool(mesh.is_watertight),
        warnings=warnings,
    ).model_dump()


def flatten_properties(props: dict) -> dict[str, float]:
    """Flat, LLM-addressable view of the tool-produced properties (all mm / kg)."""
    flat: dict[str, float] = {
        "volume_mm3": props["volume_mm3"],
        "surface_area_mm2": props["surface_area_mm2"],
        "bbox_max_dimension_mm": max(props["bounding_box_xyz_mm"]),
    }
    if props.get("mass_kg") is not None:
        flat["mass_kg"] = props["mass_kg"]
    for axis, v in zip("xyz", props["bounding_box_xyz_mm"]):
        flat[f"bbox_{axis}_mm"] = v
    for axis, v in zip("xyz", props["center_of_gravity_mm"]):
        flat[f"cog_{axis}_mm"] = v
    return flat


# ---- 2b. Stress-riser heuristics --------------------------------------------
class _MeshToolInput(BaseModel):
    file_path: str = Field(description="CAD file path. Overridden by the pipeline.")
    units: Literal["mm", "cm", "m", "in"] = Field("mm", description="Overridden by the pipeline.")


class FilletCheckInput(_MeshToolInput):
    sharp_angle_deg: float = Field(60.0, ge=20, le=170, description="Concave edges turning more than this are 'sharp'.")
    max_hotspots: int = Field(8, ge=1, le=25)


class CrossSectionInput(_MeshToolInput):
    num_slices: int = Field(80, ge=20, le=300)
    area_ratio_threshold: float = Field(1.6, ge=1.1, le=10.0, description="Adjacent-slice area ratio treated as a sudden change.")
    max_hotspots: int = Field(8, ge=1, le=25)


class ThinWallInput(_MeshToolInput):
    num_samples: int = Field(3000, ge=500, le=20000)
    min_wall_mm: float = Field(1.5, gt=0, description="Absolute thin-wall floor.")
    relative_wall_fraction: float = Field(0.25, gt=0, lt=1, description="Thin if below this fraction of the part's median wall.")
    max_hotspots: int = Field(8, ge=1, le=25)


class StressHotspotModel(BaseModel):
    heuristic: Literal["sharp_internal_radius", "section_change", "thin_wall"]
    location: dict[str, Any]
    estimated_severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    severity_score: float = Field(ge=0, le=1)
    geometric_reason: str


class HeuristicResult(BaseModel):
    tool: str
    summary: dict[str, Any]
    hotspots: list[StressHotspotModel]


def _severity_label(score: float) -> str:
    return "LOW" if score < 0.4 else "MEDIUM" if score < 0.65 else "HIGH" if score < 0.85 else "CRITICAL"


def _cluster_points(points: np.ndarray, scores: np.ndarray, cell: float, top_n: int) -> list[dict]:
    """Voxel-grid clustering so thousands of mesh edges collapse into a few hotspots."""
    keys = np.floor(points / max(cell, 1e-9)).astype(np.int64)
    buckets: dict[tuple, list[int]] = {}
    for i, k in enumerate(map(tuple, keys)):
        buckets.setdefault(k, []).append(i)
    clusters = []
    for idx_list in buckets.values():
        idx = np.array(idx_list)
        peak = int(idx[np.argmax(scores[idx])])
        clusters.append({"idx": idx, "peak": peak, "centroid": points[idx].mean(axis=0), "score": float(scores[peak])})
    clusters.sort(key=lambda c: -c["score"])
    return clusters[:top_n]


def check_sharp_internal_radii(file_path: str, units: str = "mm", sharp_angle_deg: float = 60.0, max_hotspots: int = 8) -> dict:
    """Concave mesh edges with a large dihedral turn == internal corners with ~zero fillet radius."""
    mesh, _ = load_mesh(file_path, units)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    angles = np.degrees(mesh.face_adjacency_angles)
    concave = ~mesh.face_adjacency_convex
    mask = concave & (angles >= sharp_angle_deg)

    hotspots: list[StressHotspotModel] = []
    if mask.any():
        seg = mesh.vertices[mesh.face_adjacency_edges[mask]]  # (m, 2, 3)
        mids = seg.mean(axis=1)
        lengths = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1)
        ang = angles[mask]
        scores = 0.5 + 0.5 * np.clip((ang - sharp_angle_deg) / max(150.0 - sharp_angle_deg, 1.0), 0, 1)
        for c in _cluster_points(mids, scores, cell=0.02 * diag, top_n=max_hotspots * 3):
            total_len = float(lengths[c["idx"]].sum())
            if total_len < 0.01 * diag:  # tessellation noise
                continue
            peak_angle = float(ang[c["peak"]])
            hotspots.append(
                StressHotspotModel(
                    heuristic="sharp_internal_radius",
                    location={"xyz_mm": [round(float(x), 2) for x in c["centroid"]]},
                    estimated_severity=_severity_label(c["score"]),
                    severity_score=round(c["score"], 3),
                    geometric_reason=(
                        f"Concave edge with a {peak_angle:.0f} deg face-normal turn and no resolvable fillet "
                        f"(~{total_len:.1f} mm of edge length). Sharp internal corners concentrate stress."
                    ),
                )
            )
    hotspots = sorted(hotspots, key=lambda h: -h.severity_score)[:max_hotspots]
    return HeuristicResult(
        tool="check_sharp_internal_radii",
        summary={
            "concave_edges_total": int(concave.sum()),
            "sharp_concave_edges": int(mask.sum()),
            "sharp_angle_deg": sharp_angle_deg,
            "limitation": "Radii below the tessellation tolerance appear sharp; use a fine STEP/STL export.",
        },
        hotspots=hotspots,
    ).model_dump()


def _section_area(section) -> float:
    try:
        if hasattr(section, "to_2D"):            # Path3D (trimesh 4.x)
            planar, _ = section.to_2D()
        elif hasattr(section, "to_planar"):      # Path3D (older trimesh)
            planar, _ = section.to_planar()
        else:                                    # already Path2D (trimesh 5.x)
            planar = section
        return float(planar.area)
    except Exception:  # degenerate slice
        return 0.0


def check_section_changes(
    file_path: str, units: str = "mm", num_slices: int = 80, area_ratio_threshold: float = 1.6, max_hotspots: int = 8
) -> dict:
    """Slice along X/Y/Z; flag adjacent slices whose cross-sectional area jumps by a large ratio."""
    mesh, _ = load_mesh(file_path, units)
    centre = mesh.bounds.mean(axis=0)
    found: list[StressHotspotModel] = []
    profile: dict[str, Any] = {}

    for axis_i, axis in enumerate("xyz"):
        lo, hi = float(mesh.bounds[0][axis_i]), float(mesh.bounds[1][axis_i])
        length = hi - lo
        if length < 1e-6:
            continue
        normal = np.zeros(3)
        normal[axis_i] = 1.0
        heights = np.linspace(lo + 0.02 * length, hi - 0.02 * length, num_slices)
        sections = mesh.section_multiplane(plane_origin=np.zeros(3), plane_normal=normal, heights=heights)
        areas = np.array([_section_area(s) if s is not None else 0.0 for s in sections])
        if areas.max() <= 0:
            continue
        profile[axis] = {"min_area_mm2": round(float(areas.min()), 2), "max_area_mm2": round(float(areas.max()), 2)}
        floor = 0.02 * areas.max()  # ignore tips / part ends
        for i in range(len(areas) - 1):
            a, b = areas[i], areas[i + 1]
            small, big = min(a, b), max(a, b)
            if small < floor or big / small < area_ratio_threshold:
                continue
            ratio = big / small
            score = 0.4 + 0.6 * float(np.clip(math.log(ratio / area_ratio_threshold) / math.log(4), 0, 1))
            pos = float((heights[i] + heights[i + 1]) / 2)
            xyz = centre.copy()
            xyz[axis_i] = pos
            found.append(
                StressHotspotModel(
                    heuristic="section_change",
                    location={"axis": axis, "axis_position_mm": round(pos, 2), "xyz_mm": [round(float(x), 2) for x in xyz]},
                    estimated_severity=_severity_label(score),
                    severity_score=round(score, 3),
                    geometric_reason=(
                        f"Cross-sectional area changes by a factor of {ratio:.2f} over one slice step along {axis.upper()} "
                        f"(from {a:.0f} to {b:.0f} mm^2). Abrupt section changes disturb the load path."
                    ),
                )
            )
    found.sort(key=lambda h: -h.severity_score)
    return HeuristicResult(
        tool="check_section_changes",
        summary={"num_slices": num_slices, "area_ratio_threshold": area_ratio_threshold, "area_profile_by_axis": profile},
        hotspots=found[:max_hotspots],
    ).model_dump()


def check_thin_walls(
    file_path: str,
    units: str = "mm",
    num_samples: int = 3000,
    min_wall_mm: float = 1.5,
    relative_wall_fraction: float = 0.25,
    max_hotspots: int = 8,
) -> dict:
    """Ray-cast inward from surface samples to estimate local wall thickness."""
    mesh, _ = load_mesh(file_path, units)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    try:
        pts, face_idx = trimesh.sample.sample_surface(mesh, num_samples, seed=0)
    except TypeError:  # older trimesh without seed kwarg
        pts, face_idx = trimesh.sample.sample_surface(mesh, num_samples)
    normals = mesh.face_normals[face_idx]
    origins = pts - normals * (1e-4 * diag)
    locs, ray_idx, _ = mesh.ray.intersects_location(origins, -normals, multiple_hits=False)
    thickness = np.full(len(pts), np.nan)
    thickness[ray_idx] = np.linalg.norm(locs - origins[ray_idx], axis=1)
    valid = ~np.isnan(thickness)
    if valid.sum() < 0.5 * len(pts):
        raise CADToolError(f"Wall-thickness probing failed for {int((~valid).sum())}/{len(pts)} samples (check mesh quality / rtree install).")

    median = float(np.nanmedian(thickness))
    threshold = max(min_wall_mm, relative_wall_fraction * median)
    thin = valid & (thickness < threshold)
    hotspots: list[StressHotspotModel] = []
    if thin.any():
        scores = np.zeros(len(pts))
        scores[thin] = 0.4 + 0.6 * np.clip(1 - thickness[thin] / threshold, 0, 1)
        for c in _cluster_points(pts[thin], scores[thin], cell=0.04 * diag, top_n=max_hotspots):
            t_local = float(thickness[thin][c["peak"]])
            hotspots.append(
                StressHotspotModel(
                    heuristic="thin_wall",
                    location={"xyz_mm": [round(float(x), 2) for x in c["centroid"]]},
                    estimated_severity=_severity_label(c["score"]),
                    severity_score=round(c["score"], 3),
                    geometric_reason=(
                        f"Local wall thickness ~{t_local:.2f} mm versus a part-median of {median:.2f} mm "
                        f"(thin-wall threshold {threshold:.2f} mm). Thin regions next to thicker mass risk buckling, "
                        "distortion and strain localisation."
                    ),
                )
            )
    return HeuristicResult(
        tool="check_thin_walls",
        summary={
            "samples_valid": int(valid.sum()),
            "median_wall_mm": round(median, 3),
            "p5_wall_mm": round(float(np.nanpercentile(thickness, 5)), 3),
            "thin_threshold_mm": round(threshold, 3),
            "thin_sample_fraction": round(float(thin.sum() / valid.sum()), 4),
        },
        hotspots=hotspots,
    ).model_dump()


# ---- 2c. External FEA stub ---------------------------------------------------
class FEAJobRequest(BaseModel):
    file_path: str
    solver: Literal["calculix"] = "calculix"
    material: str | None = None
    target_hotspot_ids: list[str] = []
    mesh_size_mm: float = Field(5.0, gt=0)


class FEAJobStatus(BaseModel):
    job_id: str | None = None
    status: Literal["NOT_CONFIGURED", "SUBMITTED", "FAILED"]
    message: str


def trigger_external_fea(req: FEAJobRequest) -> FEAJobStatus:
    """STUB. Wire this to your solver service.

    A real implementation would: (1) volume-mesh the STEP (gmsh) at req.mesh_size_mm refined near the
    hotspots, (2) write a CalculiX .inp with materials, boundary conditions and loads, (3) submit it, and
    (4) have a later graph node poll the job and parse the .frd for von Mises stress. Loads and constraints
    must come from the engineer - they cannot be inferred from geometry.

        resp = requests.post(f"{os.environ['FEA_SOLVER_URL']}/jobs", json=req.model_dump(), timeout=30)
        resp.raise_for_status()
        return FEAJobStatus(job_id=resp.json()["id"], status="SUBMITTED", message="queued")
    """
    return FEAJobStatus(
        status="NOT_CONFIGURED",
        message="External FEA solver is not wired up; only geometric heuristics were run. No stress values exist.",
    )


def _build_tool(func, name: str, description: str, schema: type[BaseModel]) -> StructuredTool:
    return StructuredTool.from_function(func=func, name=name, description=description, args_schema=schema)


EXTRACT_TOOL = _build_tool(
    extract_geometry, "extract_geometry",
    "Parse the CAD file and return volume, mass, bounding box and centre of gravity.", ExtractGeometryInput,
)
STRESS_TOOLS = [
    _build_tool(check_sharp_internal_radii, "check_sharp_internal_radii",
                "Find concave corners with no fillet (stress risers).", FilletCheckInput),
    _build_tool(check_section_changes, "check_section_changes",
                "Find abrupt cross-sectional area changes along X/Y/Z.", CrossSectionInput),
    _build_tool(check_thin_walls, "check_thin_walls",
                "Find walls that are thin in absolute terms or relative to the part's bulk.", ThinWallInput),
]


def execute_tool_calls(ai_msg, tools: dict[str, StructuredTool], locked: dict[str, Any]) -> list[tuple[dict, dict]]:
    """Run the LLM's tool calls. `locked` values (path, units, material...) always win over LLM args."""
    results = []
    for call in ai_msg.tool_calls:
        tool = tools.get(call["name"])
        if tool is None:
            raise CADToolError(f"LLM requested unknown tool '{call['name']}'")
        fields = tool.args_schema.model_fields
        args = {**call["args"], **{k: v for k, v in locked.items() if k in fields}}
        results.append((call, tool.invoke(args)))  # pydantic-validates args; raises on bad input
    return results


# =============================================================================
# 3. System prompts
# =============================================================================
GEOMETRY_SYSTEM_PROMPT = """\
You are the Geometry Extractor agent in a deterministic mechanical-CAD review pipeline.

Your ONLY job: call the `extract_geometry` tool exactly once for the CAD file you are given.

Hard rules:
1. Never compute, estimate, round, convert or restate any mass, volume, dimension or coordinate. Every number in this
   pipeline must come from the tool output. You are forbidden from guessing physical properties.
2. The file path, units, material and density are fixed by the pipeline. Pass them through exactly as given. Do not
   infer a material or density from the file name, the part description or your own knowledge.
3. Output no prose. If you cannot call the tool, say nothing - the pipeline will treat that as a failure.
"""

COMPLIANCE_SYSTEM_PROMPT = """\
You are the Compliance Checker agent for a BAJA SAE-style mechanical design review.

You receive (a) numbered rulebook excerpts [R1], [R2]... retrieved from the official rulebook and (b) a dictionary of
MEASURED PROPERTIES produced by CAD tools. Produce one finding per distinct, geometry-checkable rule.

Rules of engagement:
1. Use ONLY the supplied excerpts. Never cite a rule that is not in them and never rely on remembered rulebook content.
2. Use ONLY the supplied measured properties. `property_key` must be one of the provided keys. Copy `measured_value`
   exactly; never estimate, round or invent a measurement. If a needed property is absent (e.g. mass_kg when no
   material was supplied), the status is NOT_VERIFIABLE.
3. `limit_value` must be in the same unit as `property_key` (mm, mm^2, mm^3 or kg). Converting the rule's own units
   (e.g. metres -> mm) is allowed; quote the rule's wording in `limit_text` so a human can audit it.
4. PASS or FAIL requires a numeric comparator and limit. Rules that cannot be decided from geometry alone (material
   certification, weld procedure, tube wall specification, etc.) are NOT_VERIFIABLE - never PASS them optimistically.
5. Be conservative: if a rule is ambiguous, use NOT_VERIFIABLE and explain the ambiguity in `reasoning`.
Your arithmetic will be re-checked by code; wrong verdicts are overwritten and logged.
"""

STRESS_SYSTEM_PROMPT = """\
You are the Stress/FEA Review agent. You screen geometry for stress concentrators using Python heuristic tools.

Procedure:
1. Call ALL THREE tools: `check_sharp_internal_radii`, `check_section_changes`, `check_thin_walls`. Defaults are
   fine; adjust a threshold only if the part description clearly justifies it, within the schema bounds.
2. After the tool results arrive, write a qualitative note (max 120 words) grouping the hotspots by likely
   load-path interaction (e.g. a thin web next to a sharp corner).

Hard rules:
- File path and units are fixed by the pipeline. Do not change them.
- Do not introduce any number that is not in a tool result. Never state a stress value (MPa), safety factor or
  fatigue life: these heuristics do not compute stress.
- Never describe the design as "safe" or "passing". You are screening; a solver or an engineer decides.
"""

REPORTER_SYSTEM_PROMPT = """\
You are the Reporter agent writing the narrative sections of an engineering design-review report.

You receive JSON with verified properties, compliance findings, stress hotspots and the engineer's review decision.
Tables are generated by code; you write ONLY:
- `executive_summary`: at most 120 words.
- `recommendations`: 3-7 concrete, prioritised actions.

Hard rules:
1. Do not introduce any number that does not appear verbatim in the JSON (no new totals, percentages or conversions).
   Write numbers without thousands separators.
2. Reference hotspots by id (e.g. H2) and rules by their rule_ref.
3. State clearly which rules were NOT_VERIFIABLE and that heuristics are screening tools, not FEA results.
4. Do not alter or second-guess the verdict, and never claim a certification or approval authority.
5. Respect the engineer's dismissals; mention dismissed hotspots only as dismissed.
"""


# =============================================================================
# 4. Rulebook RAG
# =============================================================================
def _embeddings():
    s = Settings.from_env()
    if s.embedding_provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=s.gemini_embedding_model)
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=s.embedding_model)


def ingest_rulebook(path: str) -> int:
    """Index a rulebook (.pdf / .txt / .md, or a directory of them) into Chroma. Returns chunk count."""
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    s = Settings.from_env()
    p = Path(path)
    files = sorted(f for f in (p.rglob("*") if p.is_dir() else [p]) if f.suffix.lower() in {".pdf", ".txt", ".md"})
    if not files:
        raise CADToolError(f"No .pdf/.txt/.md rulebook files found at {path}")
    docs = []
    for f in files:
        if f.suffix.lower() == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader

            docs.extend(PyPDFLoader(str(f)).load())
        else:
            from langchain_community.document_loaders import TextLoader

            docs.extend(TextLoader(str(f), encoding="utf-8").load())
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
    Chroma.from_documents(
        chunks, _embeddings(), collection_name=s.rulebook_collection, persist_directory=s.rulebook_db_dir
    )
    return len(chunks)


def get_rulebook_retriever(k: int = 6):
    from langchain_chroma import Chroma

    s = Settings.from_env()
    store = Chroma(
        collection_name=s.rulebook_collection, embedding_function=_embeddings(), persist_directory=s.rulebook_db_dir
    )
    if store._collection.count() == 0:  # noqa: SLF001 - no public count API
        raise CADToolError("Rulebook index is empty. Run: python cad_companion.py ingest <rulebook>")
    return store.as_retriever(search_kwargs={"k": k})


# =============================================================================
# 5. Compliance schemas + validators
# =============================================================================
class RuleFinding(BaseModel):
    rule_ref: str = Field(description="Excerpt tag such as 'R3'.")
    rule_checked: str = Field(description="One-sentence paraphrase of the rule.")
    limit_text: str | None = Field(None, description="Short verbatim quote of the limit from the excerpt.")
    property_key: str | None = Field(None, description="One of the provided measured-property keys.")
    measured_value: float | None = Field(None, description="Copied from the measured properties.")
    comparator: Literal["<=", ">=", "<", ">", "=="] | None = None
    limit_value: float | None = Field(None, description="Limit in the same unit as property_key.")
    status: Literal["PASS", "FAIL", "NOT_VERIFIABLE"]
    reasoning: str


class ComplianceReport(BaseModel):
    findings: list[RuleFinding]


def _compare(value: float, comparator: str, limit: float) -> bool:
    return {
        "<=": value <= limit, ">=": value >= limit, "<": value < limit, ">": value > limit,
        "==": math.isclose(value, limit, rel_tol=1e-6),
    }[comparator]


def validate_finding(f: RuleFinding, flat: dict[str, float], refs: dict[str, dict]) -> dict:
    """Never trust the LLM's numbers or arithmetic: replace with tool values and recompute."""
    notes: list[str] = []
    status, measured = f.status, None

    if f.rule_ref not in refs:
        notes.append(f"rule_ref '{f.rule_ref}' is not among the retrieved excerpts")
        status = "NOT_VERIFIABLE"
    if status in ("PASS", "FAIL"):
        if f.property_key not in flat:
            notes.append(f"property_key '{f.property_key}' is not a measured property")
            status = "NOT_VERIFIABLE"
        elif f.comparator is None or f.limit_value is None:
            notes.append("PASS/FAIL requires comparator and limit_value")
            status = "NOT_VERIFIABLE"
        else:
            measured = flat[f.property_key]
            if f.measured_value is not None and not math.isclose(f.measured_value, measured, rel_tol=1e-3, abs_tol=1e-6):
                notes.append(f"LLM-reported value {f.measured_value} replaced by tool value {measured:.6g}")
            recomputed = "PASS" if _compare(measured, f.comparator, f.limit_value) else "FAIL"
            if recomputed != status:
                notes.append(f"LLM verdict {status} overridden by deterministic comparison -> {recomputed}")
                status = recomputed
    elif f.property_key in flat:
        measured = flat[f.property_key]

    return {
        "rule_ref": f.rule_ref,
        "rule_checked": f.rule_checked,
        "status": status,
        "reasoning": f.reasoning,
        "source": refs.get(f.rule_ref, {}).get("source"),
        "limit_text": f.limit_text,
        "property_key": f.property_key,
        "measured_value": measured,
        "comparator": f.comparator,
        "limit_value": f.limit_value,
        "validation_notes": notes,
    }


# =============================================================================
# 6. Graph nodes
# =============================================================================
def safe_node(step: str):
    """Any exception inside a worker becomes an explicit failure state (no silent fallbacks)."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(state: AgentState) -> dict:
            try:
                update = fn(state)
            except Exception as exc:  # noqa: BLE001
                log.exception("Node %s failed", step)
                return {"error": f"[{step}] {type(exc).__name__}: {exc}", "current_step": step}
            update["completed"] = [*state.get("completed", []), step]
            update.setdefault("current_step", step)
            return update

        return wrapper

    return deco


PIPELINE = ["geometry_extractor", "compliance_checker", "stress_reviewer", "human_review", "reporter"]


def orchestrator_node(state: AgentState) -> dict:
    """Hub. Routing happens in route_from_orchestrator; here we just record that control returned."""
    return {"current_step": "orchestrator"}


def route_from_orchestrator(state: AgentState) -> str:
    if state.get("error"):
        return "failure_handler"
    review = state.get("human_review") or {}
    if review and not review.get("approved"):
        return "failure_handler"
    for step in PIPELINE:
        if step not in state["completed"]:
            return step
    return END


# ---- 6a. Geometry Extractor ----------------------------------------------------
@safe_node("geometry_extractor")
def geometry_extractor_node(state: AgentState) -> dict:
    llm = get_llm()
    try:
        llm = llm.bind_tools([EXTRACT_TOOL], tool_choice="extract_geometry")
    except (TypeError, ValueError, NotImplementedError):  # provider does not support forcing a tool
        llm = llm.bind_tools([EXTRACT_TOOL])
    ai = llm.invoke([
        SystemMessage(GEOMETRY_SYSTEM_PROMPT),
        HumanMessage(json.dumps({
            "file_path": state["cad_file_path"], "units": state["units"],
            "material": state["material"], "density_kg_m3": state["density_kg_m3"],
        })),
    ])
    if not ai.tool_calls:
        raise CADToolError("Geometry agent did not call extract_geometry; refusing to proceed without measurements.")
    locked = {"file_path": state["cad_file_path"], "units": state["units"],
              "material": state["material"], "density_kg_m3": state["density_kg_m3"]}
    (_, props), *_ = execute_tool_calls(ai, {EXTRACT_TOOL.name: EXTRACT_TOOL}, locked)
    return {"extracted_properties": props}


# ---- 6b. Compliance Checker ----------------------------------------------------
@safe_node("compliance_checker")
def compliance_checker_node(state: AgentState) -> dict:
    props = state["extracted_properties"]
    flat = flatten_properties(props)
    retriever = get_rulebook_retriever()

    queries = [
        "maximum overall vehicle or component dimensions length width height envelope",
        "weight mass limit restrictions",
        "material requirements steel tubing grade wall thickness",
        "structural member geometry requirements fillets welds gussets",
    ]
    if state["part_description"]:
        queries.append(f"{state['part_description']} requirements")

    seen, chunks = set(), []
    for q in queries:
        for doc in retriever.invoke(q):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                chunks.append(doc)
    chunks = chunks[:14]
    refs = {f"R{i}": {"source": f"{d.metadata.get('source', '?')} p.{d.metadata.get('page', '?')}"}
            for i, d in enumerate(chunks, 1)}
    excerpts = "\n\n".join(f"[R{i}] ({refs[f'R{i}']['source']})\n{d.page_content}" for i, d in enumerate(chunks, 1))

    llm = get_llm().with_structured_output(ComplianceReport)
    report = llm.invoke([
        SystemMessage(COMPLIANCE_SYSTEM_PROMPT),
        HumanMessage(
            f"Part description: {state['part_description'] or '(none given)'}\n\n"
            f"MEASURED PROPERTIES (mm / kg; authoritative):\n{json.dumps(flat, indent=2)}\n\n"
            f"RULEBOOK EXCERPTS:\n{excerpts}"
        ),
    ])
    if report is None:
        raise RuntimeError("Compliance LLM returned no structured output.")
    findings = [validate_finding(f, flat, refs) for f in report.findings]
    if not findings:
        findings = [{
            "rule_ref": None, "rule_checked": "No geometry-checkable rules found in retrieved excerpts",
            "status": "NOT_VERIFIABLE", "reasoning": "Retrieval returned nothing the model could evaluate.",
            "source": None, "limit_text": None, "property_key": None, "measured_value": None,
            "comparator": None, "limit_value": None, "validation_notes": [],
        }]
    return {"compliance_status": findings}


# ---- 6c. Stress / FEA Review -----------------------------------------------------
@safe_node("stress_reviewer")
def stress_reviewer_node(state: AgentState) -> dict:
    tools = {t.name: t for t in STRESS_TOOLS}
    locked = {"file_path": state["cad_file_path"], "units": state["units"]}
    llm = get_llm().bind_tools(STRESS_TOOLS)

    messages: list[Any] = [
        SystemMessage(STRESS_SYSTEM_PROMPT),
        HumanMessage(json.dumps({"part_description": state["part_description"], **locked,
                                 "bounding_box_xyz_mm": state["extracted_properties"]["bounding_box_xyz_mm"]})),
    ]
    results: dict[str, dict] = {}
    notes = ""
    for _ in range(MAX_STRESS_TOOL_ROUNDS):
        ai = llm.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            txt = getattr(ai, "text", None)  # property in langchain-core >= 1.0, method before
            notes = (txt if isinstance(txt, str) else txt() if callable(txt) else "") or str(ai.content)
            break
        for call, res in execute_tool_calls(ai, tools, locked):
            results[call["name"]] = res
            messages.append(ToolMessage(content=json.dumps(res), tool_call_id=call["id"]))

    # Coverage guarantee: any heuristic the LLM skipped is run with defaults.
    for name, tool in tools.items():
        if name not in results:
            results[name] = tool.invoke(locked)

    merged = [h for r in results.values() for h in r["hotspots"]]
    merged.sort(key=lambda h: -h["severity_score"])
    hotspots = [{"id": f"H{i}", **h, "reviewer_status": "pending"} for i, h in enumerate(merged, 1)]

    fea = trigger_external_fea(FEAJobRequest(
        file_path=state["cad_file_path"], material=state["material"],
        target_hotspot_ids=[h["id"] for h in hotspots if h["estimated_severity"] in ("HIGH", "CRITICAL")],
    ))
    return {
        "stress_hotspots": hotspots,
        "stress_notes": notes.strip(),
        "fea_job": {**fea.model_dump(), "heuristic_summaries": {k: v["summary"] for k, v in results.items()}},
    }


# ---- 6d. Human-in-the-loop gate (runs BEFORE the reporter) -----------------------
def human_review_node(state: AgentState) -> dict:
    """Pauses the graph. Resume with Command(resume={"approved": bool, "reviewer": str, "notes": str,
    "dismiss_hotspots": ["H3", ...]}). Note: LangGraph re-runs this node from the top on resume, so
    nothing before interrupt() may have side effects."""
    decision = interrupt({
        "instruction": "Review extracted data, compliance findings and stress hotspots; approve to generate the report.",
        "extracted_properties": state["extracted_properties"],
        "compliance_status": state["compliance_status"],
        "stress_hotspots": state["stress_hotspots"],
        "stress_notes": state["stress_notes"],
        "fea_job": state["fea_job"],
    })
    if not isinstance(decision, dict):
        decision = {"approved": False, "notes": "Malformed review response; treated as rejection."}
    dismissed = set(decision.get("dismiss_hotspots") or [])
    hotspots = [{**h, "reviewer_status": "dismissed" if h["id"] in dismissed else "active"}
                for h in state["stress_hotspots"]]
    review = {
        "approved": bool(decision.get("approved")),
        "reviewer": str(decision.get("reviewer") or "unknown"),
        "notes": str(decision.get("notes") or ""),
        "dismissed_hotspots": sorted(dismissed),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return {
        "human_review": review,
        "stress_hotspots": hotspots,
        "completed": [*state["completed"], "human_review"],
        "current_step": "human_review",
    }


# ---- 6e. Reporter -----------------------------------------------------------------
class ReportNarrative(BaseModel):
    executive_summary: str
    recommendations: list[str] = Field(min_length=1)


_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def ungrounded_numbers(text: str, context: Any) -> list[str]:
    """Numbers in `text` that don't appear (to the stated precision) in `context`. Small integers are exempt."""
    ctx = [float(x) for x in _NUM.findall(json.dumps(context, default=str))]
    bad = []
    for tok in _NUM.findall(text):
        val = float(tok)
        if "." not in tok and abs(val) <= 12:
            continue
        d = len(tok.split(".")[1]) if "." in tok else 0
        if not any(math.isclose(round(c, d), val, abs_tol=1e-9) for c in ctx):
            bad.append(tok)
    return bad


def compute_verdict(state: AgentState) -> str:
    if any(f["status"] == "FAIL" for f in state["compliance_status"]):
        return "NON_COMPLIANT"
    active_serious = any(h["reviewer_status"] != "dismissed" and h["estimated_severity"] in ("HIGH", "CRITICAL")
                         for h in state["stress_hotspots"])
    unverified = any(f["status"] == "NOT_VERIFIABLE" for f in state["compliance_status"])
    return "APPROVED_WITH_WARNINGS" if (active_serious or unverified) else "APPROVED"


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    esc = lambda v: str(v).replace("|", "\\|").replace("\n", " ")  # noqa: E731
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _fallback_narrative(state: AgentState, verdict: str) -> ReportNarrative:
    n_fail = sum(f["status"] == "FAIL" for f in state["compliance_status"])
    n_unv = sum(f["status"] == "NOT_VERIFIABLE" for f in state["compliance_status"])
    n_hot = sum(h["reviewer_status"] != "dismissed" for h in state["stress_hotspots"])
    return ReportNarrative(
        executive_summary=(f"Verdict: {verdict}. Compliance failures: {n_fail}. Rules not verifiable from geometry: {n_unv}. "
                           f"Active stress hotspots: {n_hot}. Heuristic screening only - no FEA stresses were computed."),
        recommendations=["Resolve every FAIL finding before release.",
                         "Manually verify all NOT_VERIFIABLE rules against the rulebook.",
                         "Add fillets or smooth transitions at HIGH/CRITICAL hotspots, or confirm them with FEA."],
    )


@safe_node("reporter")
def reporter_node(state: AgentState) -> dict:
    verdict = compute_verdict(state)
    props, flat = state["extracted_properties"], flatten_properties(state["extracted_properties"])
    context = {"verdict": verdict, "properties": flat, "compliance": state["compliance_status"],
               "hotspots": state["stress_hotspots"], "review": state["human_review"], "fea": state["fea_job"]}

    llm = get_llm().with_structured_output(ReportNarrative)
    narrative: ReportNarrative | None = None
    feedback = ""
    for _ in range(2):  # one retry if the model invents numbers
        cand = llm.invoke([SystemMessage(REPORTER_SYSTEM_PROMPT),
                           HumanMessage(json.dumps(context, default=str) + feedback)])
        bad = ungrounded_numbers(cand.executive_summary + " " + " ".join(cand.recommendations), context) if cand else ["?"]
        if cand and not bad:
            narrative = cand
            break
        feedback = f"\n\nYour previous draft contained numbers not present in the JSON: {bad}. Remove them."
        log.warning("Reporter draft rejected for ungrounded numbers: %s", bad)
    if narrative is None:
        narrative = _fallback_narrative(state, verdict)

    s = Settings.from_env()
    now = datetime.now(timezone.utc)
    bb = props["bounding_box_xyz_mm"]
    prop_rows = [
        ["Mass (kg)", f"{props['mass_kg']:.4f}" if props["mass_kg"] is not None else "n/a (no material/density supplied)"],
        ["Material / density", f"{props['material'] or 'n/a'} / {props['density_kg_m3'] or 'n/a'} kg/m^3"],
        ["Volume (mm^3)", f"{props['volume_mm3']:.2f} ({props['volume_source']})"],
        ["Surface area (mm^2)", f"{props['surface_area_mm2']:.2f}"],
        ["Bounding box X/Y/Z (mm)", f"{bb[0]:.2f} x {bb[1]:.2f} x {bb[2]:.2f}"],
        ["Centre of gravity (mm)", ", ".join(f"{c:.2f}" for c in props["center_of_gravity_mm"])],
    ]
    comp_rows = [[f.get("rule_ref") or "-", f["rule_checked"], f["status"],
                  (f"{f['property_key']} = {f['measured_value']:.6g} {f['comparator']} {f['limit_value']:g}"
                   if f["measured_value"] is not None and f["limit_value"] is not None else "-"),
                  f["reasoning"] + (f" _[validator: {'; '.join(f['validation_notes'])}]_" if f["validation_notes"] else "")]
                 for f in state["compliance_status"]]
    active = [h for h in state["stress_hotspots"] if h["reviewer_status"] != "dismissed"]
    dismissed = [h for h in state["stress_hotspots"] if h["reviewer_status"] == "dismissed"]
    hot_rows = lambda hs: [[h["id"], h["heuristic"], h["estimated_severity"], json.dumps(h["location"]),  # noqa: E731
                            h["geometric_reason"]] for h in hs]
    rv = state["human_review"]

    md = [
        f"# Engineering Review Report - {Path(state['cad_file_path']).name}",
        f"*Generated {now.isoformat(timespec='seconds')} | Verdict: **{verdict}***",
        "> Advisory output. Geometry numbers come from CAD tools; stress findings are geometric heuristics, "
        "not FEA results. Final sign-off belongs to the reviewing engineer.",
        "## Executive summary", narrative.executive_summary,
        "## Extracted properties", _md_table(["Property", "Value"], prop_rows),
        "## Rulebook compliance", _md_table(["Rule", "Requirement", "Status", "Check", "Reasoning"], comp_rows),
        "## Stress-concentration hotspots",
        _md_table(["ID", "Heuristic", "Severity", "Location", "Geometric reason"], hot_rows(active)) if active
        else "_No active hotspots._",
    ]
    if dismissed:
        md += ["### Dismissed by reviewer", _md_table(["ID", "Heuristic", "Severity", "Location", "Geometric reason"], hot_rows(dismissed))]
    if state["stress_notes"]:
        md += ["### Analyst note (LLM, advisory)", state["stress_notes"]]
    md += [
        "## External FEA", f"Status: **{state['fea_job'].get('status')}** - {state['fea_job'].get('message')}",
        "## Recommendations", "\n".join(f"{i}. {r}" for i, r in enumerate(narrative.recommendations, 1)),
        "## Human review",
        f"Reviewer: {rv.get('reviewer')} | Approved: {rv.get('approved')} | {rv.get('timestamp_utc')}\n\nNotes: {rv.get('notes') or '-'}",
    ]
    report = "\n\n".join(md)

    out_dir = Path(s.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{Path(state['cad_file_path']).stem}_{now.strftime('%Y%m%dT%H%M%SZ')}.md"
    path.write_text(report, encoding="utf-8")
    return {"final_report": report, "verdict": verdict, "report_path": str(path)}


# ---- 6f. Failure / rejection terminal ----------------------------------------------
def failure_handler_node(state: AgentState) -> dict:
    if state.get("error"):
        verdict, body = "FAILED", (
            f"# Review FAILED\n\n**{state['error']}**\n\nNo physical properties were estimated or inferred. "
            "Fix the underlying problem (file, rulebook index, API access) and re-run."
        )
    else:
        rv = state["human_review"]
        verdict, body = "REJECTED", (
            f"# Review REJECTED by engineer\n\nReviewer: {rv.get('reviewer')}\n\nNotes: {rv.get('notes') or '-'}\n\n"
            "No report was generated."
        )
    return {"final_report": body, "verdict": verdict, "current_step": "failure_handler"}


# =============================================================================
# 7. Graph assembly
# =============================================================================
def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("geometry_extractor", geometry_extractor_node)
    g.add_node("compliance_checker", compliance_checker_node)
    g.add_node("stress_reviewer", stress_reviewer_node)
    g.add_node("human_review", human_review_node)
    g.add_node("reporter", reporter_node)
    g.add_node("failure_handler", failure_handler_node)

    g.add_edge(START, "orchestrator")
    g.add_conditional_edges("orchestrator", route_from_orchestrator, {n: n for n in PIPELINE} | {"failure_handler": "failure_handler", END: END})
    for spoke in PIPELINE:
        g.add_edge(spoke, "orchestrator")
    g.add_edge("failure_handler", END)
    # Use SqliteSaver / PostgresSaver in production so a paused review survives restarts.
    return g.compile(checkpointer=checkpointer or MemorySaver())


# =============================================================================
# 8. Execution loop (with terminal HITL)
# =============================================================================
def prompt_engineer(payload: dict) -> dict:
    print("\n" + "=" * 72 + "\nHUMAN REVIEW REQUIRED\n" + "=" * 72)
    print("Properties:", json.dumps(payload["extracted_properties"], indent=2))
    print("\nCompliance:")
    for f in payload["compliance_status"]:
        print(f"  [{f['status']:<14}] {f.get('rule_ref') or '-':<4} {f['rule_checked']}")
        for n in f["validation_notes"]:
            print(f"        ! {n}")
    print("\nStress hotspots:")
    for h in payload["stress_hotspots"]:
        print(f"  {h['id']:<4} {h['estimated_severity']:<9} {h['heuristic']:<22} {h['location']}\n        {h['geometric_reason']}")
    print(f"\nFEA: {payload['fea_job'].get('status')} - {payload['fea_job'].get('message')}\n")
    reviewer = input("Reviewer name: ").strip() or "unknown"
    dismiss = [x.strip() for x in input("Hotspot ids to dismiss (comma-separated, blank for none): ").split(",") if x.strip()]
    notes = input("Notes: ").strip()
    approved = input("Approve and generate report? [y/N]: ").strip().lower() == "y"
    return {"approved": approved, "reviewer": reviewer, "notes": notes, "dismiss_hotspots": dismiss}


def run_review(state: AgentState, thread_id: str | None = None, reviewer_callback=prompt_engineer, graph=None) -> dict:
    """Run to completion, pausing at each HITL interrupt. `reviewer_callback(payload) -> decision dict`
    can be swapped for a web/API handler."""
    graph = graph or build_graph()
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}, "recursion_limit": 40}
    graph.invoke(state, config)
    while True:
        snapshot = graph.get_state(config)
        if not snapshot.next:
            return snapshot.values
        pending = [i for t in snapshot.tasks for i in t.interrupts]
        if not pending:
            raise RuntimeError(f"Graph halted at {snapshot.next} without an interrupt payload")
        graph.invoke(Command(resume=reviewer_callback(pending[0].value)), config)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Mechanical CAD Companion AI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest", help="Index a rulebook into the vector DB")
    ing.add_argument("path")
    run = sub.add_parser("run", help="Review a CAD file")
    run.add_argument("cad_file")
    run.add_argument("--units", default="mm", choices=list(UNIT_TO_MM))
    run.add_argument("--material", choices=sorted(MATERIAL_DENSITY_KG_M3))
    run.add_argument("--density", type=float, help="kg/m^3; overrides --material")
    run.add_argument("--part", default="", help="Free-text part description used for rulebook retrieval")
    run.add_argument("--thread-id")
    run.add_argument("--auto-approve", action="store_true", help="Skip the human gate (CI only)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "ingest":
        print(f"Indexed {ingest_rulebook(args.path)} chunks.")
        return 0

    callback = (lambda _p: {"approved": True, "reviewer": "auto", "notes": "auto-approved"}) if args.auto_approve else prompt_engineer
    final = run_review(
        make_initial_state(args.cad_file, args.units, args.material, args.density, args.part),
        thread_id=args.thread_id, reviewer_callback=callback,
    )
    print("\n" + final["final_report"])
    print(f"\nVERDICT: {final['verdict']}" + (f" | report: {final['report_path']}" if final.get("report_path") else ""))
    return 0 if final["verdict"] in ("APPROVED", "APPROVED_WITH_WARNINGS") else 1


if __name__ == "__main__":
    sys.exit(main())
