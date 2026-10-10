# Architecture & Development Schema

Mechanical CAD Companion AI — LangGraph backend. This document is the canonical
development schema: state shape, graph topology, node/tool contracts, and the
trust model that keeps every number in the output auditable.

---

## 1. System topology

```
                            ┌──────────────────────────────────────────┐
                            │              START                       │
                            └───────────────────┬──────────────────────┘
                                                ▼
        ┌────────────────────────── orchestrator (deterministic Python router) ──────────────────────────┐
        │                                                                                               │
        │  next unfinished spoke, in fixed order:                                                       │
        │                                                                                               │
        ├─► geometry_extractor ──► back to orchestrator                                                 │
        │      (LLM tool-call wrapper over trimesh; mass/volume/bbox/CoG)                                │
        │                                                                                               │
        ├─► compliance_checker ──► back to orchestrator                                                 │
        │      (RAG over rulebook + LLM findings; code re-validates every number)                        │
        │                                                                                               │
        ├─► stress_reviewer ──► back to orchestrator                                                    │
        │      (LLM drives 3 geometric heuristics; coverage forced in code)                              │
        │                                                                                               │
        ├─► human_review ──► back to orchestrator      [HITL: interrupt(), pauses the graph]            │
        │      engineer approves / rejects, may dismiss hotspots                                         │
        │                                                                                               │
        ├─► reporter ──► END        (markdown report + verdict)                                          │
        │                                                                                               │
        └─► failure_handler ──► END  (any tool error, or engineer rejection)                             │
                                                                                                          │
   Routing keys: `error` (non-empty → failure_handler) · `human_review.approved == False` → failure_handler
                 `completed[]` (each spoke appends itself; drives the fixed-order loop)
```

Design rule: **the orchestrator never calls an LLM.** Routing is pure Python over
`completed`, `error` and `human_review`. LLMs appear only inside three nodes and only
for (a) tool-call argument passing, (b) rulebook interpretation, (c) prose.

---

## 2. State schema (`AgentState`, TypedDict)

| Key | Type | Written by | Meaning |
|---|---|---|---|
| `cad_file_path` | `str` | user input | `.stl` / `.step` / `.stp` path |
| `extracted_properties` | `dict` | geometry_extractor | volume, mass, bbox, CoG + provenance (`volume_source`, `warnings`) |
| `compliance_status` | `list[dict]` | compliance_checker | one validated finding per rule (see §5) |
| `stress_hotspots` | `list[dict]` | stress_reviewer / human_review | merged heuristic hits, ids `H1..Hn`, `reviewer_status` |
| `final_report` | `str` | reporter / failure_handler | full markdown, or failure/rejection text |
| `current_step` | `str` | every node | last node that ran (debugging aid) |
| `units` | `str` | user input | `mm`/`cm`/`m`/`in` (STL only; STEP is mm) |
| `material` \| `density_kg_m3` | `str` \| `float` | user input | handbook key or explicit density; drives `mass_kg` |
| `part_description` | `str` | user input | free text, used only to steer rulebook retrieval |
| `completed` | `list[str]` | `safe_node` wrapper | finished spokes; the router's loop driver |
| `error` | `str` \| `None` | `safe_node` wrapper | non-empty ⇒ straight to failure_handler |
| `stress_notes` | `str` | stress_reviewer | advisory LLM prose (no numbers allowed) |
| `fea_job` | `dict` | stress_reviewer | external-solver stub + heuristic summaries |
| `human_review` | `dict` | human_review | `{approved, reviewer, notes, dismissed_hotspots, timestamp_utc}` |
| `verdict` | `str` | reporter | `APPROVED` / `APPROVED_WITH_WARNINGS` / `NON_COMPLIANT` / `FAILED` / `REJECTED` |
| `report_path` | `str` \| `None` | reporter | written markdown file |

Verdict is **computed by code** (`compute_verdict`), never by the LLM:
any validated `FAIL` ⇒ `NON_COMPLIANT`; else any active `HIGH`/`CRITICAL`
hotspot or any `NOT_VERIFIABLE` rule ⇒ `APPROVED_WITH_WARNINGS`; else `APPROVED`.

---

## 3. Node contracts

| Node | LLM role | Deterministic core | Failure behaviour |
|---|---|---|---|
| `geometry_extractor` | forced tool-call `extract_geometry` (falls back to unfocused call if provider can't force) | trimesh: watertight check, normal fix, mm scaling, b-rep vs mesh volume provenance | no tool call ⇒ `CADToolError` ⇒ failure_handler |
| `compliance_checker` | interprets rulebook excerpts → structured `ComplianceReport` | retrieval (5 queries, deduped, ≤14 chunks), then `validate_finding()` re-derives every verdict | empty findings ⇒ explicit `NOT_VERIFIABLE` row |
| `stress_reviewer` | drives heuristics via tool-calls, then ≤120-word advisory note | **coverage guarantee**: any heuristic the LLM skipped is run with defaults; hotspots merged, sorted, re-id'd | tool error ⇒ failure_handler |
| `human_review` | — (human is the model) | `interrupt()`; re-runs safely on resume (no side effects before the interrupt) | malformed decision ⇒ treated as rejection |
| `reporter` | narrative: ≤120-word summary + 3-7 recommendations | tables, verdict, file write; `ungrounded_numbers()` rejects any number absent from state (1 retry, then coded fallback) | fallback narrative, never blocks output |
| `failure_handler` | — | renders FAILED or REJECTED text | terminal |

`@safe_node` wraps every spoke: any exception becomes
`error = "[node] ExceptionType: message"` in state — never a silent fallback.

---

## 4. Tool contracts (all deterministic Python — the only source of physical numbers)

| Tool | Input (locked by pipeline) | Output | Notes |
|---|---|---|---|
| `extract_geometry` | file, units, material/density | `GeometryProperties` (volume, area, bbox min/max, CoG, mass, tri count, warnings) | `mass_kg` is null without material/density; STEP gives exact b-rep volume/CoG |
| `check_sharp_internal_radii` | file, units, `sharp_angle_deg=60`, `max_hotspots=8` | hotspots: concave dihedral turns, voxel-clustered | severity from turn angle; tessellation noise filtered by edge length |
| `check_section_changes` | file, units, `num_slices=80`, `area_ratio_threshold=1.6` | hotspots: adjacent-slice area jumps along X/Y/Z | log-scaled severity; part tips ignored via 2% floor |
| `check_thin_walls` | file, units, `num_samples=3000`, `min_wall_mm=1.5`, `relative_wall_fraction=0.25` | hotspots: ray-cast local thickness vs part median | fails loudly if <50% of rays land (mesh/rtree problem) |
| `trigger_external_fea` | file, material, hotspot ids | `NOT_CONFIGURED` stub | swap for a real CalculiX/gmsh service; loads must come from an engineer |

Shared: `load_mesh()` caches per `(path, mtime, units)`, enforces watertightness,
fixes winding/normals, and raises `CADToolError` on anything unmeasurable.

---

## 5. Compliance finding schema (LLM output → code-validated row)

```
rule_ref · rule_checked · limit_text · property_key · measured_value
comparator(<=|>=|<|>|==) · limit_value · status(PASS|FAIL|NOT_VERIFIABLE) · reasoning
```

`validate_finding()` enforces, in order:
1. `rule_ref` must exist among retrieved excerpts → else `NOT_VERIFIABLE`
2. PASS/FAIL requires a `property_key` that is a measured property + comparator + limit → else `NOT_VERIFIABLE`
3. `measured_value` is **replaced** by the tool value; LLM arithmetic is recomputed;
   a wrong verdict is overwritten and logged in `validation_notes`

---

## 6. Trust model (why you can audit the report)

- Every physical number originates in a Python tool; LLMs may only quote or copy.
- Reporter prose is scanned for ungrounded numbers (must appear verbatim in state JSON).
- Hotspot dismissal is honoured and reported as dismissed, never silently dropped.
- Heuristics are labelled screening, never FEA; the FEA stub states plainly that no
  stress values exist.
- The external gate belongs to a human engineer; `--auto-approve` exists for CI only.

---

## 7. Execution & persistence

- `run_review()` loops over HITL interrupts with `Command(resume=decision)`; swap
  `reviewer_callback` for a web/API handler.
- Checkpointer: `MemorySaver` (in-process). Production: `SqliteSaver`/`PostgresSaver`
  so a paused review survives restarts.
- Artifacts: `reports/*.md`, Chroma index at `.rulebook_chroma/` (valid only for the
  embedding model that built it — re-ingest after switching providers).

## 8. Providers

| `LLM_PROVIDER` | Chat | Embeddings (default) | Cost |
|---|---|---|---|
| `gemini` | `gemini-3.8-flash` | `models/gemini-embedding-001` | free tier, rate-limited |
| `openai` | `gpt-4o` | `text-embedding-3-small` | billed per token |
| `anthropic` | `claude-sonnet-5-5` | `text-embedding-3-small` (OpenAI key required) | billed per token |
