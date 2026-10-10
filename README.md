# Mechanical CAD Companion AI — LangGraph backend

Hub-and-spoke `StateGraph` orchestrating four specialist nodes over a CAD file:

```
START -> orchestrator --+--> geometry_extractor (trimesh)        --+
                        +--> compliance_checker (RAG + rulebook) --+--> back to orchestrator
                        +--> stress_reviewer (3 heuristics)      --+
                        +--> human_review  (HITL interrupt, before the reporter)
                        +--> reporter -> END
                        +--> failure_handler -> END   (tool error or engineer rejection)
```

The orchestrator is deterministic Python — no LLM routing. Every physical number comes
from a Python tool and is re-validated by code; invented numbers are overwritten or
rejected. An unavailable LLM or rulebook degrades to `NOT_VERIFIABLE` / `INCOMPLETE`
rather than crashing or approving optimistically.

**Source:** https://github.com/jasanjoseph/CAD.ai

## Quick start

```powershell
git clone https://github.com/jasanjoseph/CAD.ai.git
cd CAD.ai
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt shapely mapbox_earcut pytest

Copy-Item .env.example .env
notepad .env          # paste your GOOGLE_API_KEY (free: aistudio.google.com/apikey)

.\.venv\Scripts\python.exe make_sample_part.py                      # demo part
.\.venv\Scripts\python.exe cad_companion.py ingest .\rulebook\baja_sae_2026.txt
.\.venv\Scripts\python.exe cad_companion.py run .\parts\bracket.stl --units mm --material aisi_4130 --part "suspension mounting bracket"
```

Python 3.10+ (tested on 3.14). On Windows use `py` — the `python` alias is the
Microsoft Store stub. The interactive run pauses at the human gate; add
`--auto-approve` for non-interactive/CI use.

Your own part: `... run .\parts\your_part.stl --units mm --material al_6061_t6 --part "wheel mount"`
— materials `aisi_4130`, `aisi_1018`, `al_6061_t6`, `al_7075_t6`, `ti_6al_4v`, or skip
`--material` and pass `--density <kg/m^3>`. The mesh must be watertight or the pipeline
fails loudly.

## Test (no API key, no network, no quota)

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

55 tests: geometry (mass/volume checked against hand calculations), all three stress
heuristics, the anti-hallucination validator, verdict rules, reporter number-guarding,
graph routing, and two full end-to-end runs with the LLM stubbed — including a
simulated LLM outage at every stage. Run this first; it proves the pipeline before you
spend any quota.

## Verdicts and exit codes

| Verdict | Meaning | Exit code |
|---|---|---|
| `APPROVED` | All checked rules pass, no serious hotspots | 0 |
| `APPROVED_WITH_WARNINGS` | Some rules not verifiable from geometry, or serious hotspots | 0 |
| `NON_COMPLIANT` | At least one rule validated as FAIL | 1 |
| `INCOMPLETE` | **No rule was evaluated** (rulebook/model unavailable) — not a review | 1 |
| `FAILED` | A tool or file error stopped the pipeline | 1 |
| `REJECTED` | The engineer rejected it at the human gate | 1 |

## Free-tier quota (important)

A review makes **3 LLM calls** plus 5 embedding calls; all geometry and stress analysis
is pure Python. The default `gemini-3.5-flash` is chosen because `gemini-3.8-flash`
allows only ~20 free requests/day (~2 reviews). Quota is tracked per model, so if you
hit a limit, setting `GEMINI_MODEL` to another free model in `.env` moves you to a
separate budget — no re-ingest needed. Hitting a limit is not fatal: measurements,
heuristics and the report still complete, compliance returns `NOT_VERIFIABLE`, and the
verdict becomes `INCOMPLETE` rather than a false approval.

## Layout

| Path | Purpose |
|---|---|
| `cad_companion.py` | The whole backend: tools, agents, graph, CLI |
| `ARCHITECTURE.md` | Development schema: state shape, node/tool contracts, trust model |
| `tests/test_pipeline.py` | Offline test suite (no API key needed) |
| `make_sample_part.py` | Generates `parts/bracket.stl` (sharp corner, section change, 2.5 mm thin wall) |
| `rulebook/baja_sae_2026.txt` | Synthetic rulebook extract for demo runs |
| `.env.example` | Copy to `.env` and fill in your key |

## Expected demo output

On the sample bracket (steel, 95400 mm^3, 0.749 kg) the deterministic tools find a HIGH
sharp 90° internal corner, two CRITICAL section changes (48× and 6× area jumps) and 8
thin-wall hotspots (2.48 mm vs 12 mm median). Rule T-2.7 (mass ≤ 0.50 kg) fails at
0.749 kg, so the verdict is `NON_COMPLIANT`; T-4.1/T-4.2 come back `NOT_VERIFIABLE`
because a mesh cannot prove fillet radii or certify welds.
