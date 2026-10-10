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

The orchestrator is deterministic Python — no LLM routing. LLMs are used only for
tool-call argument passing, rulebook interpretation and prose. Every physical number
comes from a Python tool and is re-validated by code; invented numbers are overwritten
or rejected.

## Layout

| Path | Purpose |
|---|---|
| `cad_companion.py` | The whole backend: tools, agents, graph, CLI |
| `make_sample_part.py` | Generates `parts/bracket.stl`, a demo L-bracket (sharp corner, section change, 2.5 mm thin wall) |
| `rulebook/baja_sae_2026.txt` | Synthetic rulebook extract for demo runs |
| `.env.example` | Copy to `.env` and fill in your keys |
| `reports/` | Markdown review reports (created on first run) |

## Setup (already done — for reference / other machines)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install shapely mapbox_earcut   # for the sample-part generator
```

Python 3.14 on Windows: use `py` (the `python` alias is the Microsoft Store stub).

## Configuration

```powershell
Copy-Item .env.example .env
notepad .env    # paste your GOOGLE_API_KEY; keep LLM_PROVIDER=gemini (free tier)
```

Provider support: `gemini` (free, rate-limited), `openai` (billed), `anthropic` (billed).
The rulebook embeddings follow the provider — `LLM_PROVIDER=gemini` keeps everything on the
free tier. If you change `EMBEDDING_PROVIDER`, re-run `ingest`: a Chroma index is only valid
for the embedding model that produced it.

## Run

From `C:\Users\jasan\OneDrive\Desktop\AI\cad-companion`:

```powershell
# 0) one-time: make the demo part
.\.venv\Scripts\python.exe make_sample_part.py

# 1) index the rulebook (needs OPENAI_API_KEY — embeddings)
.\.venv\Scripts\python.exe cad_companion.py ingest .\rulebook\baja_sae_2026.txt

# 2) run a review; pauses at the human gate
.\.venv\Scripts\python.exe cad_companion.py run .\parts\bracket.stl --units mm --material aisi_4130 --part "suspension mounting bracket"

# or non-interactive (CI)
.\.venv\Scripts\python.exe cad_companion.py run .\parts\bracket.stl --units mm --material aisi_4130 --part "suspension mounting bracket" --auto-approve
```

Your own part:

```powershell
.\.venv\Scripts\python.exe cad_companion.py run .\parts\your_part.stl --units mm --material al_6061_t6 --part "wheel mount"
```

Materials: `aisi_4130`, `aisi_1018`, `al_6061_t6`, `al_7075_t6`, `ti_6al_4v`, or skip `--material`
and pass `--density <kg/m^3>`. The mesh must be watertight or the pipeline fails loudly.

## Expected demo output

On the sample bracket (steel, 95400 mm^3, 0.749 kg) the deterministic tools find:
a HIGH sharp 90° internal corner, two CRITICAL section changes (48× and 6× area jumps)
and 8 thin-wall hotspots (2.48 mm vs 12 mm median). Rule T-2.7 (mass ≤ 0.50 kg) fails at
0.749 kg, so the verdict is `NON_COMPLIANT`; rules T-4.1/T-4.2 come back `NOT_VERIFIABLE`
because a mesh cannot prove fillet radii or certify welds.
