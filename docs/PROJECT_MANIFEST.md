# Project Manifest

Root folder:

```text
./
```

## Runtime Entry Points

| Component | Path | Purpose |
|---|---|---|
| Backend API | `backend/app.py` | FastAPI service, `/simulate`, `/pir/fetch`, `/mpc/compare`, `/model/info` |
| Simulation runtime | `backend/simulation_runtime.py` | The Well emulator routing, modes, trained checkpoint loading, fallback renderers |
| PIR implementation | `backend/crypto/pir.py` | CGKS and DPF PIR primitive |
| PIR service | `backend/crypto/pir_service.py` | Scenario library wrapper and API response formatting |
| GC bridge | `backend/crypto/gc_bridge.py` | Private risk comparison demonstrator |
| Frontend UI | `frontend/src/app/page.tsx` | Minimal assembly-style UI, simulation canvas, tables, mode controls |
| Global CSS | `frontend/src/app/globals.css` | Inverted black/white terminal style |

## Models And Training Artifacts

| File | Status |
|---|---|
| `backend/models/well_mhd64_emulator.pt` | installed trained The Well `MHD_64` checkpoint |
| `backend/models/well_post_neutron_star_merger_emulator.pt` | installed trained The Well `post_neutron_star_merger` checkpoint |
| `kaggle/outputs/well_mhd64_emulator.pt` | original MHD training output copy |
| `kaggle/outputs/training_report.json` | MHD training report |
| `kaggle/outputs_post_merger/well_post_neutron_star_merger_emulator.pt` | downloaded post-merger Kaggle artifact |
| `kaggle/outputs_post_merger/training_report.json` | post-merger training report |
| `kaggle/kernel/well_mhd64_kernel.py` | reusable Kaggle GPU trainer template |
| `kaggle/generated/prometheus-well-post-merger-emulator/` | fixed post-merger Kaggle kernel |
| `ml/train_well_emulator.py` | local/Colab trainer with fixed 2D scalar slice extraction |

## Documents

| Document | Purpose |
|---|---|
| `README.md` | User guide, local runner |
| `docs/RESEARCH_PAPER.md` | Technical and research specifications |
| `docs/TECHNICAL_REPORT.md` | System engineering design |
| `docs/PITCH_DECK.md` | Core architecture deck outline |
| `docs/DEMO_SCRIPT.md` | Step-by-step walk-through script |
| `docs/RUNBOOK.md` | Detailed setup and troubleshooting |

## Verification Commands

```powershell
powershell -ExecutionPolicy Bypass -File .\FINAL_BUILD.ps1
powershell -ExecutionPolicy Bypass -File .\FINAL_VERIFY.ps1
```

Expected result:

```text
phase_10_full_verify PASS
```

## Deliberate Exclusions

The following are not stored in this folder:

- Raw private API keys.
- Browser login sessions.
- Kaggle access tokens.
- Full The Well raw datasets.
- Python virtual environments.
- Frontend `node_modules` dependency cache in the archive.

The code includes reproducible scripts and `.env.example` placeholders instead.

