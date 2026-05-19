# Physics Engine — Confidential, Provenance-Honest Physics Simulation & Explanation

> An AI-native workbench that turns a natural-language physics prompt into simulated field frames from real scientific data (Polymathic AI's *The Well*), an animated concept explainer, and a plain-language description — served with **explicit provenance** (you always see whether output came from a trained model, a numerical solver, or a deterministic renderer) and a **confidentiality layer** (two-server Private Information Retrieval) so a query need not reveal a lab's research direction to any single server.

[![status](https://img.shields.io/badge/status-research%20prototype-blue)]()
[![python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![license](https://img.shields.io/badge/license-see%20ATTRIBUTION-lightgrey)]()
[![data](https://img.shields.io/badge/data-The%20Well%20(CC--BY--4.0)-green)]()

---

## Table of Contents
1. [What this is](#1-what-this-is)
2. [What it is not (claim boundaries)](#2-what-it-is-not-claim-boundaries)
3. [System architecture](#3-system-architecture)
4. [Components](#4-components)
5. [Quick start](#5-quick-start)
6. [Configuration](#6-configuration)
7. [API reference](#7-api-reference)
8. [Trained models — real numbers](#8-trained-models--real-numbers)
9. [Confidentiality layer](#9-confidentiality-layer)
10. [Provenance & honesty model](#10-provenance--honesty-model)
11. [Testing & verification](#11-testing--verification)
12. [Deployment](#12-deployment)
13. [Roadmap](#13-roadmap)
14. [Attribution, data licensing & security](#14-attribution-data-licensing--security)

---

## 1. What this is

A research prototype that composes four layers behind one prompt box:

1. **Routing.** Natural-language prompt → a *The Well* dataset family (e.g. `MHD_64`, `post_neutron_star_merger`, `supernova_explosion_64`, `acoustic_scattering_maze`, `rayleigh_benard`, `gray_scott_reaction_diffusion`, `shear_flow`, `planetswe`).
2. **Simulation.** If a trained checkpoint exists for the route, a residual-CNN next-frame emulator is rolled forward autoregressively; otherwise a numerical PDE solver or deterministic renderer is used — **and the source is labeled**.
3. **Explanation.** An LLM layer rewrites *already-computed metadata* into a plain-language description, and (optional) generates a Manim concept animation from the prompt.
4. **Confidentiality.** A two-server PIR service lets a client retrieve a scenario record so that **neither single non-colluding server learns which record** was requested.

The central design rule is **honest provenance**: the system never presents a deterministic renderer as a trained model, and never claims trained output for a route whose checkpoint is missing.

## 2. What it is not (claim boundaries)

To keep the project credible with reviewers and domain experts, these boundaries are stated explicitly and enforced in the UI:

- **Not a validated solver.** The trained emulators predict a one-step evolution of a single auto-selected scalar slice. They are **not** GR black-hole physics, **not** reactor-grade fusion simulation, and **not** full multi-field MHD solvers.
- **Not observational imagery.** Rendered frames are normalized scalar fields, not NASA/telescope data.
- **Not a benchmarked accuracy claim.** Reported validation MSE (Section 8) is a training-time metric on a small data slice **without** an identity/persistence baseline; treat it as provisional, not state-of-the-art.
- **Not metadata-private or collusion-resistant.** The PIR layer protects the *queried index* against a single non-colluding server only; it makes no claim against server collusion, traffic analysis, or side channels.
- The "garbled-circuit comparison" is currently a **plaintext placeholder** pending a real implementation (see Roadmap). It is labeled as such in code and UI; it provides **no input privacy** today.

## 3. System architecture

```
                          ┌─────────────────────────────────────────────┐
   prompt ───────────────►│  FastAPI backend (backend/app.py)            │
                          │                                             │
                          │  route_task_to_dataset()  ── Well catalog   │
                          │         │                                   │
                          │         ▼                                   │
                          │  simulate_task() ──► trained checkpoint     │
                          │         │            OR PDE solver          │
                          │         │            OR deterministic render│
                          │         ▼                                   │
                          │  maybe_explain()  (provider-agnostic LLM)   │
                          │  explainer/*      (Manim animation, async)  │
                          │                                             │
                          │  crypto/pir_service  (CGKS + BGI (2,2)-DPF) │
                          └───────────────┬─────────────────────────────┘
                                          │ JSON / mp4
                          ┌───────────────▼─────────────────────────────┐
                          │  Next.js frontend (frontend/)               │
                          │  canvas + provenance badge + stats + explain│
                          └─────────────────────────────────────────────┘
```

## 4. Components

| Layer | Path | Status |
|------|------|--------|
| API | `backend/app.py` | Working — 9 endpoints |
| Routing | `backend/well_catalog.py` | Working — keyword routing over 8 families |
| Simulation runtime | `backend/simulation_runtime.py` | Working — checkpoint / PDE / renderer dispatch |
| Surrogate model | `backend/surrogate/well_emulator.py` | Working — `ResidualFrameEmulator` (residual CNN) |
| LLM explanation | `backend/llm_explain.py` | Working — OpenAI/Anthropic/Gemini (provider-agnostic) |
| Animated explainer | `backend/explainer/` | Planned — see build spec |
| PIR | `backend/crypto/pir.py`, `pir_service.py` | Working — CGKS + BGI (2,2)-DPF |
| Private comparison | `backend/crypto/gc_bridge.py` | Placeholder — plaintext, labeled |
| Frontend | `frontend/` | Working — Next.js console UI |

## 5. Quick start

**Requirements:** Python 3.11+, Node 18+. (Manim animation also needs LaTeX + ffmpeg — see Deployment.)

```bash
# 1. Backend
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn backend.app:app --reload --port 8000

# 2. Frontend (separate shell)
cd frontend && npm install && npm run dev            # http://localhost:3000
```

Open `http://localhost:3000`; the API is at `http://127.0.0.1:8000` (docs at `/docs`).

> **Note:** `torch` is required by the surrogate runtime and **must** be installed (it is not auto-pulled by some minimal installs). The CPU wheel is sufficient for inference.

## 6. Configuration

Copy `.env.example` → `.env`. Key variables:

| Variable | Purpose | Default |
|---|---|---|
| `WELL_EMULATOR_CHECKPOINT` | path to the loaded `.pt` | `backend/models/well_mhd64_emulator.pt` |
| `PROMETHEUS_LLM_EXPLAIN` | `1` to enable LLM explanation | `0` |
| `PROMETHEUS_LLM_PROVIDER` | `openai` \| `anthropic` \| `gemini` | — |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | OpenAI creds | — / `gpt-4.1-mini` |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | Gemini creds | — / `gemini-2.5-flash` |
| `ALLOWED_ORIGINS` | extra CORS origins (comma-sep) | — |

Secrets are never committed; `.env` is git-ignored.

## 7. API reference

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/healthz` | — | service + model load status |
| GET | `/scenarios` | — | fixed scenario library |
| POST | `/simulate` | `{task, steps, mode}` | `frames, meta, route, interpretation, llm_explanation, warning` |
| POST | `/predict` | `{mach_sonic, mach_alfvenic, steps}` | frames + risk |
| POST | `/pir/fetch` | `{scenario_index, method}` | PIR-retrieved record + server views |
| POST | `/mpc/compare` | `{lab_a_value, lab_b_value}` | comparison (plaintext placeholder, labeled) |
| GET | `/well/catalog` | — | dataset catalog + loaded checkpoint |
| GET | `/model/info` | — | checkpoint meta + training report + LLM status |
| POST | `/agent` | `{request_text, channel}` | agent action |

## 8. Trained models — real numbers

Two residual-CNN next-frame emulators (`ResidualFrameEmulator`, `model_width=64`, output bilinearly resampled to 128×128) trained on *The Well*:

| Dataset | Train / Val items | Epochs | Batch | Best Val MSE |
|---|---|---|---|---|
| `MHD_64` | 7,623 / 990 | 6 | 1 | **0.2199** |
| `post_neutron_star_merger` | 1,080 / 180 | 3 | 1 | **0.2917** |

**Honest reading of these numbers:** the MHD validation curve across 6 epochs is `0.2207, 0.2227, 0.3192, 0.2200, 0.2203, 0.2199` — it **oscillates and barely improves** over the epoch-1 value, and the report includes **no identity/persistence baseline**. This means the current checkpoints are best understood as *demo-grade* (the model likely learns a smoothed local-structure predictor), not as a validated, benchmarked surrogate. Improving capacity, adding early stopping, and reporting an identity baseline are tracked in the Roadmap.

## 9. Confidentiality layer

`backend/crypto/pir.py` implements **two real, single-server-private** PIR schemes in pure stdlib (`hashlib`, `secrets`), no third-party crypto:

- **CGKS** information-theoretic 2-server PIR (O(n)-bit query) — the always-correct fallback.
- **BGI (2,2)-Distributed Point Function** PIR (O(λ·log n)-bit query) — the primitive underlying the PIRSONA lineage (per-party leaf flags satisfy `flags0 XOR flags1 = e_alpha`).

The client reconstructs record *j* from the XOR of the two server answers; neither single server learns *j*. **Production note:** swap the SHAKE256 PRG for fixed-key AES (as PIRSONA does); the protocol logic is identical.

## 10. Provenance & honesty model

Every `/simulate` response carries `meta.data_source_kind ∈ {trained_the_well_checkpoint, missing_checkpoint, deterministic_physics_renderer}` plus `trained_for_request`. The UI renders this as a visible badge so a trained-model frame is never confused with a renderer frame. Of the 8 presets, **2 use trained checkpoints** (MHD, black hole) and **6 use numerical solvers or deterministic renderers** — and the UI says so.

## 11. Testing & verification

```bash
python scripts/mini_pytest.py          # runs tests/test_*.py
python scripts/verify.py               # appends PASS/FAIL with real values to checkpoints.log
```
Test suite covers PIR correctness, surrogate loading, the agent, the API surface, and the comparison contract.

## 12. Deployment

- **Backend:** any long-running Python host (Render / Railway / Fly / Cloud Run with a Dockerfile). A FastAPI + torch backend will **not** run on size-limited serverless.
- **Animated explainer:** requires a container with **LaTeX + ffmpeg + Cairo/Pango** for Manim; renders run **asynchronously** (job + poll), not inside a request.
- **Frontend:** Vercel; set `NEXT_PUBLIC_BACKEND_URL` to the backend host and add that origin to `ALLOWED_ORIGINS`.

## 13. Roadmap

- Replace the plaintext comparison with a **real Yao garbled circuit** (wire labels, point-and-permute, OT).
- Retrain the surrogate (wider model, early stopping, **identity baseline**, leakage audit).
- Ship the **LLM→Manim animated explainer** (async, sandboxed, with pre-rendered fallback).
- Replace SHAKE256 PRG with fixed-key AES in PIR.
- Add bring-your-own-data ingestion (`.npy` → HDF5/VTK).

## 14. Attribution, data licensing & security

- **Data:** *The Well* (Polymathic AI) — data CC-BY-4.0, code BSD-3-Clause; commercial use permitted **with attribution**. Verify per-dataset metadata before any commercial use.
- **PIR:** clean-room pure-Python; no GPL code vendored. Follows the BGI (2,2)-DPF lineage (Vadapalli, Bayatbabolghani & Henry, *PoPETs* 2021.4).
- **Security:** secrets are environment-only; LLM error strings are key-redacted; any code-generation/animation path executes generated code **only behind a static AST allowlist sandbox**.
- See `ATTRIBUTION.md` for the full breakdown of pre-existing vs. new assets.
