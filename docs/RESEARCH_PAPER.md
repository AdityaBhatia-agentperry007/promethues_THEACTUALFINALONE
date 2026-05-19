# PROMETHEUS: Confidential Physics-as-a-Service with Private Retrieval and Honest Simulation Provenance

## Abstract

Shared scientific simulation services are useful only if researchers can trust both the outputs and the privacy boundary around their requests. PROMETHEUS demonstrates a confidential physics-as-a-service architecture where a natural-language request is routed to a physics dataset family, simulated frame by frame, and served with explicit provenance. The system combines trained The Well `MHD_64` and `post_neutron_star_merger` frame emulators, route-specific deterministic or numerical fallback simulators, two-server private information retrieval, and a small garbled-circuit comparison bridge. The central design rule is honesty: if a route-specific The Well checkpoint is missing, the UI reports that state and does not claim trained output. The demo therefore separates three concerns that are often blurred in typical proxy systems: privacy of the requested scenario index, fidelity/provenance of the simulation source, and presentation of useful scientific metadata.

## 1. Motivation

Fusion, plasma physics, astrophysics, and safety research increasingly rely on large simulation corpora and learned surrogate models. A shared service can lower access barriers, but it creates a privacy problem: the request itself can reveal a lab's research direction. For example, repeatedly requesting a narrow class of plasma instabilities, compact-object environments, or risk-heavy parameter regimes can leak strategy even when the output data is not secret.

PROMETHEUS addresses this demo-scale problem with two mechanisms. First, route selection and simulation provenance are exposed to the user so claims remain bounded by the model actually loaded. Second, a PIR layer demonstrates how a user can retrieve a scenario record without revealing the selected index to either single non-colluding server.

## 2. System Overview

The system has four layers:

1. Request routing. The backend maps prompt text to a The Well dataset family such as `MHD_64`, `post_neutron_star_merger`, `supernova_explosion_64`, `acoustic_scattering_maze`, `rayleigh_benard`, `gray_scott_reaction_diffusion`, `shear_flow`, or `planetswe`.
2. Simulation runtime. If a matching trained checkpoint exists, the runtime loads it and rolls it forward autoregressively. If no checkpoint exists, the runtime returns a deterministic renderer or numerical PDE solver with a warning.
3. Confidential compute layer. The PIR service fetches fixed scenario records through CGKS or DPF two-server PIR. The comparison bridge demonstrates a minimal garbled-circuit-style private scalar comparison.
4. User interface. The frontend is intentionally minimal: an assembly-like black/white interface with a large simulation canvas, mode toggles, frame statistics, source rows, interpretation tables, and model registry.

## 3. Simulation Provenance

The installed trained models are `backend/models/well_mhd64_emulator.pt`, trained on The Well `MHD_64`, and `backend/models/well_post_neutron_star_merger_emulator.pt`, trained on The Well `post_neutron_star_merger`. They are residual convolutional next-frame emulators. A deterministic task hash chooses a validation seed frame, and the network predicts subsequent frames autoregressively.

For routes without installed checkpoints, PROMETHEUS avoids making false claims. A black-hole prompt routes to the installed `post_neutron_star_merger` checkpoint. Supernova, acoustic maze, convection, reaction diffusion, shear flow, and planetary shallow-water presets still use route-specific deterministic or numerical fallbacks until their own checkpoints are installed.

This approach is intentionally conservative. The demo remains visually useful, but the source table distinguishes trained checkpoint output from live numerical PDE output and deterministic fallback rendering.

## 4. Mode System

Each route supports multiple views. These views are not only color palettes; the backend returns mode-specific scalar fields.

- `MHD_64`: field, gradient, shock edges, magnetic proxy.
- Black hole / post-merger: intensity, heat radiation, lensing, Doppler.
- Supernova: density, shock front, temperature, ejecta.
- Acoustic maze: pressure, wave energy, maze geometry, interference.
- Rayleigh-Benard: temperature, heat flux, plume velocity, rolls.
- Gray-Scott: concentration, reaction rate, pattern edges.
- Shear flow: scalar, vorticity, mixing, velocity.
- Planetary shallow water: height, vorticity, jet stream, storm track.

The frontend sends `task`, `steps`, and `mode` to `/simulate`. The response includes `supported_modes`, `mode`, `visual_style`, `data_source_kind`, `frame_stats`, and `interpretation`.

## 5. Private Information Retrieval

PROMETHEUS includes two single-server-private PIR constructions at demo scale.

CGKS uses an O(n)-bit random binary query vector to one server and a second vector differing at the desired index to the other server. Each server sees a uniformly random-looking vector. XORing both server answers reconstructs the desired record.

The DPF route implements a two-party distributed point function in the Boyle-Gilboa-Ishai style. Each server receives one key. Evaluating both keys over the database gives two flag vectors whose XOR equals a one-hot vector at the target index. A server only sees its own key and flag share.

The security claim is deliberately scoped: no single non-colluding server learns the selected scenario index from its own view. The demo does not claim protection against collusion, side channels, timing leakage, deployment metadata, or production adversaries.

## 6. Garbled-Circuit Bridge

The comparison module demonstrates a private risk scalar comparison. Two laboratories can compare quantized risk values and reveal only whether one is lower than the other. The transcript describes wire-label commitment, evaluator input labels, and a single opened output bit. The implementation uses hash-derived labels for readability and should be replaced with production primitives before deployment.

## 7. Evaluation And Verification

The verification harness checks:

- PIR correctness and DPF one-hot reconstruction.
- PIR query byte accounting.
- Backend API behavior.
- Agent route guardrails.
- Garbled-circuit comparison behavior.
- Scenario precomputation.
- CLI predict/private-fetch/compare flows.
- Frontend production build.
- Documentation presence and honesty constraints.

The expected full verification result is:

```text
phase_10_full_verify PASS
```

## 8. Safety And Honesty

PROMETHEUS avoids three common failure modes:

1. Claiming synthetic output as trained scientific output.
2. Showing one heatmap for every physics family.
3. Overstating cryptographic privacy beyond the assumptions.

The UI makes route and provenance visible. Missing checkpoints are not hidden. When the system falls back to a deterministic renderer or numerical PDE solver, that source kind is shown in the first data table and in the warning notice.

## 9. Limitations

The installed trained checkpoints cover `MHD_64` and `post_neutron_star_merger` scalar slices. Supernova, acoustic, convection, reaction-diffusion, shear, and planetary routes need their own checkpoints before they can be called trained The Well output. The deterministic renderers are useful demo visualizers, not validated scientific solvers. The PIR database is small and fixed-width for demo purposes. The garbled-circuit bridge is educational and not a production MPC stack.

## 10. Reproducibility

The final folder includes the backend, frontend, Kaggle scripts, local training script, model artifacts, tests, docs, and launcher scripts. The MHD and post-merger checkpoints are stored locally. The post-merger Kaggle kernel has been regenerated with a fixed tensor-slicing path for 5D GRMHD fields and its output is installed.

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\FINAL_VERIFY.ps1
```

Then:

```powershell
powershell -ExecutionPolicy Bypass -File .\FINAL_RUN.ps1
```

Open `http://localhost:3000`.
