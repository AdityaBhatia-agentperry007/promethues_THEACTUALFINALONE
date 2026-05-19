# Technical Report

## Backend

The backend is a FastAPI service in `backend/app.py`.

Important endpoints:

- `GET /healthz`
- `GET /model/info`
- `GET /well/catalog`
- `POST /simulate`
- `POST /pir/fetch`
- `POST /mpc/compare`
- `POST /agent`

`/simulate` accepts:

```json
{
  "task": "create a black hole accretion disk in deep space",
  "steps": 72,
  "mode": "heat_radiation"
}
```

It returns frame arrays, route metadata, model provenance, interpretation rows, and warnings.

## Simulation Runtime

The runtime is `backend/simulation_runtime.py`.

It uses The Well route selection from `backend/well_catalog.py` and checks for local checkpoints under `backend/models`.

If a matching checkpoint exists, the runtime loads `WellEmulatorRuntime` and returns trained autoregressive frames. If no checkpoint exists, it returns a route-specific solver or renderer and explicitly marks the output as not trained.

## Trained Emulator

The installed checkpoints are:

```text
backend/models/well_mhd64_emulator.pt
backend/models/well_post_neutron_star_merger_emulator.pt
```

The model is a residual CNN next-frame emulator. The training scripts are:

- `ml/train_well_emulator.py`
- `kaggle/kernel/well_mhd64_kernel.py`
- `scripts/kaggle_submit_dataset.py`

The tensor extractor handles common The Well field shapes, including 5D post-merger field tensors.

This is not a reactor simulation. Reactor, fusion, and disruption wording in the demo is a product framing around confidential scientific infrastructure, not a validated reactor physics claim.

## Frontend

The frontend is a Next.js app in `frontend/src/app`.

The UI uses:

- one large canvas for frame playback;
- preset buttons;
- mode buttons;
- source/provenance tables;
- interpretation table;
- model registry;
- raw metadata view.

The visual theme is intentionally minimal, black/white, and assembly-like. Only the simulation canvas uses color because the user needs to inspect the physical field.

## PIR

The PIR implementation is pure Python:

- `backend/crypto/pir.py`
- `backend/crypto/pir_service.py`

The DPF primitive returns two keys whose evaluation shares reconstruct a one-hot vector. The CGKS fallback uses random bit queries over a replicated database. Both are real single-server-private demo constructions.

## Private Comparison

The comparison bridge is in `backend/crypto/gc_bridge.py`. It demonstrates a garbled-circuit-style private comparison of two risk scalars.

## Verification

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\FINAL_VERIFY.ps1
```

The verification script calls `scripts/verify_all.ps1`, which runs the Python and frontend checks.

## Main Limitations

- `MHD_64` and `post_neutron_star_merger` are installed as trained local checkpoints.
- Other routes need trained checkpoint files before they can be called The Well-trained output.
- The fallback renderers are not validated scientific solvers.
- PIR is demo scale and assumes non-colluding servers.
- The GC bridge is demonstrative, not production MPC.
