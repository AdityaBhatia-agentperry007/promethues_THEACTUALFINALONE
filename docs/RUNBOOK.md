# Runbook

## Build

```powershell
cd "C:\Users\asus\og projects\promethues_THEACTUALFINALONE"
powershell -ExecutionPolicy Bypass -File .\FINAL_BUILD.ps1
```

This checks Python, precomputes the PIR scenario library, and runs the Next.js production build.

## Start Demo

```powershell
powershell -ExecutionPolicy Bypass -File .\FINAL_RUN.ps1
```

Open:

```text
http://localhost:3000
```

## Verify

```powershell
powershell -ExecutionPolicy Bypass -File .\FINAL_VERIFY.ps1
```

## Ports

| Port | Service |
|---|---|
| `8000` | FastAPI backend |
| `3000` | Next.js frontend |

If ports are busy:

```powershell
netstat -ano | findstr :8000
netstat -ano | findstr :3000
```

Then stop the listed process if it is an old PROMETHEUS server.

## Known Checkpoint Behavior

`MHD_64` and `post_neutron_star_merger` are trained locally. Other routes report checkpoint missing unless their route-specific `.pt` file exists in `backend/models`.

For black-hole prompts, the target checkpoint is already installed:

```text
backend/models/well_post_neutron_star_merger_emulator.pt
```

If this file is removed or fails to load, the fallback black-hole renderer is useful for the demo visual, but the UI explicitly marks it as not The Well-trained.

## Kaggle Post-Merger Job

Completed fixed kernel path:

```text
kaggle/generated/prometheus-well-post-merger-emulator
```

Kaggle page:

```text
https://www.kaggle.com/agentperry007/prometheus-well-post-merger-emulator
```

The previous failure was caused by a post-merger tensor shape `(B,T,192,128,12)`. The trainer now keeps the first two axes as batch/time and selects a deterministic 2D scalar slice from the remaining field axes. The completed checkpoint is installed in `backend/models`.

## Environment Variables

Use `.env.example` as the template. Do not commit secrets into this folder.

Optional LLM explanation keys can be set in the shell before launching the backend. The demo works without them because the backend already returns structured interpretation rows.
