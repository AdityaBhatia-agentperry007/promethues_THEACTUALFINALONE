# Demo Script

## Setup

```powershell
cd "C:\Users\asus\og projects\promethues_THEACTUALFINALONE"
powershell -ExecutionPolicy Bypass -File .\FINAL_BUILD.ps1
powershell -ExecutionPolicy Bypass -File .\FINAL_RUN.ps1
```

Open `http://localhost:3000`.

## Opening

"PROMETHEUS is a confidential physics-as-a-service demo. The important part is not just the animation. The important part is that the request routes to a physics dataset, the UI tells you whether the output is trained or fallback, and the private retrieval layer hides the requested scenario index from any single server."

## Demo Step 1: MHD

Click `#0 MHD`.

Point out:

- `LOADED_DATASET=MHD_64`.
- `data_source_kind=trained_the_well_checkpoint`.
- Modes: `field`, `gradient`, `shock_edges`, `magnetic_proxy`.

Say:

"This is the installed trained The Well checkpoint path. This is the route where the local trained model is available."

## Demo Step 2: Black Hole

Click `#1 BLACK_HOLE`, then mode `HEAT_RADIATION`, `LENSING`, or `DOPPLER`.

Point out:

- Route is `post_neutron_star_merger`.
- `JNZ ROUTE_CHECKPOINT_LOADED`.
- `data_source_kind=trained_the_well_checkpoint`.
- The output is a trained post-merger scalar-slice emulator, not MHD_64 and not telescope imagery.

Say:

"The demo does not pretend that MHD_64 is black-hole GR. It routes to the post-merger The Well family and uses the installed checkpoint."

## Demo Step 3: Acoustic Maze

Click `#3 MAZE_WAVE`.

Toggle:

- `PRESSURE`
- `WAVE_ENERGY`
- `MAZE_GEOMETRY`
- `INTERFERENCE`

Say:

"This route uses a live finite-difference wave solver when the matching The Well checkpoint is missing. The maze geometry and wave fields are actual backend frames, not a static video."

## Demo Step 4: More Presets

Click `#5 GRAY_SCOTT`, `#6 SHEAR`, and `#7 PLANET`.

Say:

"The same backend contract handles multiple physics families. Every response returns source rows, interpretation rows, frame stats, and supported modes."

## Demo Step 5: Privacy

Open a terminal:

```powershell
python cli/prometheus_cli.py private-fetch 4
```

Point out:

- `reconstructed_equals_direct=true`.
- `index_bits_leaked_to_any_single_server=0`.
- DPF query byte accounting.

Say:

"This is the confidential part: the single server view does not identify which scenario was requested."

## Demo Step 6: Private Comparison

```powershell
python cli/prometheus_cli.py compare 0.2 0.8
```

Say:

"The comparison bridge opens only one output bit: whether Lab A is lower risk."

## Closing

"PROMETHEUS is a demo of the full product loop: private request, routed simulation, visible provenance, and guardrails against fake scientific claims."
