This contract is frozen. All payloads are JSON.

## FastAPI Backend

Default base URL: `http://localhost:8000`.

### `GET /healthz`

Returns:

```json
{"status":"ok","surrogate_loaded":true,"n_scenarios":16}
```

### `GET /scenarios`

Returns:

```json
{"scenarios":[{"index":0,"label":"subsonic, sub-Alfvenic laminar reference","mach_sonic":0.5,"mach_alfvenic":0.7,"mode":"laminar reference"}]}
```

### `POST /predict`

Body:

```json
{"mach_sonic":1.5,"mach_alfvenic":0.7,"steps":12}
```

Returns:

```json
{"frames":[[[0.0]]],"risk":0.61,"meta":{"resolution":[18,18],"steps":12,"source":"surrogate"}}
```

### `POST /pir/fetch`

Body:

```json
{"scenario_index":4,"method":"dpf"}
```

Returns:

```json
{
  "record_summary":{"risk":0.61,"resolution":[18,18],"steps":12},
  "reconstructed_equals_direct":true,
  "server0_view":"hex",
  "server1_view":"hex",
  "index_bits_leaked_to_any_single_server":0,
  "method":"dpf",
  "query_bytes":178
}
```

### `POST /mpc/compare`

Body:

```json
{"lab_a_value":0.31,"lab_b_value":0.58}
```

Returns:

```json
{"a_lower":true,"transcript":["..."],"note":"Protocol demonstrator: 32-bit hash stands in for AES."}
```

### `POST /agent`

Body:

```json
{"request_text":"confidential supersonic sub-Alfvenic prediction","channel":"dashboard"}
```

Returns:

```json
{
  "parsed":{"mach_sonic":1.5,"mach_alfvenic":0.7,"mode":"sub-Alfvenic confinement","intent":"predict_private"},
  "plan":["parse_scenario","run_surrogate","private_fetch","summarize"],
  "tool_trace":[{"tool":"parse_scenario","input":{},"output_summary":{}}],
  "result":{"risk":0.61,"used_private_fetch":true},
  "answer_text":"..."
}
```

### `POST /simulate`

This is the real emulator path used by the minimal UI.

Body:

```json
{"task":"generate an MHD plasma simulation","steps":48}
```

Returns, when a Kaggle/Colab-trained Well checkpoint is present:

```json
{
  "task":"generate an MHD plasma simulation",
  "frames":[[[0.0,0.5,1.0]]],
  "meta":{"checkpoint_loaded":true,"dataset":"MHD_64","steps":48},
  "warning":null
}
```

Returns, before the real checkpoint is copied into `backend/models/`:

```json
{
  "task":"generate a black hole simulation",
  "frames":[],
  "meta":{"checkpoint_loaded":false,"dataset_hint":"post_neutron_star_merger"},
  "warning":"No The Well-trained emulator checkpoint is loaded. Run the Kaggle/Colab training script first."
}
```

## Agent Tools

- `parse_scenario(request_text: str)` returns `{mach_sonic, mach_alfvenic, mode, intent, scenario_index}`.
- `run_surrogate(mach_sonic: float, mach_alfvenic: float, steps: int)` returns `{risk, frames_ref, frames, meta}`.
- `private_fetch(scenario_index: int, method: str)` returns the `/pir/fetch` payload.
- `compare_private(lab_a_value: float, lab_b_value: float)` returns the `/mpc/compare` payload.
- `summarize(context: str)` returns final natural-language text.

## Scenario Library

`N_SCENARIOS = 16`. Records are JSON serialized and padded to one fixed byte width before PIR XOR. `/simulate` is checkpoint-gated and does not claim Well training unless `backend/models/well_mhd64_emulator.pt` exists.
