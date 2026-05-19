# Attribution

## Pre-existing Aditya assets

- `C:\Users\asus\og projects\prometheus`: prior PROMETHEUS surrogate and disruption scaffold. This build preserves the interface and uses a deterministic local fallback until that checkpoint is wired in.
- `C:\Users\asus\og projects\web app garbled circuits`: prior Next.js garbled-circuit visualizer. The dashboard links to this app and the backend includes a Python comparison bridge.

## External references

- Vadapalli, Bayatbabolghani, and Henry, "You May Also Like... Privacy", PoPETs 2021.4. The DPF primitive here follows the BGI-style `(2,2)` DPF lineage used by PIRSONA.
- The Well / PolymathicAI MHD_64 is the intended real-data source. The local build does not download or claim use of The Well data.
- Scalekit Agent Auth is an optional Gmail gateway. Credentials and account authorization are human-only setup steps.

## License hygiene

The PIR implementation is clean-room pure Python using only `hashlib` and `secrets`. No GPLv3 `dpf++` code is vendored.

