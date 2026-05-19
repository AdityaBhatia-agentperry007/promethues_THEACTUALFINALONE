# AI Impact Statement

PROMETHEUS uses AI to parse a researcher request and drive a physics-surrogate workflow. The `/simulate` path runs Kaggle-trained The Well MHD_64 and post_neutron_star_merger checkpoints where available and animates frame-by-frame emulator output. It does not claim reactor-grade simulation, isolated event-horizon ray tracing, state-of-the-art MHD modeling, or a validated disruption predictor. Routes without checkpoints are labeled as fallback renderers or solvers.

Privacy is the central guardrail. The selected precomputed result is fetched through real two-server PIR so no single non-colluding server receives the scenario index. The garbled-circuit comparison reveals only a boolean and uses a readable 32-bit hash stand-in for AES, clearly labeled as a protocol demonstrator. Expected impact: make shared fusion simulation infrastructure more acceptable for proprietary research while keeping technical limits visible.
