# A Provenance-Honest, Confidentiality-Preserving Workbench for AI-Native Physics Simulation and Explanation

**Technical Report / Preprint (v0.1)**
*This is a systems and design report describing a working research prototype. It reports preliminary, training-time metrics and explicitly bounds its claims; it is not a finished empirical research paper, and is structured so it can be upgraded into one once the benchmarks in §7 and §9 are completed.*

---

## Abstract

Shared scientific-simulation services are useful only if researchers can trust three things at once: the **fidelity** of the output, the **provenance** of how it was produced, and the **privacy** of the query itself. Hackathon and demo systems routinely blur these, presenting deterministic renderers as learned models and ignoring that a query stream can leak a laboratory's research direction. We present a workbench that separates and surfaces all three. A natural-language prompt is routed to a family of the Polymathic AI *The Well* dataset; depending on availability, it is served by a trained residual-CNN next-frame emulator, a numerical PDE solver, or a deterministic renderer — with the source explicitly labeled in every response. An LLM layer rewrites already-computed metadata into plain language and (optionally) generates a Manim concept animation through a sandboxed, self-correcting pipeline. A two-server Private Information Retrieval (PIR) layer, built clean-room from a Boyle–Gilboa–Ishai (2,2)-Distributed Point Function, lets a client retrieve a scenario record so that **neither single non-colluding server learns which record** was requested. We report the system design, the real (and modest) training metrics of the current emulators, and a precise statement of the privacy guarantee and its limits. The central contribution is a **discipline of honest provenance** as a first-class system property, alongside a concrete demonstration that query-index privacy can be added to a simulation service at demo scale.

**Keywords:** scientific machine learning, neural surrogates, private information retrieval, distributed point functions, provenance, simulation-as-a-service.

---

## 1. Introduction

Learned surrogate models increasingly approximate expensive physics simulations across fluid dynamics, plasma physics, and astrophysics. Shared services that expose such models lower the barrier to exploration but introduce two under-addressed problems.

**Problem 1 — provenance collapse.** When a single interface can return a trained-model output, a numerical-solver output, or a hand-authored deterministic rendering, users cannot tell which they are looking at. This is not a cosmetic issue: a deterministic renderer presented as a "trained AI physics model" is a false claim that, once discovered, discredits the entire system.

**Problem 2 — query leakage.** Even when output data is non-sensitive, the *pattern of requests* can reveal strategy. A laboratory repeatedly probing a narrow class of instabilities or a specific compact-object regime exposes its research direction to whoever operates the service.

This report describes a prototype that treats both as primary design constraints. We make provenance an explicit, per-response, UI-surfaced property, and we add a PIR layer that protects the queried index against a single non-colluding server. We are deliberately conservative about what the learned components claim; §7 reports their real, modest metrics and states their limitations plainly.

### 1.1 Contributions

1. A **provenance-honest** simulation runtime that labels every output as trained-checkpoint, numerical-solver, or deterministic-renderer, and refuses to claim trained output for routes whose checkpoint is absent.
2. A clean-room, dependency-free implementation of **two-server PIR** (CGKS and a BGI (2,2)-DPF) integrated into a simulation service, with a precise guarantee statement.
3. A **provider-agnostic explanation layer** and a design for a **sandboxed, self-correcting LLM→Manim** concept-animation pipeline, including a static AST allowlist that neither of the reference systems we build on provides.
4. An **honest empirical baseline**: real training numbers for two *The Well* emulators, together with an explicit account of why they are demo-grade rather than benchmarked.

---

## 2. Related work

**Neural surrogates and operators.** Fourier Neural Operators (Li et al., 2021) and DeepONet (Lu et al., 2021) established learned operators for PDE families; weather emulators such as GraphCast and FourCastNet demonstrated large-scale practical surrogates. Our current emulator is a deliberately small residual-CNN next-frame predictor — a baseline architecture — and we position the operator-based upgrade as future work (§9).

**Scientific simulation data.** The Polymathic AI *The Well* corpus provides standardized multi-physics simulation datasets with trajectory-level splits; we train on the `MHD_64` and `post_neutron_star_merger` families.

**Private Information Retrieval.** Chor–Goldreich–Kushilevitz–Sudan introduced multi-server PIR; Boyle, Gilboa & Ishai introduced function secret sharing and Distributed Point Functions, the basis for compact two-server PIR. Our DPF follows the (2,2) construction in the PIRSONA lineage (Vadapalli, Bayatbabolghani & Henry, *PoPETs* 2021.4), whose key observation is that per-party leaf flags XOR to the indicator vector `e_alpha`.

**LLM-driven animation.** manimator (HyperCluster) and TheoremExplainAgent (Ku et al., *arXiv* 2502.19400) generate Manim animations from prompts; the latter contributes a render-error self-correction loop. We adopt the generation/repair pattern but add a static execution sandbox absent from both.

---

## 3. System overview

The system comprises four layers behind one prompt interface:

1. **Routing** (`well_catalog.route_task_to_dataset`): keyword routing of prompt text to one of eight *The Well* families.
2. **Simulation runtime** (`simulation_runtime.simulate_task`): dispatches to a trained checkpoint, a numerical PDE solver, or a deterministic renderer, and annotates the response with a `data_source_kind` provenance tag.
3. **Explanation**: a provider-agnostic LLM rewrites computed metadata (`llm_explain.maybe_explain`); an async Manim pipeline (design in §8) renders concept animations.
4. **Confidential retrieval** (`crypto/pir_service`): CGKS or DPF two-server PIR over a fixed scenario library.

A FastAPI backend exposes nine endpoints; a Next.js frontend renders frames on a canvas with a provenance badge, per-frame statistics, and the explanation panel.

---

## 4. Provenance-honest simulation

The runtime attaches to every response a tag:

- `trained_the_well_checkpoint` — a `.pt` emulator matching the routed dataset was loaded and rolled forward;
- `missing_checkpoint` — no matching checkpoint exists; the response is a labeled fallback;
- `deterministic_physics_renderer` — an analytic/PDE renderer produced the frames.

It also reports `trained_for_request` (does the loaded checkpoint's dataset match the routed dataset). Of eight presets, two are backed by trained checkpoints (MHD plasma, black-hole/post-merger); six are numerical solvers or deterministic renderers. The interface surfaces this distinction rather than hiding it. We argue this **honest-by-construction** posture is the correct default for shared scientific services, where a single false fidelity claim is more damaging than a visible limitation.

---

## 5. Surrogate model

Each emulator is a residual convolutional network (`ResidualFrameEmulator`, width 64) trained for one-step next-frame prediction on an auto-selected scalar slice of the routed dataset, with outputs bilinearly resampled to a 128×128 display grid. At inference, a deterministic hash of the prompt selects a validation seed frame, which is rolled forward autoregressively. We emphasize (and the UI states) that this is a **one-step scalar-slice emulator**, not a multi-field solver and not GR or reactor-grade physics.

---

## 6. Confidential retrieval

We integrate two real two-server PIR schemes (pure `hashlib`/`secrets`, no third-party crypto):

**CGKS (information-theoretic).** The client samples a random bit-vector `q0` and sets `q1 = q0 ⊕ e_j`; each server returns the XOR of the database records selected by its query; the client XORs the two answers to recover record *j*. Query size is O(n) bits. Always correct; used as the fallback.

**BGI (2,2)-DPF.** A Distributed Point Function produces two keys whose full-domain evaluations are additive shares of the indicator vector `e_j`. Each server XOR-accumulates the records flagged by its share; the client XORs the two answers to recover record *j*. Query size is O(λ·log n). The per-party leaf flags satisfy `flags0 ⊕ flags1 = e_alpha`, matching the PIRSONA observation.

**Guarantee and limits.** Against a **single non-colluding server**, the queried index *j* is information-theoretically hidden (CGKS) or computationally hidden under the PRG (DPF). We claim **nothing** against server collusion, traffic analysis, timing/access-pattern side channels, or metadata. For production, the SHAKE256 PRG should be replaced by fixed-key AES; the protocol logic is unchanged.

---

## 7. Preliminary results (and their limits)

We report the **actual** training-time metrics from the shipped checkpoints.

| Dataset | Train / Val items | Epochs | Batch | Width | Best Val MSE |
|---|---|---|---|---|---|
| `MHD_64` | 7,623 / 990 | 6 | 1 | 64 | 0.2199 |
| `post_neutron_star_merger` | 1,080 / 180 | 3 | 1 | 64 | 0.2917 |

The `MHD_64` validation curve over six epochs is `0.2207, 0.2227, 0.3192, 0.2200, 0.2203, 0.2199`.

**Honest interpretation.** (i) The curve **oscillates and barely improves** beyond the epoch-1 value, indicating the optimization is near a plateau under this capacity/schedule. (ii) The training report contains **no identity/persistence baseline**, so we cannot yet claim the model meaningfully beats "predict the previous frame." (iii) Batch size 1, six epochs, and a single auto-selected scalar slice are demo-scale settings. We therefore characterize these checkpoints as **demo-grade**: useful for an honest, provenance-labeled visualization, but **not** a benchmarked surrogate-accuracy result. Establishing an identity baseline, increasing capacity with early stopping, and auditing trajectory-level split disjointness are prerequisites (§9) before any accuracy claim is made.

---

## 8. LLM explanation and animation (design)

The explanation layer is provider-agnostic (OpenAI / Anthropic / Gemini / Vertex AI) and is constrained to rewrite **already-computed metadata** — it is instructed not to assert datasets, solvers, or imagery that were not used. The planned animation layer adapts the generate→extract→render pipeline of manimator and the render-error self-correction loop of TheoremExplainAgent, with three additions required for safe deployment: (i) a **static AST allowlist sandbox** that rejects any generated script importing or calling outside a small allowlist before it is executed; (ii) **asynchronous** rendering (job submission + polling), since Manim rendering exceeds request timeouts; and (iii) a **pre-rendered fallback library** keyed by routed dataset, so a failed live generation degrades to a prepared animation rather than an error. Generated animations are labeled as **concept illustrations**, explicitly distinct from model-predicted simulation frames.

---

## 9. Limitations and future work

- **Surrogate fidelity.** Establish an identity/persistence baseline; increase model capacity with early stopping and LR decay; audit train/validation trajectory disjointness for leakage; evaluate a Fourier Neural Operator against the residual-CNN baseline.
- **Private comparison.** The current "garbled-circuit comparison" is a **plaintext placeholder providing no input privacy**, labeled as such. Replace it with a real Yao garbled circuit (wire labels, point-and-permute garbled tables, oblivious transfer), verified by exhaustive small-bit differential testing.
- **PIR hardening.** Replace SHAKE256 with fixed-key AES; analyze access-pattern leakage at the service layer.
- **Provenance evaluation.** Empirically test whether users correctly distinguish trained vs. rendered output with and without the badge.

---

## 10. Conclusion

We described a workbench that elevates **provenance honesty** to a first-class system property and demonstrates **query-index privacy** for a simulation service via clean-room two-server PIR. We report the real, modest metrics of the current learned components and bound their claims precisely. The contribution is not a state-of-the-art surrogate; it is an architecture and a discipline — label what produced every output, and let a query stay private — that we argue shared scientific-simulation services should adopt. The roadmap in §9 converts this prototype into a system whose accuracy and privacy claims are empirically substantiated.

---

## Reproducibility & artifacts

- Code: the accompanying repository (FastAPI backend, Next.js frontend, training scripts, PIR module, test harness).
- Data: Polymathic AI *The Well* (`MHD_64`, `post_neutron_star_merger`), CC-BY-4.0.
- Metrics in §7 are read directly from `kaggle/outputs/training_report.json` and `kaggle/outputs_post_merger/training_report.json`.
- Verification: `scripts/mini_pytest.py` and `scripts/verify.py` (PIR correctness, model loading, API contract).

## References (selected)

1. B. Chor, O. Goldreich, E. Kushilevitz, M. Sudan. *Private Information Retrieval.* FOCS 1995.
2. E. Boyle, N. Gilboa, Y. Ishai. *Function Secret Sharing.* EUROCRYPT 2015; and *Function Secret Sharing: Improvements and Extensions.* CCS 2016.
3. A. Vadapalli, F. Bayatbabolghani, R. Henry. *You May Also Like... Privacy (PIRSONA).* PoPETs 2021.4.
4. Z. Li et al. *Fourier Neural Operator for Parametric PDEs.* ICLR 2021.
5. L. Lu et al. *DeepONet.* Nature Machine Intelligence, 2021.
6. The Well — Polymathic AI. *A 15TB collection of physics simulation datasets.* NeurIPS 2024 (datasheet: arXiv 2412.00568).
7. Ku et al. *TheoremExplainAgent.* arXiv 2502.19400, 2025.

---

*Claim-integrity statement: every quantitative result in this report is read from the repository's training artifacts. Where a capability is not yet substantiated (surrogate accuracy vs. baseline; input-private comparison), this report says so explicitly rather than implying a result. This document should not be presented as a peer-reviewed paper; it is a preprint-style technical report intended for honest external review and iteration.*
