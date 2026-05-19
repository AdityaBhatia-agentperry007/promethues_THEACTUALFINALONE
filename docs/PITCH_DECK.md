# Pitch Deck

## Slide 1: PROMETHEUS

Confidential physics-as-a-service for labs that need shared simulation access without revealing what scenario they are running.

## Slide 2: Problem

Simulation requests leak intent. In fusion, plasma, and safety research, the query can be as sensitive as the output.

## Slide 3: Product

A researcher types a physics task. PROMETHEUS routes it to a dataset family, generates frame-by-frame output, and shows provenance, model status, and useful interpretation tables.

## Slide 4: Demo

Live presets:

- MHD plasma turbulence from trained The Well `MHD_64`.
- Black-hole accretion route backed by the trained post-merger checkpoint.
- Supernova blast wave.
- Acoustic maze wave solver.
- Rayleigh-Benard convection.
- Gray-Scott reaction diffusion.
- Kelvin-Helmholtz shear.
- Planetary shallow-water vortex.

## Slide 5: Confidentiality

Two-server PIR lets a client fetch a scenario record without revealing the selected index to either single non-colluding server.

## Slide 6: Safety

The UI refuses fake provenance. If a checkpoint is missing, it says so in the first viewport.

## Slide 7: Architecture

Frontend -> FastAPI -> route classifier -> simulation runtime -> The Well checkpoint or route-specific solver -> metadata tables -> PIR and private comparison services.

## Slide 8: What Is Real

- Real FastAPI backend.
- Real trained local MHD checkpoint.
- Real CGKS and DPF PIR demo.
- Real frame-by-frame simulation buffers.
- Real verification harness.

## Slide 9: What Is Not Claimed

- Not production cryptography.
- Not a validated reactor simulator.
- Not NASA imagery.
- Not isolated event-horizon ray tracing or telescope imagery.

## Slide 10: Roadmap

1. Finish the remaining route-specific The Well checkpoints.
2. Replace hash-label GC demo with production MPC library.
3. Deploy two non-colluding PIR services.
4. Add audited model cards for every trained route.
5. Add usage policy and export-control review workflow.

## Slide 11: Ask

Pilot with labs that need private simulation access, benchmark additional The Well families, and harden the cryptographic deployment boundary.
