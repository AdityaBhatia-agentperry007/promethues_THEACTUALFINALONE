from __future__ import annotations

from typing import Literal
import json

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend import config
from backend.explainer import jobs
from backend.explainer.pipeline import run_pipeline
from backend.agent import run_agent
from backend.crypto.gc_bridge import compare_private
from backend.crypto.pir_service import get_library, pir_fetch
from backend.llm_explain import llm_status, maybe_explain
from backend.simulation_runtime import simulate_task, simulation_status
from backend.surrogate import get_engine
from backend.surrogate.well_emulator import get_well_emulator
from backend.well_catalog import catalog_rows, collection_summary, route_task_to_dataset

app = FastAPI(title="Animated Explainer API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        *config.CORS_ORIGINS,
    ],
    allow_origin_regex=r"https://.*\.(vercel\.app|run\.app)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/exports", StaticFiles(directory=str(config.EXPORTS_DIR)), name="exports")
app.mount("/fallback", StaticFiles(directory=str(config.FALLBACK_DIR)), name="fallback")

class PredictRequest(BaseModel):
    mach_sonic: float
    mach_alfvenic: float
    steps: int = Field(default=config.DEFAULT_STEPS, ge=1, le=64)


class PIRFetchRequest(BaseModel):
    scenario_index: int
    method: Literal["dpf", "cgks"] = "dpf"


class CompareRequest(BaseModel):
    lab_a_value: float
    lab_b_value: float


class AgentRequest(BaseModel):
    request_text: str
    channel: Literal["dashboard", "gmail"] = "dashboard"


class SimulateRequest(BaseModel):
    task: str
    steps: int = Field(default=48, ge=1, le=256)
    mode: str = Field(default="auto", min_length=1, max_length=64)


class ExplainerRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=400)

@app.get("/healthz")
def healthz() -> dict[str, object]:
    library = get_library()
    emulator = get_well_emulator()
    return {
        "status": "ok",
        "surrogate_loaded": get_engine().loaded,
        "well_emulator_loaded": emulator.loaded,
        "well_emulator_checkpoint": str(emulator.checkpoint_path),
        "well_catalog_rows": len(catalog_rows()),
        **simulation_status(),
        "n_scenarios": len(library.scenarios),
    }


@app.get("/scenarios")
def scenarios() -> dict[str, object]:
    return {"scenarios": config.SCENARIOS}


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, object]:
    try:
        prediction = get_engine().predict(request.mach_sonic, request.mach_alfvenic, request.steps)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"frames": prediction.frames, "risk": prediction.risk, "meta": prediction.meta}


@app.post("/pir/fetch")
def fetch(request: PIRFetchRequest) -> dict[str, object]:
    try:
        return pir_fetch(request.scenario_index, request.method)
    except (IndexError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/mpc/compare")
def mpc_compare(request: CompareRequest) -> dict[str, object]:
    return compare_private(request.lab_a_value, request.lab_b_value)


@app.post("/agent")
def agent(request: AgentRequest) -> dict[str, object]:
    return run_agent(request.request_text, request.channel)


@app.post("/simulate")
def simulate(request: SimulateRequest) -> dict[str, object]:
    result = simulate_task(request.task, request.steps, request.mode)
    llm_explanation = maybe_explain(result.task, result.meta, result.warning)
    return {
        "task": result.task,
        "frames": result.frames,
        "meta": result.meta,
        "route": route_task_to_dataset(request.task),
        "interpretation": result.meta.get("interpretation", []),
        "llm_explanation": llm_explanation,
        "warning": result.warning,
    }


def _explainer_worker(job_id: str, prompt: str) -> None:
    jobs.set(job_id, {"status": "rendering", "prompt": prompt})
    result = run_pipeline(prompt)
    jobs.set(job_id, result)


@app.post("/explainer/animate")
def explainer_animate(request: ExplainerRequest, background_tasks: BackgroundTasks) -> dict[str, object]:
    job_id = jobs.create()
    background_tasks.add_task(_explainer_worker, job_id, request.prompt)
    return {"job_id": job_id, "status": "queued"}


@app.get("/explainer/status/{job_id}")
def explainer_status(job_id: str) -> dict[str, object]:
    state = jobs.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return {"job_id": job_id, **state}

@app.get("/well/catalog")
def well_catalog() -> dict[str, object]:
    emulator = get_well_emulator()
    loaded_dataset = emulator.meta.get("dataset")
    return {
        "summary": collection_summary(),
        "datasets": catalog_rows(),
        "loaded_checkpoint_dataset": loaded_dataset,
        "trained_locally": [loaded_dataset] if loaded_dataset else [],
        "training_artifact": {
            "loaded": emulator.loaded,
            "path": str(emulator.checkpoint_path),
            "dataset": loaded_dataset,
            "frame_size": emulator.meta.get("frame_size"),
            "model_width": emulator.meta.get("model_width"),
            "train_loss": emulator.meta.get("train_loss"),
            "val_loss": emulator.meta.get("val_loss"),
        },
        "routing_examples": [
            {"prompt": "black hole accretion disk in deep space", "routes_to": "post_neutron_star_merger", "modes": ["intensity", "heat_radiation", "lensing", "doppler"]},
            {"prompt": "supernova blast wave in turbulent gas", "routes_to": "supernova_explosion_64", "modes": ["density", "shock_front", "temperature", "ejecta"]},
            {"prompt": "magnetic plasma turbulence", "routes_to": "MHD_64", "modes": ["field", "gradient", "shock_edges", "magnetic_proxy"]},
            {"prompt": "sound wave scattering through a maze", "routes_to": "acoustic_scattering_maze", "modes": ["pressure", "wave_energy", "maze_geometry", "interference"]},
            {"prompt": "thermal convection plume", "routes_to": "rayleigh_benard", "modes": ["temperature", "heat_flux", "plume_velocity", "rolls"]},
            {"prompt": "planetary storm vortex and jet stream", "routes_to": "planetswe", "modes": ["height", "vorticity", "jet_stream", "storm_track"]},
        ],
        "sources": [
            "https://polymathic-ai.org/the_well/",
            "https://polymathic-ai.org/the_well/datasets_overview/",
            "https://huggingface.co/collections/polymathic-ai/the-well",
        ],
        **simulation_status(),
    }


@app.get("/model/info")
def model_info() -> dict[str, object]:
    emulator = get_well_emulator()
    report_path = config.ROOT_DIR / "kaggle" / "outputs" / "training_report.json"
    report = None
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "checkpoint_loaded": emulator.loaded,
        "checkpoint_path": str(emulator.checkpoint_path),
        "load_error": emulator.load_error,
        "meta": emulator.meta,
        "training_report": report,
        "catalog_summary": collection_summary(),
        "catalog": catalog_rows(),
        **simulation_status(),
        "datasets_trained": emulator.meta.get("datasets_trained")
        or [
            {
                "name": emulator.meta.get("dataset", "MHD_64"),
                "source": emulator.meta.get("base_path", "hf://datasets/polymathic-ai/"),
                "train_split": "train",
                "validation_split": "valid",
                "field_extraction": "input_fields -> output_fields, auto-selected scalar 2D slice",
                "emulator_frame_size": emulator.meta.get("frame_size"),
            }
        ],
        "interpretation": {
            "model": "Residual CNN one-step frame emulator rolled forward autoregressively.",
            "input": "A deterministic seed frame selected from validation seed_bank using the task text hash.",
            "output": "Normalized scalar field intensity frames. The UI colors them with a scientific colormap.",
            "trained_scope": "The Well MHD_64 only, unless a newer checkpoint metadata says otherwise.",
            "limits": "Not black-hole GR, not NASA observational imagery, not a validated reactor simulator.",
        },
        "llm_explanation": {
            **llm_status(),
            "setup": "Set provider keys as environment variables if you want an external LLM to rewrite the already-real metadata. Secrets are not stored in this repo.",
            "safe_inputs": "task text, dataset route, checkpoint metadata, frame statistics; raw frame arrays are not needed.",
        },
    }
