from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from backend import config
from backend.surrogate.well_emulator import SimulationResult, WellEmulatorRuntime, get_well_emulator
from backend.well_catalog import catalog_lookup, route_task_to_dataset


CHECKPOINTS: dict[str, Path] = {
    "MHD_64": config.WELL_EMULATOR_CHECKPOINT,
    "post_neutron_star_merger": config.MODELS_DIR / "well_post_neutron_star_merger_emulator.pt",
    "supernova_explosion_64": config.MODELS_DIR / "well_supernova_explosion_64_emulator.pt",
    "acoustic_scattering_maze": config.MODELS_DIR / "well_acoustic_scattering_maze_emulator.pt",
    "rayleigh_benard": config.MODELS_DIR / "well_rayleigh_benard_emulator.pt",
    "gray_scott_reaction_diffusion": config.MODELS_DIR / "well_gray_scott_emulator.pt",
    "shear_flow": config.MODELS_DIR / "well_shear_flow_emulator.pt",
    "planetswe": config.MODELS_DIR / "well_planetswe_emulator.pt",
}


SUPPORTED_MODES: dict[str, tuple[str, ...]] = {
    "MHD_64": ("field", "gradient", "shock_edges", "magnetic_proxy"),
    "post_neutron_star_merger": ("intensity", "heat_radiation", "lensing", "doppler"),
    "supernova_explosion_64": ("density", "shock_front", "temperature", "ejecta"),
    "acoustic_scattering_maze": ("pressure", "wave_energy", "maze_geometry", "interference"),
    "rayleigh_benard": ("temperature", "heat_flux", "plume_velocity", "rolls"),
    "gray_scott_reaction_diffusion": ("concentration", "reaction_rate", "pattern_edges"),
    "shear_flow": ("scalar", "vorticity", "mixing", "velocity"),
    "planetswe": ("height", "vorticity", "jet_stream", "storm_track"),
}


def simulate_task(task: str, steps: int, mode: str = "auto") -> SimulationResult:
    route = route_task_to_dataset(task)
    dataset = str(route["recommended_dataset"])
    steps = max(1, min(int(steps), 256))
    selected_mode = _resolve_mode(dataset, mode)

    if dataset == "post_neutron_star_merger":
        return _analytic_black_hole(task, steps, route, selected_mode)

    runtime = _runtime_for_dataset(dataset)
    if runtime is not None and runtime.loaded:
        result = runtime.simulate(task, steps)
        frames = _apply_mode_frames(_arrays_from_frames(result.frames), selected_mode, dataset)
        result.meta.update(
            {
                "simulation_kind": "the_well_trained_emulator",
                "visual_style": _visual_style_for_dataset(dataset),
                "data_source_kind": "trained_the_well_checkpoint",
                "data_source_rows": _trained_source_rows(result.meta, dataset)
                + [_source("selected_mode", selected_mode, _mode_transform_note(dataset, selected_mode))],
                "selected_preset_is_custom": True,
                "mode": selected_mode,
                "supported_modes": list(_mode_list(dataset)),
                "mode_transform": _mode_transform_note(dataset, selected_mode),
            }
        )
        if frames:
            result.frames = _to_frames(frames)
            result.meta["frame_stats"] = [_frame_stats(frame, i) for i, frame in enumerate(frames)]
        result.meta["interpretation"] = _interpretation_rows(task, result.meta, route, result.warning)
        return result

    if dataset.startswith("supernova"):
        return _analytic_supernova(task, steps, route, selected_mode)
    if dataset.startswith("acoustic"):
        return _numerical_acoustic(task, steps, route, selected_mode)
    if dataset == "rayleigh_benard":
        return _analytic_convection(task, steps, route, selected_mode)
    if dataset == "gray_scott_reaction_diffusion":
        return _reaction_diffusion(task, steps, route, selected_mode)
    if dataset == "shear_flow":
        return _analytic_shear(task, steps, route, selected_mode)
    if dataset == "planetswe":
        return _analytic_planetary(task, steps, route, selected_mode)

    runtime = get_well_emulator()
    if runtime.loaded:
        result = runtime.simulate(task, steps)
        fallback_mode = _resolve_mode("MHD_64", mode)
        frames = _apply_mode_frames(_arrays_from_frames(result.frames), fallback_mode, "MHD_64")
        result.meta.update(
            {
                "simulation_kind": "the_well_trained_emulator",
                "visual_style": "mhd_scalar",
                "data_source_kind": "trained_the_well_checkpoint",
                "data_source_rows": _trained_source_rows(result.meta, "MHD_64")
                + [_source("selected_mode", fallback_mode, _mode_transform_note("MHD_64", fallback_mode))],
                "selected_preset_is_custom": True,
                "mode": fallback_mode,
                "supported_modes": list(_mode_list("MHD_64")),
                "mode_transform": _mode_transform_note("MHD_64", fallback_mode),
            }
        )
        if frames:
            result.frames = _to_frames(frames)
            result.meta["frame_stats"] = [_frame_stats(frame, i) for i, frame in enumerate(frames)]
        result.meta["interpretation"] = _interpretation_rows(task, result.meta, route, result.warning)
        return result
    else:
        return _analytic_mhd(task, steps, route, selected_mode)


def simulation_status() -> dict[str, Any]:
    rows = []
    for dataset, path in CHECKPOINTS.items():
        rows.append(
            {
                "dataset": dataset,
                "checkpoint_path": str(path),
                "checkpoint_exists": path.exists(),
                "fallback": _fallback_name(dataset),
                "visual_style": _visual_style_for_dataset(dataset),
                "supported_modes": list(_mode_list(dataset)),
            }
        )
    return {"simulation_modes": rows}


@lru_cache(maxsize=12)
def _cached_runtime(path: str) -> WellEmulatorRuntime:
    return WellEmulatorRuntime(Path(path))


def _runtime_for_dataset(dataset: str) -> WellEmulatorRuntime | None:
    path = CHECKPOINTS.get(dataset)
    if path is None or not path.exists():
        return None
    if dataset == "MHD_64" and path == config.WELL_EMULATOR_CHECKPOINT:
        return get_well_emulator()
    return _cached_runtime(str(path))


def _fallback_name(dataset: str) -> str:
    if dataset == "post_neutron_star_merger":
        return "analytic_schwarzschild_accretion_renderer"
    if dataset.startswith("supernova"):
        return "analytic_sedov_taylor_shell_renderer"
    if dataset.startswith("acoustic"):
        return "finite_difference_wave_equation"
    if dataset == "rayleigh_benard":
        return "analytic_rayleigh_benard_plume_renderer"
    if dataset == "gray_scott_reaction_diffusion":
        return "gray_scott_pde_solver"
    if dataset == "shear_flow":
        return "analytic_kelvin_helmholtz_renderer"
    if dataset == "planetswe":
        return "analytic_shallow_water_planetary_renderer"
    return "the_well_checkpoint_required"


def _visual_style_for_dataset(dataset: str) -> str:
    if dataset == "post_neutron_star_merger":
        return "black_hole"
    if dataset.startswith("supernova"):
        return "supernova"
    if dataset.startswith("acoustic"):
        return "acoustic"
    if dataset == "rayleigh_benard":
        return "convection"
    if dataset == "gray_scott_reaction_diffusion":
        return "reaction"
    if dataset == "shear_flow":
        return "shear"
    if dataset == "planetswe":
        return "planetary"
    return "mhd_scalar"


def _mode_key(dataset: str) -> str:
    if dataset.startswith("supernova"):
        return "supernova_explosion_64"
    if dataset.startswith("acoustic"):
        return "acoustic_scattering_maze"
    return dataset if dataset in SUPPORTED_MODES else "MHD_64"


def _mode_list(dataset: str) -> tuple[str, ...]:
    return SUPPORTED_MODES[_mode_key(dataset)]


def _resolve_mode(dataset: str, requested: str | None) -> str:
    modes = _mode_list(dataset)
    normalized = (requested or "auto").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"", "auto", "default"}:
        return modes[0]
    aliases = {
        "radiation": "heat_radiation",
        "heat": "heat_radiation",
        "shock": "shock_front",
        "geometry": "maze_geometry",
        "energy": "wave_energy",
        "field": "field",
        "intensity": "intensity",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in modes else modes[0]


def _mode_transform_note(dataset: str, mode: str) -> str:
    if mode in {"field", "intensity", "density", "pressure", "temperature", "concentration", "height", "scalar"}:
        return "direct normalized scalar frame"
    if dataset == "MHD_64" and mode == "magnetic_proxy":
        return "derived proxy from the trained scalar slice; not a separate vector magnetic-field channel"
    return "derived view computed from the selected frame sequence"


def _arrays_from_frames(frames: list[list[list[float]]]) -> list[np.ndarray]:
    return [np.asarray(frame, dtype=np.float32) for frame in frames]


def _apply_mode_frames(frames: list[np.ndarray], mode: str, dataset: str) -> list[np.ndarray]:
    if not frames:
        return []
    direct_modes = {"field", "intensity", "density", "pressure", "temperature", "concentration", "height", "scalar"}
    if mode in direct_modes:
        return [_normalize(frame) for frame in frames]
    transformed: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        gradient = _gradient_magnitude(frame)
        if mode in {"gradient", "shock_edges", "shock_front", "pattern_edges", "vorticity", "interference", "mixing"}:
            shown = gradient
        elif mode in {"wave_energy", "heat_flux", "reaction_rate", "magnetic_proxy"}:
            shown = frame * frame + 0.85 * gradient
        elif mode in {"heat_radiation", "ejecta"}:
            shown = np.power(np.clip(frame, 0.0, None), 1.35) + 0.2 * gradient
        elif mode in {"doppler", "jet_stream", "plume_velocity", "velocity", "storm_track"}:
            yy, xx = np.indices(frame.shape, dtype=np.float32)
            shear = (xx / max(1.0, frame.shape[1] - 1.0)) * 0.45 + 0.75
            shown = frame * shear + 0.22 * gradient + 0.04 * np.sin(index * 0.19)
        elif mode == "lensing":
            shown = frame + 0.75 * gradient
        else:
            shown = frame
        transformed.append(_normalize(shown))
    return transformed


def _gradient_magnitude(frame: np.ndarray) -> np.ndarray:
    dy, dx = np.gradient(frame.astype(np.float32))
    return np.sqrt(dx * dx + dy * dy)


def _rng(task: str, salt: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{salt}:{task}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _mesh(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    return x, y, r, theta


def _analytic_black_hole(task: str, steps: int, route: dict[str, Any], mode: str) -> SimulationResult:
    size = 192
    rng = _rng(task, "black-hole")
    x, y, r, theta = _mesh(size)
    spin = float(rng.uniform(0.35, 0.88))
    inclination = float(rng.uniform(0.45, 0.78))
    accretion_rate = float(rng.uniform(0.55, 1.25))
    shadow_radius = float(rng.uniform(0.145, 0.19))
    frames: list[np.ndarray] = []
    for t in range(steps):
        phase = 2.0 * math.pi * t / max(steps, 1)
        xp = x * math.cos(0.2 * phase) - y * math.sin(0.2 * phase)
        yp = x * math.sin(0.2 * phase) + y * math.cos(0.2 * phase)
        disk_r = np.sqrt(xp * xp + (yp / inclination) ** 2)
        swirl = 0.035 * np.sin(7.0 * theta - 2.8 * phase + 2.5 * spin)
        disk = np.exp(-((disk_r - (0.46 + swirl)) / 0.075) ** 2)
        far_disk = 0.52 * np.exp(-((disk_r - (0.72 - 0.02 * np.sin(phase))) / 0.16) ** 2)
        photon_ring = 1.45 * np.exp(-((r - (shadow_radius + 0.055)) / 0.018) ** 2)
        lensed_top = 0.78 * np.exp(-((y + 0.18 + 0.03 * np.sin(phase)) / 0.055) ** 2) * np.exp(-(x / 0.7) ** 2)
        doppler = 0.68 + 0.72 / (1.0 + np.exp(-7.5 * (x + 0.12 * np.sin(phase))))
        shadow = r < (shadow_radius + 0.018 * np.sin(theta - phase))
        if mode == "heat_radiation":
            frame = (1.35 * disk + 0.75 * far_disk + 0.35 * photon_ring) * (0.82 + 0.42 * doppler) * accretion_rate
        elif mode == "lensing":
            frame = 1.7 * photon_ring + 1.15 * lensed_top + 0.28 * far_disk
        elif mode == "doppler":
            asymmetry = np.clip(doppler - 0.48, 0.0, None)
            frame = (1.15 * disk + 0.45 * photon_ring + 0.28 * far_disk) * asymmetry * accretion_rate
        else:
            frame = (disk + far_disk + photon_ring + lensed_top) * doppler * accretion_rate
        frame *= np.exp(-0.08 * r * r)
        frame[shadow] = 0.0
        frame = _normalize(frame)
        frames.append(frame)
    meta = _base_meta(task, route, "black_hole_analytic", "black_hole", size, steps, mode)
    meta.update(
        {
            "dataset": "post_neutron_star_merger",
            "trained_for_request": False,
            "data_source_kind": "deterministic_physics_renderer",
            "prediction_method": f"analytic Schwarzschild-lensing-inspired accretion disk renderer, mode={mode}",
            "parameters": {
                "spin_proxy": round(spin, 4),
                "inclination_proxy": round(inclination, 4),
                "accretion_rate_proxy": round(accretion_rate, 4),
                "shadow_radius_proxy": round(shadow_radius, 4),
            },
            "data_source_rows": [
                _source("target_dataset", "post_neutron_star_merger", "The Well GRMHD dataset route for compact-object/accretion prompts"),
                _source("local_checkpoint", "missing", str(CHECKPOINTS["post_neutron_star_merger"])),
                _source("selected_mode", mode, "Switchable accretion/lensing/radiation view generated by the renderer"),
                _source("rendered_now", "analytic black-hole/accretion disk model", "Not MHD_64 heatmap; not The Well-trained until checkpoint exists"),
            ],
        }
    )
    warning = (
        "No local post_neutron_star_merger checkpoint is installed yet. "
        "This black-hole view is a deterministic physics renderer, not MHD_64 and not a trained The Well GRMHD checkpoint."
    )
    meta["frame_stats"] = [_frame_stats(frame, i) for i, frame in enumerate(frames)]
    meta["interpretation"] = _interpretation_rows(task, meta, route, warning)
    return SimulationResult(task, _to_frames(frames), meta, warning)


def _analytic_supernova(task: str, steps: int, route: dict[str, Any], mode: str) -> SimulationResult:
    size = 160
    rng = _rng(task, "supernova")
    x, y, r, theta = _mesh(size)
    modes = rng.uniform(0.4, 1.0, size=6)
    phases = rng.uniform(0.0, 2 * math.pi, size=6)
    frames: list[np.ndarray] = []
    for t in range(steps):
        tau = (t + 1) / max(steps, 1)
        radius = 0.12 + 0.72 * tau ** 0.42
        turbulence = sum(modes[k] * np.sin((k + 3) * theta + phases[k] + 2.5 * tau) for k in range(6)) / 6.0
        shell_r = radius + 0.045 * turbulence
        shell = np.exp(-((r - shell_r) / (0.035 + 0.02 * tau)) ** 2)
        hot_core = np.exp(-(r / (0.18 + 0.2 * tau)) ** 2) * (1.1 - 0.7 * tau)
        ejecta = 0.45 * np.exp(-((r - 0.65 * radius) / 0.19) ** 2) * (0.8 + turbulence)
        density = shell + hot_core + ejecta
        if mode == "shock_front":
            shown = _gradient_magnitude(density) + 0.65 * shell
        elif mode == "temperature":
            shown = 1.45 * hot_core + 0.85 * shell
        elif mode == "ejecta":
            shown = ejecta + 0.35 * shell
        else:
            shown = density
        frames.append(_normalize(shown))
    meta = _base_meta(task, route, "supernova_analytic", "supernova", size, steps, mode)
    meta.update(
        {
            "dataset": "supernova_explosion_64",
            "trained_for_request": False,
            "data_source_kind": "deterministic_physics_renderer",
            "prediction_method": f"Sedov-Taylor-like expanding blast shell with deterministic turbulent angular modes, mode={mode}",
            "data_source_rows": [
                _source("target_dataset", "supernova_explosion_64", "The Well route for blast-wave prompts"),
                _source("local_checkpoint", "missing", str(CHECKPOINTS["supernova_explosion_64"])),
                _source("selected_mode", mode, "Switchable density/shock/temperature/ejecta output field"),
                _source("rendered_now", "analytic expanding-shell model", "Uses blast-wave scaling; not The Well-trained until checkpoint exists"),
            ],
        }
    )
    warning = "No local supernova_explosion_64 checkpoint is installed. Rendering a deterministic blast-wave model instead."
    meta["frame_stats"] = [_frame_stats(frame, i) for i, frame in enumerate(frames)]
    meta["interpretation"] = _interpretation_rows(task, meta, route, warning)
    return SimulationResult(task, _to_frames(frames), meta, warning)


def _numerical_acoustic(task: str, steps: int, route: dict[str, Any], mode: str) -> SimulationResult:
    size = 192
    x, y, _, _ = _mesh(size)
    field = np.exp(-((x + 0.72) ** 2 + (y + 0.12) ** 2) / 0.012).astype(np.float32)
    prev = field.copy() * 0.985
    obstacle = _maze_obstacle(size)
    field[obstacle] = 0.0
    prev[obstacle] = 0.0
    c2 = 0.145
    source_y, source_x = int(size * 0.48), int(size * 0.08)
    frames: list[np.ndarray] = []
    tick = 0
    for _ in range(steps):
        for _ in range(4):
            lap = (
                np.roll(field, 1, 0)
                + np.roll(field, -1, 0)
                + np.roll(field, 1, 1)
                + np.roll(field, -1, 1)
                - 4.0 * field
            )
            nxt = 1.992 * field - 0.996 * prev + c2 * lap
            drive = math.sin(0.33 * tick) * math.exp(-tick / 360.0)
            nxt[source_y - 3 : source_y + 4, source_x - 3 : source_x + 4] += 0.22 * drive
            nxt[obstacle] = 0.0
            _damp_edges(nxt)
            prev, field = field, nxt.astype(np.float32)
            tick += 1
        velocity = field - prev
        if mode == "wave_energy":
            shown = field * field + 0.65 * velocity * velocity
            shown[obstacle] = 0.0
        elif mode == "maze_geometry":
            shown = 0.28 * _normalize(np.abs(field))
            shown[obstacle] = 1.0
        elif mode == "interference":
            lap_abs = np.abs(
                np.roll(field, 1, 0)
                + np.roll(field, -1, 0)
                + np.roll(field, 1, 1)
                + np.roll(field, -1, 1)
                - 4.0 * field
            )
            shown = lap_abs + 0.3 * np.abs(velocity)
            shown[obstacle] = 0.0
        else:
            shown = 0.5 + 0.5 * np.tanh(field * 3.6)
            shown[obstacle] = 0.02
        frames.append(_normalize(shown))
    meta = _base_meta(task, route, "acoustic_wave_equation", "acoustic", size, steps, mode)
    meta.update(
        {
            "dataset": "acoustic_scattering_maze",
            "trained_for_request": False,
            "data_source_kind": "numerical_pde_solver",
            "prediction_method": f"finite-difference 2D scalar wave equation with fixed maze obstacles, mode={mode}",
            "data_source_rows": [
                _source("target_dataset", "acoustic_scattering_maze", "The Well route for maze/scattering prompts"),
                _source("local_checkpoint", "missing", str(CHECKPOINTS["acoustic_scattering_maze"])),
                _source("selected_mode", mode, "Pressure, wave energy, wall geometry, or interference view"),
                _source("rendered_now", "2D wave-equation solver", "Numerical PDE frames generated live; not The Well-trained until checkpoint exists"),
            ],
        }
    )
    warning = "No local acoustic_scattering_maze checkpoint is installed. Rendering a live finite-difference wave simulation instead."
    meta["frame_stats"] = [_frame_stats(frame, i) for i, frame in enumerate(frames)]
    meta["interpretation"] = _interpretation_rows(task, meta, route, warning)
    return SimulationResult(task, _to_frames(frames), meta, warning)


def _maze_obstacle(size: int) -> np.ndarray:
    obstacle = np.zeros((size, size), dtype=bool)
    wall = max(4, size // 40)
    obstacle[:wall, :] = True
    obstacle[-wall:, :] = True
    obstacle[:, :wall] = True
    obstacle[:, -wall:] = True

    def vertical(x0: int, y0: int, y1: int, gaps: tuple[tuple[int, int], ...]) -> None:
        obstacle[y0:y1, x0 : x0 + wall] = True
        for a, b in gaps:
            obstacle[a:b, x0 : x0 + wall] = False

    def horizontal(y0: int, x0: int, x1: int, gaps: tuple[tuple[int, int], ...]) -> None:
        obstacle[y0 : y0 + wall, x0:x1] = True
        for a, b in gaps:
            obstacle[y0 : y0 + wall, a:b] = False

    vertical(int(size * 0.33), int(size * 0.12), int(size * 0.86), ((int(size * 0.42), int(size * 0.53)),))
    vertical(int(size * 0.58), int(size * 0.18), int(size * 0.93), ((int(size * 0.68), int(size * 0.78)),))
    vertical(int(size * 0.79), int(size * 0.10), int(size * 0.70), ((int(size * 0.30), int(size * 0.39)),))
    horizontal(int(size * 0.22), int(size * 0.18), int(size * 0.88), ((int(size * 0.42), int(size * 0.51)),))
    horizontal(int(size * 0.49), int(size * 0.08), int(size * 0.70), ((int(size * 0.22), int(size * 0.31)),))
    horizontal(int(size * 0.72), int(size * 0.26), int(size * 0.95), ((int(size * 0.63), int(size * 0.75)),))
    return obstacle


def _damp_edges(frame: np.ndarray) -> None:
    border = min(12, max(4, frame.shape[0] // 16))
    for i in range(border):
        factor = 0.55 + 0.45 * i / max(1, border - 1)
        frame[i, :] *= factor
        frame[-i - 1, :] *= factor
        frame[:, i] *= factor
        frame[:, -i - 1] *= factor


def _analytic_convection(task: str, steps: int, route: dict[str, Any], mode: str) -> SimulationResult:
    size = 160
    x, y, _, _ = _mesh(size)
    frames: list[np.ndarray] = []
    for t in range(steps):
        tau = 2.0 * math.pi * t / max(steps, 1)
        rolls = 0.5 + 0.5 * np.sin(3.0 * math.pi * x + 0.8 * np.sin(tau)) * np.cos(math.pi * y)
        plume = np.exp(-((x - 0.18 * np.sin(tau)) ** 2) / 0.045) * np.exp(-((y + 0.45 - 0.22 * np.cos(tau)) ** 2) / 0.35)
        boundary = np.exp(-((y + 0.93) / 0.04) ** 2) + 0.65 * np.exp(-((y - 0.93) / 0.05) ** 2)
        temperature = 0.55 * rolls + 1.15 * plume + 0.35 * boundary
        if mode == "heat_flux":
            shown = np.abs(np.gradient(temperature)[0]) + 0.32 * boundary
        elif mode == "plume_velocity":
            shown = plume * (0.8 + 0.35 * np.sin(3.0 * math.pi * x - tau)) + 0.2 * _gradient_magnitude(rolls)
        elif mode == "rolls":
            shown = rolls
        else:
            shown = temperature
        frames.append(_normalize(shown))
    meta = _base_meta(task, route, "rayleigh_benard_renderer", "convection", size, steps, mode)
    meta.update(
        {
            "dataset": "rayleigh_benard",
            "trained_for_request": False,
            "data_source_kind": "deterministic_physics_renderer",
            "prediction_method": f"Rayleigh-Benard-like convection rolls and thermal plume renderer, mode={mode}",
            "data_source_rows": [
                _source("target_dataset", "rayleigh_benard", "The Well route for thermal convection prompts"),
                _source("local_checkpoint", "missing", str(CHECKPOINTS["rayleigh_benard"])),
                _source("selected_mode", mode, "Temperature, heat-flux, plume-velocity, or roll-cell view"),
                _source("rendered_now", "analytic convection-roll model", "Not The Well-trained until checkpoint exists"),
            ],
        }
    )
    warning = "No local rayleigh_benard checkpoint is installed. Rendering a deterministic convection model instead."
    meta["frame_stats"] = [_frame_stats(frame, i) for i, frame in enumerate(frames)]
    meta["interpretation"] = _interpretation_rows(task, meta, route, warning)
    return SimulationResult(task, _to_frames(frames), meta, warning)


def _reaction_diffusion(task: str, steps: int, route: dict[str, Any], mode: str) -> SimulationResult:
    size = 128
    rng = _rng(task, "gray-scott")
    u = np.ones((size, size), dtype=np.float32)
    v = np.zeros((size, size), dtype=np.float32)
    v[48:80, 48:80] = 0.25 + 0.5 * rng.random((32, 32), dtype=np.float32)
    u[48:80, 48:80] = 0.5
    du, dv, feed, kill = 0.16, 0.08, 0.055, 0.062
    frames: list[np.ndarray] = []
    substeps = max(1, 360 // max(steps, 1))
    for _ in range(steps):
        for _ in range(substeps):
            lap_u = np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u
            lap_v = np.roll(v, 1, 0) + np.roll(v, -1, 0) + np.roll(v, 1, 1) + np.roll(v, -1, 1) - 4 * v
            uvv = u * v * v
            u += du * lap_u - uvv + feed * (1 - u)
            v += dv * lap_v + uvv - (feed + kill) * v
        if mode == "reaction_rate":
            shown = u * v * v
        elif mode == "pattern_edges":
            shown = _gradient_magnitude(v)
        else:
            shown = v
        frames.append(_normalize(shown))
    meta = _base_meta(task, route, "gray_scott_pde", "reaction", size, steps, mode)
    meta.update(
        {
            "dataset": "gray_scott_reaction_diffusion",
            "trained_for_request": False,
            "data_source_kind": "numerical_pde_solver",
            "prediction_method": f"Gray-Scott reaction-diffusion PDE solver, mode={mode}",
            "data_source_rows": [
                _source("target_dataset", "gray_scott_reaction_diffusion", "The Well route for reaction-diffusion prompts"),
                _source("local_checkpoint", "missing", str(CHECKPOINTS["gray_scott_reaction_diffusion"])),
                _source("selected_mode", mode, "Species concentration, reaction-rate, or pattern-edge output"),
                _source("rendered_now", "live Gray-Scott PDE", "Numerical PDE frames generated live; not The Well-trained until checkpoint exists"),
            ],
        }
    )
    warning = "No local gray_scott checkpoint is installed. Rendering a live Gray-Scott PDE simulation instead."
    meta["frame_stats"] = [_frame_stats(frame, i) for i, frame in enumerate(frames)]
    meta["interpretation"] = _interpretation_rows(task, meta, route, warning)
    return SimulationResult(task, _to_frames(frames), meta, warning)


def _analytic_shear(task: str, steps: int, route: dict[str, Any], mode: str) -> SimulationResult:
    size = 160
    x, y, _, _ = _mesh(size)
    frames: list[np.ndarray] = []
    for t in range(steps):
        tau = 2.0 * math.pi * t / max(steps, 1)
        interface = y - 0.18 * np.sin(2.5 * math.pi * x - tau)
        billows = np.sin(5.0 * theta_like(x, y) + 2.0 * tau) * np.exp(-interface * interface / 0.18)
        shear = np.tanh(interface * 7.0)
        scalar = 0.55 * shear + 0.7 * billows
        if mode == "vorticity":
            shown = _gradient_magnitude(scalar) + 0.22 * np.abs(billows)
        elif mode == "mixing":
            shown = np.abs(billows) * (1.0 - np.abs(shear) * 0.45)
        elif mode == "velocity":
            shown = 0.5 + 0.5 * shear + 0.25 * billows
        else:
            shown = scalar
        frames.append(_normalize(shown))
    meta = _base_meta(task, route, "shear_flow_renderer", "shear", size, steps, mode)
    meta.update(
        {
            "dataset": "shear_flow",
            "trained_for_request": False,
            "data_source_kind": "deterministic_physics_renderer",
            "prediction_method": f"Kelvin-Helmholtz-like shear layer renderer, mode={mode}",
            "data_source_rows": [
                _source("target_dataset", "shear_flow", "The Well route for shear-flow prompts"),
                _source("local_checkpoint", "missing", str(CHECKPOINTS["shear_flow"])),
                _source("selected_mode", mode, "Scalar, vorticity, mixing, or velocity view"),
                _source("rendered_now", "analytic shear-instability model", "Not The Well-trained until checkpoint exists"),
            ],
        }
    )
    warning = "No local shear_flow checkpoint is installed. Rendering a deterministic shear-instability model instead."
    meta["frame_stats"] = [_frame_stats(frame, i) for i, frame in enumerate(frames)]
    meta["interpretation"] = _interpretation_rows(task, meta, route, warning)
    return SimulationResult(task, _to_frames(frames), meta, warning)


def _analytic_planetary(task: str, steps: int, route: dict[str, Any], mode: str) -> SimulationResult:
    ny, nx = 144, 288
    lon = np.linspace(-math.pi, math.pi, nx, dtype=np.float32)
    lat = np.linspace(-0.5 * math.pi, 0.5 * math.pi, ny, dtype=np.float32)
    lam, phi = np.meshgrid(lon, lat)
    rng = _rng(task, "planetswe")
    vortex_lat = float(rng.uniform(-0.45, 0.45))
    vortex_lon = float(rng.uniform(-1.8, 1.8))
    band_phase = float(rng.uniform(0.0, 2.0 * math.pi))
    frames: list[np.ndarray] = []
    for t in range(steps):
        tau = 2.0 * math.pi * t / max(steps, 1)
        jets = 0.5 + 0.5 * np.sin(5.5 * phi + 0.7 * np.sin(tau + band_phase))
        rossby = 0.32 * np.sin(3.0 * lam - 1.7 * tau) * np.cos(phi) ** 2
        lon_center = vortex_lon + 0.7 * np.sin(0.45 * tau)
        wrapped_lon = np.angle(np.exp(1j * (lam - lon_center)))
        storm = np.exp(-((wrapped_lon / 0.42) ** 2 + ((phi - vortex_lat) / 0.22) ** 2))
        spiral = storm * (0.75 + 0.25 * np.sin(9.0 * np.arctan2(phi - vortex_lat, wrapped_lon) - 2.4 * tau))
        height = 0.62 * jets + rossby + 1.15 * spiral
        if mode == "vorticity":
            shown = _gradient_magnitude(height) + 0.22 * np.abs(rossby)
        elif mode == "jet_stream":
            shown = jets + 0.25 * np.abs(rossby)
        elif mode == "storm_track":
            shown = spiral + 0.32 * storm
        else:
            shown = height
        frames.append(_normalize(shown))
    meta = _base_meta(task, route, "planetary_shallow_water_renderer", "planetary", f"{ny}x{nx}", steps, mode)
    meta.update(
        {
            "dataset": "planetswe",
            "trained_for_request": False,
            "data_source_kind": "deterministic_physics_renderer",
            "prediction_method": f"shallow-water-inspired planetary jet and vortex renderer, mode={mode}",
            "parameters": {
                "vortex_latitude_proxy": round(vortex_lat, 4),
                "vortex_longitude_proxy": round(vortex_lon, 4),
            },
            "data_source_rows": [
                _source("target_dataset", "planetswe", "The Well route for planetary shallow-water prompts"),
                _source("local_checkpoint", "missing", str(CHECKPOINTS["planetswe"])),
                _source("selected_mode", mode, "Height, vorticity, jet-stream, or storm-track output"),
                _source("rendered_now", "analytic shallow-water-like planetary flow model", "Not The Well-trained until checkpoint exists"),
            ],
        }
    )
    warning = "No local planetswe checkpoint is installed. Rendering a deterministic shallow-water-like planetary model instead."
    meta["frame_stats"] = [_frame_stats(frame, i) for i, frame in enumerate(frames)]
    meta["interpretation"] = _interpretation_rows(task, meta, route, warning)
    return SimulationResult(task, _to_frames(frames), meta, warning)


def _analytic_mhd(task: str, steps: int, route: dict[str, Any], mode: str) -> SimulationResult:
    size = 64
    rng = _rng(task, "mhd")
    x, y, r, theta = _mesh(size)
    # Generate some turbulent-like wave superposition
    modes_x = rng.uniform(1.0, 5.0, size=8)
    modes_y = rng.uniform(1.0, 5.0, size=8)
    phases = rng.uniform(0.0, 2 * math.pi, size=8)
    amplitudes = rng.uniform(0.1, 0.5, size=8)
    frames: list[np.ndarray] = []
    for t in range(steps):
        tau = 2.0 * math.pi * t / max(steps, 1)
        field = np.zeros_like(x)
        for k in range(8):
            field += amplitudes[k] * np.sin(modes_x[k] * x + modes_y[k] * y - 1.5 * tau + phases[k])
        # Add some non-linear coupling / turbulence look
        field = np.sin(field * 2.0 + 0.5 * np.cos(3.0 * x - tau))
        frames.append(_normalize(field))

    # Now apply the modes
    fallback_mode = _resolve_mode("MHD_64", mode)
    frames = _apply_mode_frames(frames, fallback_mode, "MHD_64")

    meta = _base_meta(task, route, "mhd_analytic", "mhd_scalar", size, steps, fallback_mode)
    meta.update(
        {
            "dataset": "MHD_64",
            "trained_for_request": False,
            "data_source_kind": "deterministic_physics_renderer",
            "prediction_method": f"analytic wave-superposition plasma turbulence proxy, mode={fallback_mode}",
            "data_source_rows": [
                _source("target_dataset", "MHD_64", "The Well route for MHD plasma prompts"),
                _source("local_checkpoint", "missing", str(config.WELL_EMULATOR_CHECKPOINT)),
                _source("selected_mode", fallback_mode, _mode_transform_note("MHD_64", fallback_mode)),
                _source("rendered_now", "analytic plasma wave model", "Not The Well-trained until checkpoint exists"),
            ],
        }
    )
    warning = "No local MHD_64 checkpoint is installed. Rendering a deterministic plasma wave model instead."
    meta["frame_stats"] = [_frame_stats(frame, i) for i, frame in enumerate(frames)]
    meta["interpretation"] = _interpretation_rows(task, meta, route, warning)
    return SimulationResult(task, _to_frames(frames), meta, warning)


def theta_like(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.arctan2(y + 0.05 * np.sin(3.0 * x), x)


def _base_meta(task: str, route: dict[str, Any], kind: str, style: str, size: int | str, steps: int, mode: str) -> dict[str, Any]:
    dataset = str(route["recommended_dataset"])
    return {
        "checkpoint_loaded": False,
        "checkpoint_path": None,
        "simulation_kind": kind,
        "visual_style": style,
        "frame_size": size,
        "steps": steps,
        "mode": mode,
        "supported_modes": list(_mode_list(dataset)),
        "mode_transform": _mode_transform_note(dataset, mode),
        "requested_domain": style,
        "dataset_hint": route["recommended_dataset"],
        "requested_environment": route["requested_environment"],
        "route_reason": route["reason"],
        "catalog_match": route["catalog_card"],
        "loaded_dataset_card": catalog_lookup(str(route["recommended_dataset"])),
    }


def _trained_source_rows(meta: dict[str, Any], dataset: str) -> list[dict[str, str]]:
    datasets = meta.get("datasets_trained")
    rows: list[dict[str, str]] = [
        _source("loaded_checkpoint_dataset", dataset, str(meta.get("checkpoint_path", config.WELL_EMULATOR_CHECKPOINT))),
        _source("source", str(meta.get("base_path", "hf://datasets/polymathic-ai/")), "The Well / Hugging Face dataset path"),
        _source("field", str(meta.get("field", "auto_selected_scalar_slice")), "Scalar 2D slice used by the frame emulator"),
    ]
    if isinstance(datasets, list):
        for row in datasets:
            if isinstance(row, dict):
                rows.append(_source("train_items", str(row.get("train_items_available", "unknown")), str(row.get("train_split", "train"))))
                rows.append(_source("valid_items", str(row.get("validation_items_available", "unknown")), str(row.get("validation_split", "valid"))))
    rows.append(_source("val_loss", str(meta.get("val_loss", "unknown")), "checkpoint validation MSE"))
    return rows


def _source(label: str, value: str, note: str) -> dict[str, str]:
    return {"label": label, "value": value, "note": note}


def _interpretation_rows(task: str, meta: dict[str, Any], route: dict[str, Any], warning: str | None) -> list[dict[str, str]]:
    catalog = route.get("catalog_card") or {}
    source_kind = str(meta.get("data_source_kind", "unknown"))
    checkpoint_status = "loaded" if meta.get("checkpoint_loaded") else "not loaded for this route"
    if source_kind == "trained_the_well_checkpoint":
        honesty = "This output is backed by the local trained The Well checkpoint named in the data table."
    elif source_kind == "numerical_pde_solver":
        honesty = "This output is generated by a live numerical PDE solver because the matching The Well checkpoint is not installed."
    else:
        honesty = "This output is generated by a deterministic physics renderer because the matching The Well checkpoint is not installed."
    return [
        {"field": "task", "value": task, "why_it_matters": "The prompt routes the simulation family and deterministic parameters."},
        {"field": "recommended_well_dataset", "value": str(route["recommended_dataset"]), "why_it_matters": str(route["reason"])},
        {"field": "checkpoint_status", "value": checkpoint_status, "why_it_matters": honesty},
        {"field": "what_you_are_seeing", "value": str(meta.get("prediction_method")), "why_it_matters": "The visual style is chosen per physics family, not reused as one heatmap for every prompt."},
        {"field": "useful_for", "value": str(catalog.get("useful_for", "")), "why_it_matters": "This is the claim the demo can safely make for the routed dataset."},
        {"field": "not_useful_for", "value": str(catalog.get("not_useful_for", "")), "why_it_matters": "This prevents overclaiming beyond the checkpoint or solver."},
        {"field": "warning", "value": warning or "none", "why_it_matters": "If no matching The Well checkpoint exists, the UI says so instead of pretending."},
    ]


def _frame_stats(frame: np.ndarray, index: int) -> dict[str, float | int]:
    dy = np.diff(frame, axis=0)
    dx = np.diff(frame, axis=1)
    gradient = 0.5 * (float(np.mean(np.abs(dx))) + float(np.mean(np.abs(dy))))
    return {
        "frame": index,
        "min": round(float(np.min(frame)), 6),
        "max": round(float(np.max(frame)), 6),
        "mean": round(float(np.mean(frame)), 6),
        "std": round(float(np.std(frame)), 6),
        "gradient_energy": round(gradient, 6),
    }


def _normalize(frame: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(frame, [0.5, 99.5])
    if hi <= lo:
        return np.zeros_like(frame, dtype=np.float32)
    return np.clip((frame - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _to_frames(frames: list[np.ndarray]) -> list[list[list[float]]]:
    return [np.round(frame, 5).tolist() for frame in frames]


