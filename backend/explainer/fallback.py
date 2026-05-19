from __future__ import annotations

from backend import config

FALLBACK_MAP = {
    "MHD_64": "mhd_intro.mp4",
    "MHD_256": "mhd_intro.mp4",
    "post_neutron_star_merger": "blackhole_intro.mp4",
    "supernova_explosion_64": "supernova_intro.mp4",
    "supernova_explosion_128": "supernova_intro.mp4",
    "rayleigh_benard": "convection_intro.mp4",
    "convective_envelope_rsg": "convection_intro.mp4",
    "acoustic_scattering_maze": "acoustic_intro.mp4",
    "acoustic_scattering_discontinuous": "acoustic_intro.mp4",
    "acoustic_scattering_inclusions": "acoustic_intro.mp4",
    "helmholtz_staircase": "acoustic_intro.mp4",
    "planetswe": "planet_intro.mp4",
    "gray_scott_reaction_diffusion": "mhd_intro.mp4",
    "shear_flow": "mhd_intro.mp4",
}


def nearest_fallback(route: dict[str, object] | None) -> str:
    dataset = str((route or {}).get("recommended_dataset") or "MHD_64")
    name = FALLBACK_MAP.get(dataset, "mhd_intro.mp4")
    if config.GCS_BUCKET:
        return f"https://storage.googleapis.com/{config.GCS_BUCKET}/fallback/{name}"
    return f"/fallback/{name}"