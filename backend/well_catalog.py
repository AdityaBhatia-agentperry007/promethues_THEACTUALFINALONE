from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


SOURCE_HOME = "https://polymathic-ai.org/the_well/"
SOURCE_OVERVIEW = "https://polymathic-ai.org/the_well/datasets_overview/"
SOURCE_HF = "https://huggingface.co/collections/polymathic-ai/the-well"


@dataclass(frozen=True)
class WellDatasetCard:
    name: str
    family: str
    domain: str
    coordinate_system: str
    resolution: str
    n_steps: str
    n_traj: str
    size_gb: str
    fields: str
    useful_for: str
    not_useful_for: str
    source_url: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


DATASET_CATALOG: tuple[WellDatasetCard, ...] = (
    WellDatasetCard(
        name="MHD_64",
        family="MHD",
        domain="magnetohydrodynamic compressible turbulence",
        coordinate_system="Cartesian 3D",
        resolution="64^3",
        n_steps="100",
        n_traj="100",
        size_gb="72",
        fields="density scalar, velocity vector, magnetic field vector",
        useful_for="solar-wind, interstellar-medium, galaxy-formation-like MHD turbulence and wave/shock behavior",
        not_useful_for="black-hole GR, radiation transport, reactor validation, direct telescope/NASA imagery",
        source_url="https://polymathic-ai.org/the_well/datasets/MHD_64/",
    ),
    WellDatasetCard(
        name="MHD_256",
        family="MHD",
        domain="higher-resolution magnetohydrodynamic compressible turbulence",
        coordinate_system="Cartesian 3D",
        resolution="256^3",
        n_steps="100",
        n_traj="100",
        size_gb="4580",
        fields="density scalar, velocity vector, magnetic field vector",
        useful_for="same MHD physics as MHD_64 at much higher spatial fidelity",
        not_useful_for="quick demo training on small GPUs, black-hole GR, full radiation transport",
        source_url="https://polymathic-ai.org/the_well/datasets/MHD_256/",
    ),
    WellDatasetCard(
        name="post_neutron_star_merger",
        family="GRMHD",
        domain="post-merger accretion disk around compact remnant / black-hole-related environment",
        coordinate_system="log-spherical 3D",
        resolution="192 x 128 x 66",
        n_steps="181",
        n_traj="8",
        size_gb="110.1",
        fields="density, internal energy, electron fraction, temperature, entropy, velocity, magnetic field, spacetime metric",
        useful_for="black-hole accretion-disk-like prompts, compact-object merger aftermath, nucleosynthesis-oriented demos",
        not_useful_for="isolated event-horizon ray tracing, real telescope imagery, generic MHD_64 claims",
        source_url="https://polymathic-ai.org/the_well/datasets/post_neutron_star_merger/",
    ),
    WellDatasetCard(
        name="supernova_explosion_64",
        family="supernova_explosion",
        domain="supernova remnant in turbulent interstellar medium",
        coordinate_system="Cartesian 3D",
        resolution="64^3",
        n_steps="59",
        n_traj="1000",
        size_gb="268",
        fields="hydrodynamic explosion fields",
        useful_for="blast waves, expanding shells, turbulent interstellar medium demos",
        not_useful_for="black-hole accretion, MHD magnetic-field claims unless trained fields include them",
        source_url="https://polymathic-ai.org/the_well/datasets/supernova_explosion_64/",
    ),
    WellDatasetCard(
        name="supernova_explosion_128",
        family="supernova_explosion",
        domain="higher-resolution supernova remnant in turbulent interstellar medium",
        coordinate_system="Cartesian 3D",
        resolution="128^3",
        n_steps="59",
        n_traj="1000",
        size_gb="754",
        fields="hydrodynamic explosion fields",
        useful_for="higher-detail blast-wave and shell-mixing demos",
        not_useful_for="fast commodity-GPU training, black-hole accretion",
        source_url="https://polymathic-ai.org/the_well/datasets/supernova_explosion_128/",
    ),
    WellDatasetCard(
        name="turbulence_gravity_cooling",
        family="astrophysical fluid",
        domain="self-gravity/cooling turbulence",
        coordinate_system="Cartesian 3D",
        resolution="64 x 64 x 64",
        n_steps="50",
        n_traj="2700",
        size_gb="829",
        fields="astrophysical turbulence fields",
        useful_for="cooling, gravity-coupled turbulence, star-formation-adjacent visual demos",
        not_useful_for="black-hole GR or validated fusion reactor behavior",
        source_url="https://polymathic-ai.org/the_well/datasets/turbulence_gravity_cooling/",
    ),
    WellDatasetCard(
        name="turbulent_radiative_layer_2D",
        family="radiative layer",
        domain="2D turbulent radiative layer",
        coordinate_system="Cartesian 2D",
        resolution="128 x 384",
        n_steps="101",
        n_traj="90",
        size_gb="6.9",
        fields="radiation-hydrodynamic layer fields",
        useful_for="radiative layer instabilities and compact 2D training runs",
        not_useful_for="3D compact-object or black-hole claims",
        source_url="https://polymathic-ai.org/the_well/datasets/turbulent_radiative_layer_2D/",
    ),
    WellDatasetCard(
        name="turbulent_radiative_layer_3D",
        family="radiative layer",
        domain="3D turbulent radiative layer",
        coordinate_system="Cartesian 3D",
        resolution="128 x 128 x 256",
        n_steps="101",
        n_traj="90",
        size_gb="745",
        fields="radiation-hydrodynamic layer fields",
        useful_for="3D radiation/turbulence demos",
        not_useful_for="quick local training or black-hole GR",
        source_url="https://polymathic-ai.org/the_well/datasets/turbulent_radiative_layer_3D/",
    ),
    WellDatasetCard(
        name="rayleigh_taylor_instability",
        family="instability",
        domain="3D Rayleigh-Taylor instability",
        coordinate_system="Cartesian 3D",
        resolution="128 x 128 x 128",
        n_steps="120",
        n_traj="45",
        size_gb="256",
        fields="fluid instability fields",
        useful_for="buoyancy-driven mixing and interface-instability visuals",
        not_useful_for="space/black-hole claims unless presented as generic fluid instability",
        source_url="https://polymathic-ai.org/the_well/datasets/rayleigh_taylor_instability/",
    ),
    WellDatasetCard(
        name="rayleigh_benard",
        family="convection",
        domain="Rayleigh-Benard convection",
        coordinate_system="Cartesian 2D",
        resolution="512 x 128",
        n_steps="200",
        n_traj="1750",
        size_gb="358",
        fields="thermal convection fields",
        useful_for="convection rolls, thermal plumes, transport-instability demos",
        not_useful_for="MHD or black-hole-specific claims",
        source_url="https://polymathic-ai.org/the_well/datasets/rayleigh_benard/",
    ),
    WellDatasetCard(
        name="shear_flow",
        family="fluid",
        domain="2D shear-flow instability",
        coordinate_system="Cartesian 2D",
        resolution="128 x 256",
        n_steps="200",
        n_traj="1120",
        size_gb="115",
        fields="fluid velocity/scalar fields",
        useful_for="Kelvin-Helmholtz-like rolls, shear mixing, wave-like instabilities",
        not_useful_for="astrophysical black-hole or radiation claims",
        source_url="https://polymathic-ai.org/the_well/datasets/shear_flow/",
    ),
    WellDatasetCard(
        name="active_matter",
        family="biophysical",
        domain="active-matter continuum dynamics",
        coordinate_system="Cartesian 2D",
        resolution="256 x 256",
        n_steps="81",
        n_traj="360",
        size_gb="51.3",
        fields="active-matter flow/order fields",
        useful_for="swirling biological/soft-matter pattern formation",
        not_useful_for="space plasma or black-hole claims",
        source_url="https://polymathic-ai.org/the_well/datasets/active_matter/",
    ),
    WellDatasetCard(
        name="gray_scott_reaction_diffusion",
        family="reaction diffusion",
        domain="Gray-Scott reaction-diffusion",
        coordinate_system="Cartesian 2D",
        resolution="128 x 128",
        n_steps="1001",
        n_traj="1200",
        size_gb="154",
        fields="two-species reaction-diffusion scalar fields",
        useful_for="chemical pattern formation, long-horizon frame prediction demos",
        not_useful_for="fluid, plasma, or black-hole claims",
        source_url="https://polymathic-ai.org/the_well/datasets/gray_scott_reaction_diffusion/",
    ),
    WellDatasetCard(
        name="planetswe",
        family="geophysical",
        domain="planetary shallow-water equations",
        coordinate_system="angular",
        resolution="256 x 512",
        n_steps="1008",
        n_traj="120",
        size_gb="186",
        fields="spherical shallow-water fields",
        useful_for="planetary waves, vortices, atmosphere/ocean-like demos",
        not_useful_for="MHD, black-hole accretion, reactor behavior",
        source_url="https://polymathic-ai.org/the_well/datasets/planetswe/",
    ),
    WellDatasetCard(
        name="acoustic_scattering_discontinuous",
        family="acoustic_scattering",
        domain="2D acoustic scattering with discontinuities",
        coordinate_system="Cartesian 2D",
        resolution="256 x 256",
        n_steps="100",
        n_traj="8000",
        size_gb="157",
        fields="acoustic pressure/wave fields",
        useful_for="wave propagation, reflections, scattering demos",
        not_useful_for="plasma, radiation GRMHD, black-hole claims",
        source_url="https://polymathic-ai.org/the_well/datasets/acoustic_scattering_discontinuous/",
    ),
    WellDatasetCard(
        name="acoustic_scattering_inclusions",
        family="acoustic_scattering",
        domain="2D acoustic scattering with inclusions",
        coordinate_system="Cartesian 2D",
        resolution="256 x 256",
        n_steps="100",
        n_traj="8000",
        size_gb="283",
        fields="acoustic pressure/wave fields",
        useful_for="wave scattering through material inclusions",
        not_useful_for="space plasma or black-hole claims",
        source_url="https://polymathic-ai.org/the_well/datasets/acoustic_scattering_inclusions/",
    ),
    WellDatasetCard(
        name="acoustic_scattering_maze",
        family="acoustic_scattering",
        domain="2D acoustic scattering in maze-like geometry",
        coordinate_system="Cartesian 2D",
        resolution="256 x 256",
        n_steps="100",
        n_traj="8000",
        size_gb="311",
        fields="acoustic pressure/wave fields",
        useful_for="maze reflections, wave-front propagation, boundary interaction demos",
        not_useful_for="astrophysical or plasma claims",
        source_url="https://polymathic-ai.org/the_well/datasets/acoustic_scattering_maze/",
    ),
    WellDatasetCard(
        name="helmholtz_staircase",
        family="wave",
        domain="Helmholtz wave propagation through staircase geometry",
        coordinate_system="Cartesian 2D",
        resolution="1024 x 256",
        n_steps="50",
        n_traj="512",
        size_gb="52",
        fields="wave fields",
        useful_for="high-resolution wave geometry demos",
        not_useful_for="MHD turbulence or compact-object simulations",
        source_url="https://polymathic-ai.org/the_well/datasets/helmholtz_staircase/",
    ),
    WellDatasetCard(
        name="convective_envelope_rsg",
        family="stellar convection",
        domain="red-supergiant convective envelope",
        coordinate_system="spherical 3D",
        resolution="256 x 128 x 256",
        n_steps="100",
        n_traj="29",
        size_gb="570",
        fields="stellar convection fields",
        useful_for="stellar-envelope turbulence and convection demos",
        not_useful_for="black-hole disk or laboratory reactor claims",
        source_url="https://polymathic-ai.org/the_well/datasets/convective_envelope_rsg/",
    ),
    WellDatasetCard(
        name="euler_multi_quadrants_openBC",
        family="Euler",
        domain="2D Euler multi-quadrant problem with open boundaries",
        coordinate_system="Cartesian 2D",
        resolution="512 x 512",
        n_steps="100",
        n_traj="10000",
        size_gb="5170 family",
        fields="compressible Euler fluid fields",
        useful_for="shock/interface fluid dynamics and large trajectory counts",
        not_useful_for="magnetic or GRMHD claims",
        source_url="https://polymathic-ai.org/the_well/datasets/euler_multi_quadrants_openBC/",
    ),
    WellDatasetCard(
        name="euler_multi_quadrants_periodicBC",
        family="Euler",
        domain="2D Euler multi-quadrant problem with periodic boundaries",
        coordinate_system="Cartesian 2D",
        resolution="512 x 512",
        n_steps="100",
        n_traj="10000",
        size_gb="5170 family",
        fields="compressible Euler fluid fields",
        useful_for="periodic shock/interface demos",
        not_useful_for="magnetic or GRMHD claims",
        source_url="https://polymathic-ai.org/the_well/datasets/euler_multi_quadrants_periodicBC/",
    ),
    WellDatasetCard(
        name="viscoelastic_instability",
        family="viscoelastic",
        domain="2D viscoelastic instability",
        coordinate_system="Cartesian 2D",
        resolution="512 x 512",
        n_steps="variable",
        n_traj="260",
        size_gb="66",
        fields="viscoelastic flow fields",
        useful_for="polymer/elastic flow instability visual demos",
        not_useful_for="space plasma or black-hole claims",
        source_url="https://polymathic-ai.org/the_well/datasets/viscoelastic_instability/",
    ),
)


def catalog_rows() -> list[dict[str, str]]:
    return [card.to_dict() for card in DATASET_CATALOG]


def catalog_lookup(name: str | None) -> dict[str, str] | None:
    if not name:
        return None
    normalized = name.lower()
    for card in DATASET_CATALOG:
        if card.name.lower() == normalized:
            return card.to_dict()
    return None


def route_task_to_dataset(task: str) -> dict[str, Any]:
    text = task.lower()
    if any(word in text for word in ("black hole", "blackhole", "event horizon", "accretion", "neutron", "merger", "kilonova")):
        dataset = "post_neutron_star_merger"
        reason = "compact-object and accretion-disk language maps to The Well GRMHD post-merger dataset"
        environment = "reactor.inc/deep_space_compact_object"
    elif any(word in text for word in ("supernova", "blast", "explosion", "remnant")):
        dataset = "supernova_explosion_64"
        reason = "blast-wave language maps to The Well supernova remnant dataset"
        environment = "reactor.inc/stellar_blast_wave"
    elif any(word in text for word in ("sound", "acoustic", "scatter", "maze", "wavefront")):
        dataset = "acoustic_scattering_maze"
        reason = "acoustic wave language maps to The Well acoustic scattering family"
        environment = "reactor.inc/acoustic_wave_lab"
    elif any(word in text for word in ("convection", "thermal", "plume", "rayleigh")):
        dataset = "rayleigh_benard"
        reason = "thermal plume language maps to The Well Rayleigh-Benard convection dataset"
        environment = "reactor.inc/thermal_transport"
    elif any(word in text for word in ("reaction", "diffusion", "chemical", "pattern")):
        dataset = "gray_scott_reaction_diffusion"
        reason = "reaction/diffusion language maps to The Well Gray-Scott dataset"
        environment = "reactor.inc/reaction_diffusion"
    elif any(word in text for word in ("planet", "atmosphere", "ocean", "vortex")):
        dataset = "planetswe"
        reason = "planetary flow language maps to The Well shallow-water dataset"
        environment = "reactor.inc/planetary_flow"
    elif any(word in text for word in ("shear", "kelvin", "mixing layer")):
        dataset = "shear_flow"
        reason = "shear/mixing language maps to The Well shear-flow dataset"
        environment = "reactor.inc/shear_layer"
    else:
        dataset = "MHD_64"
        reason = "default plasma, magnetic, turbulence, and fusion-like prompts map to the installed MHD_64 checkpoint"
        environment = "reactor.inc/mhd_plasma"
    return {
        "requested_environment": environment,
        "recommended_dataset": dataset,
        "reason": reason,
        "catalog_card": catalog_lookup(dataset),
    }


def collection_summary() -> dict[str, Any]:
    return {
        "name": "The Well",
        "description": "15TB collection of physics simulation datasets for spatiotemporal ML surrogate modeling",
        "source_home": SOURCE_HOME,
        "source_overview": SOURCE_OVERVIEW,
        "source_huggingface": SOURCE_HF,
        "catalog_rows_in_ui": len(DATASET_CATALOG),
        "official_overview": {
            "total_collection_size": "15TB",
            "published_dataset_count": "16 dataset families with resolution/problem variants",
            "storage_format": "HDF5 data plus dataset YAML metadata",
            "split_policy": "random train/test/validation split at 0.8/0.1/0.1 trajectory level",
            "array_shape": "(n_traj, n_steps, coord1, coord2, optional coord3)",
        },
    }
