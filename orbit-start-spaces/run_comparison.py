"""Compare TNT's orbit-sampler initial conditions against DYNAMITE's
compiled `orbitstart` binary, across four NGC6278-based viewing geometries.

Reads `dynamite_output/<geometry>/begin[box].dat` -- DYNAMITE's own
already-generated tube/box orbit initial conditions from a real
`orbitstart` run, committed in this repo rather than regenerated -- see
this directory's own README.md for the pinned DYNAMITE commit and how
those files were produced. Each geometry's full physical setup (MGE,
viewing angles, BH, NFW halo, grid) is read directly from
`input/<geometry>/parameters_pot.in` -- the fully resolved input
DYNAMITE's `orbitstart` itself consumed -- so nothing here is
independently re-derived from the higher-level `input/scenario_*.yaml`
DYNAMITE scenario files (kept alongside for reference/provenance only).

Run from this directory, in a TNT development environment:
    python run_comparison.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from unxt import Quantity, unitsystem

from tnt.mge import LightMGE
from tnt.orbit_library.stationary_grid import StationaryGridOrbitSampler
from tnt.orbit_library.xz_grid_from_boundary import XZGridFromBoundaryOrbitSampler
from tnt.potential import Potential
from tnt.quantity_conversions import angular_to_physical

HERE = Path(__file__).parent
MGE_FIXTURE = HERE / "mge_lum.ecsv"
INPUT_ROOT = HERE / "input"
DYNAMITE_OUTPUT_ROOT = HERE / "dynamite_output"
UNITS = unitsystem("kpc", "Myr", "Msun", "rad")

# DYNAMITE's own `nEner, nI2, nI3` naming/roles, matched letter-for-letter by
# TNT's orbit samplers (see KNOWLEDGE.md) -- no translation needed here.
GEOMETRIES = ["oblate", "mild_triaxial", "strong_triaxial", "near_prolate"]


@dataclass
class DynamiteParameters:
    """One geometry's fully-resolved `infil/parameters_pot.in` contents."""

    distance_mpc: float
    theta_deg: float
    phi_deg: float
    psi_deg: float
    ml: float
    bh_mass_msun: float
    bh_softening_arcsec: float
    nE: int
    logrmin_arcsec: float
    logrmax_arcsec: float
    nI2: int
    nI3: int
    concentration: float
    dm_fraction: float
    hubble_km_s_mpc: float


def _parse_parameters_pot(path: Path) -> DynamiteParameters:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    i = 0
    n_gauss = int(lines[i])
    i += 1 + n_gauss  # MGE rows themselves come from `mge_lum.ecsv` instead
    distance_mpc = float(lines[i])
    i += 1
    theta_deg, phi_deg, psi_deg = (float(v) for v in lines[i].split())
    i += 1
    ml = float(lines[i])
    i += 1
    bh_mass_msun = float(lines[i])
    i += 1
    bh_softening_arcsec = float(lines[i])
    i += 1
    nE, logrmin_arcsec, logrmax_arcsec = lines[i].split()
    nE, logrmin_arcsec, logrmax_arcsec = int(nE), float(logrmin_arcsec), float(
        logrmax_arcsec
    )
    i += 1
    nI2 = int(lines[i])
    i += 1
    nI3 = int(lines[i])
    i += 1
    i += 4  # dithering, quad_nr, quad_nth, quad_nph -- not needed here
    _dm_profile_type, _n_dmparam = (int(v) for v in lines[i].split())
    i += 1
    concentration, dm_fraction = (float(v) for v in lines[i].split())
    i += 1
    hubble_km_s_mpc = float(lines[i]) * 1e6  # file stores H * 1e-6
    return DynamiteParameters(
        distance_mpc=distance_mpc,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        psi_deg=psi_deg,
        ml=ml,
        bh_mass_msun=bh_mass_msun,
        bh_softening_arcsec=bh_softening_arcsec,
        nE=nE,
        logrmin_arcsec=logrmin_arcsec,
        logrmax_arcsec=logrmax_arcsec,
        nI2=nI2,
        nI3=nI3,
        concentration=concentration,
        dm_fraction=dm_fraction,
        hubble_km_s_mpc=hubble_km_s_mpc,
    )


def _total_stellar_luminosity(mge: LightMGE) -> float:
    """Total luminosity of a projected MGE: `sum(2 * pi * I * sigma**2 * q)`."""
    value = (
        2
        * jnp.pi
        * mge.I.ustrip("Lsun/pc2")
        * mge.sigma.ustrip("pc") ** 2
        * mge.q.ustrip("")
    )
    return float(jnp.sum(value))


def build_potential(params: DynamiteParameters):
    distance = Quantity(params.distance_mpc, "Mpc").to("kpc")
    mge_angular = LightMGE.read(MGE_FIXTURE, Quantity(0.0, "deg"))
    mge_physical = mge_angular.angular_to_physical(distance)

    total_luminosity = _total_stellar_luminosity(mge_physical)
    total_stellar_mass = params.ml * total_luminosity
    m200 = params.dm_fraction * total_stellar_mass

    bh_scale = angular_to_physical(
        Quantity(params.bh_softening_arcsec, "arcsec"), distance
    )

    resolved = Potential.resolve(
        {
            "bh": {"type": "PlummerPotential"},
            "dh": {"type": "NFWPotential", "parameterization": "concentration_m200"},
            "stars": {"type": "TriaxialLightMGEPotential", "mge": "mge_lum"},
        },
        {"mge_lum": mge_physical},
    )
    proposal = {
        "bh": {
            "m_tot": Quantity(params.bh_mass_msun, "Msun"),
            "r_s": bh_scale,
        },
        "dh": {
            "c": Quantity(params.concentration, ""),
            "M_200": Quantity(m200, "Msun"),
        },
        "stars": {
            "theta": Quantity(params.theta_deg, "deg").to("rad"),
            "phi": Quantity(params.phi_deg, "deg").to("rad"),
            "psi": Quantity(params.psi_deg, "deg").to("rad"),
            "ml": Quantity(params.ml, "Msun/Lsun"),
        },
    }
    potential, valid = Potential.build_with_validity(
        resolved,
        proposal,
        {"H": Quantity(params.hubble_km_s_mpc, "km/(s*Mpc)")},
    )
    if not bool(valid):
        raise RuntimeError("TNT potential construction reported invalid.")
    return potential.to_galax(UNITS), distance


def _read_dynamite_ics(path: Path, nE: int, nI2: int, nI3: int) -> np.ndarray:
    """`(nE, nI2, nI3, 6)` phase-space array from a `begin[box].dat` file.

    Columns, matching `orbitstart_f.f90`'s own write statement: `i, j, k,
    x, y, z, vx, vy, vz, rcirc, tcirc, vcirc, noreg` -- positions in km,
    velocities in km/s.
    """
    raw = np.loadtxt(path, skiprows=1)
    ics = np.zeros((nE, nI2, nI3, 6))
    for row in raw:
        i, j, k = int(row[0]) - 1, int(row[1]) - 1, int(row[2]) - 1
        ics[i, j, k] = row[3:9]
    position_km = ics[..., :3]
    velocity_km_s = ics[..., 3:]
    position_kpc = Quantity(position_km, "km").ustrip("kpc")
    velocity_kpc_myr = Quantity(velocity_km_s, "km/s").ustrip("kpc/Myr")
    return np.concatenate([position_kpc, velocity_kpc_myr], axis=-1)


def _relative_error(tnt: np.ndarray, dynamite: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.abs(dynamite), 1e-300)
    return np.abs(tnt - dynamite) / scale


def compare_geometry(name: str) -> dict:
    params = _parse_parameters_pot(INPUT_ROOT / name / "parameters_pot.in")
    potential, distance = build_potential(params)

    rmin = angular_to_physical(
        Quantity(10.0**params.logrmin_arcsec, "arcsec"), distance
    )
    rmax = angular_to_physical(
        Quantity(10.0**params.logrmax_arcsec, "arcsec"), distance
    )

    box_sampler = StationaryGridOrbitSampler(
        rmin=rmin, rmax=rmax, nE=params.nE, nI2=params.nI2, nI3=params.nI3
    )
    tube_sampler = XZGridFromBoundaryOrbitSampler(
        rmin=rmin, rmax=rmax, nE=params.nE, nI2=params.nI2, nI3=params.nI3
    )

    box_tnt = np.asarray(box_sampler.generate_ics(potential)).reshape(
        params.nE, params.nI2, params.nI3, 6
    )
    tube_tnt = np.asarray(tube_sampler.generate_ics(potential)).reshape(
        params.nE, params.nI2, params.nI3, 6
    )

    output_dir = DYNAMITE_OUTPUT_ROOT / name
    box_dynamite = _read_dynamite_ics(
        output_dir / "beginbox.dat", params.nE, params.nI2, params.nI3
    )
    tube_dynamite = _read_dynamite_ics(
        output_dir / "begin.dat", params.nE, params.nI2, params.nI3
    )

    # Box orbits: compare only position (TNT's own velocity is exactly zero
    # by construction; DYNAMITE's recorded velocity at a box orbit's launch
    # point is nominally zero too, up to its own floating-point noise).
    box_error = _relative_error(box_tnt[..., :3], box_dynamite[..., :3])
    # Tube orbits: compare position and the single nonzero velocity
    # component (v_y); vx, vz are exactly zero by construction on both sides.
    tube_error = _relative_error(
        np.concatenate([tube_tnt[..., :3], tube_tnt[..., 4:5]], axis=-1),
        np.concatenate([tube_dynamite[..., :3], tube_dynamite[..., 4:5]], axis=-1),
    )

    return {
        "geometry": name,
        "box_median_relative_error": float(np.median(box_error)),
        "box_max_relative_error": float(np.max(box_error)),
        "tube_median_relative_error": float(np.median(tube_error)),
        "tube_max_relative_error": float(np.max(tube_error)),
        "box_tnt": box_tnt,
        "box_dynamite": box_dynamite,
        "tube_tnt": tube_tnt,
        "tube_dynamite": tube_dynamite,
    }


def main() -> None:
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    summary = []
    for name in GEOMETRIES:
        result = compare_geometry(name)
        for key in ("box_tnt", "box_dynamite", "tube_tnt", "tube_dynamite"):
            np.save(results_dir / f"{name}_{key}.npy", result.pop(key))
        summary.append(result)
        print(
            f"{name}: box median/max rel. error "
            f"{result['box_median_relative_error']:.2e}/"
            f"{result['box_max_relative_error']:.2e}, "
            f"tube median/max rel. error "
            f"{result['tube_median_relative_error']:.2e}/"
            f"{result['tube_max_relative_error']:.2e}"
        )
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
