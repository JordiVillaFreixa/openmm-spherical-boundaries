"""Planning scaffolding for droplet validation metrics."""

from __future__ import annotations

import logging
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence, Tuple

os.environ.setdefault("VMDPLUGIN_DISABLE_MESSAGES", "1")

try:  # MDTraj is optional until the analysis stack is finalized.
    import mdtraj as md
except ImportError:  # pragma: no cover - runtime guard for optional dependency
    md = None

try:  # Optional scientific-python stack for droplet analysis.
    import numpy as np
except ImportError:  # pragma: no cover - runtime guard for optional dependency
    np = None  # type: ignore[assignment]

try:
    import pandas as pd
except ImportError:  # pragma: no cover - runtime guard for optional dependency
    pd = None  # type: ignore[assignment]

try:  # Optional plotting dependency.
    from matplotlib import pyplot as plt
except ImportError:  # pragma: no cover - runtime guard for optional dependency
    plt = None  # type: ignore[assignment]
except RuntimeError:  # pragma: no cover - e.g., headless backend issues
    plt = None  # type: ignore[assignment]


DEFAULT_WATER_NUMBER_DENSITY = 33.45  # molecules / nm^3 (approximate bulk water)
LOGGER = logging.getLogger(__name__)


def _coerce_paths(paths: Iterable[str | Path]) -> list[Path]:
    """Normalize a collection of paths into resolved Path objects."""

    return [Path(p).expanduser().resolve() for p in paths]


def _concatenate_trajectories(trajectories: Sequence["md.Trajectory"]) -> "md.Trajectory":
    """Join multiple MDTraj trajectories into a single continuous object."""

    if not trajectories:
        raise ValueError("At least one trajectory is required to compute metrics.")
    if len(trajectories) == 1:
        return trajectories[0]
    if md is None:  # pragma: no cover - guard against optional dependency
        raise RuntimeError(
            "MDTraj is required to concatenate trajectory segments for analysis."
        )
    return md.join(list(trajectories))


@dataclass
class ReplicaDistanceData:
    """Flattened water-distance data extracted from a replica trajectory."""

    label: str
    distances: "np.ndarray"
    n_frames: int
    n_waters: int

    @property
    def r_max(self) -> float:
        if self.distances.size == 0:
            return 0.0
        return float(self.distances.max())


def _compute_center_of_geometry(x: "np.ndarray") -> "np.ndarray":
    """Center of geometry of coordinates for a single frame (n_atoms, 3)."""

    return x.mean(axis=0)


def _make_bin_edges(
    r_max: float,
    n_bins: int,
    mode: Literal["equal_dr", "equal_volume"] = "equal_dr",
) -> "np.ndarray":
    """
    Compute radial bin edges for [0, r_max].

    equal_dr      -> constant Δr
    equal_volume  -> constant shell volume (thus constant theoretical N_wat)
    """

    if mode == "equal_dr":
        return np.linspace(0.0, r_max, n_bins + 1)
    if mode == "equal_volume":
        k = np.arange(0, n_bins + 1, dtype=float)
        return r_max * (k / n_bins) ** (1.0 / 3.0)
    raise ValueError(f"Unknown mode {mode!r}, use 'equal_dr' or 'equal_volume'.")


def _extract_water_distances(
    traj: "md.Trajectory",
    *,
    water_selection: str,
    discard_fraction: float,
) -> tuple["np.ndarray", int, int]:
    """Return flattened water distances and metadata after equilibration discard."""

    if md is None or np is None:  # pragma: no cover - runtime guard
        raise RuntimeError("MDTraj and NumPy are required to compute water distances.")
    if not 0.0 <= discard_fraction < 1.0:
        raise ValueError("discard_fraction must be in the range [0, 1).")

    n_frames = traj.n_frames
    if n_frames == 0:
        raise ValueError("Trajectory must contain at least one frame.")

    start_frame = int(discard_fraction * n_frames)
    traj_prod = traj[start_frame:]
    if traj_prod.n_frames == 0:
        raise ValueError("No frames remain after applying discard_fraction.")

    water_idx = traj_prod.top.select(water_selection)
    n_waters = len(water_idx)
    if n_waters == 0:
        raise ValueError(
            f"Selection '{water_selection}' did not yield any atoms in the topology."
        )

    coords = traj_prod.xyz
    centers = coords.mean(axis=1)
    water_coords = coords[:, water_idx, :]
    deltas = water_coords - centers[:, None, :]
    dist = np.linalg.norm(deltas, axis=-1)
    dist_flat = dist.ravel()
    if dist_flat.size == 0:
        raise ValueError("Unable to compute distances for the supplied trajectory.")

    return dist_flat, traj_prod.n_frames, n_waters


def _resolve_bin_count(
    r_max: float,
    n_bins: int | None,
    bin_width: float | None,
) -> int:
    """Determine how many bins are needed for a target radius."""

    if bin_width is not None:
        if bin_width <= 0:
            raise ValueError("bin_width must be positive when provided.")
        return max(1, int(np.ceil(r_max / bin_width)))
    if n_bins is not None:
        if n_bins <= 0:
            raise ValueError("n_bins must be a positive integer when provided.")
        return n_bins
    return 30


def _build_density_dataframe_from_edges(
    dist_flat: "np.ndarray",
    *,
    n_frames_prod: int,
    n_waters: int,
    edges: "np.ndarray",
) -> Tuple["pd.DataFrame", float]:
    """Create the radial density DataFrame for supplied histogram edges."""

    counts, _ = np.histogram(dist_flat, bins=edges)
    r_inner = edges[:-1]
    r_outer = edges[1:]
    r_center = 0.5 * (r_inner + r_outer)
    shell_volumes = (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)
    if np.any(shell_volumes <= 0):
        raise ValueError("Encountered non-positive shell volume while binning data.")

    N_avg = counts / n_frames_prod
    rho = N_avg / shell_volumes

    r_max = edges[-1]
    if r_max <= 0:
        raise ValueError("Invalid radial extent computed for density histogram.")
    V_bulk = (4.0 / 3.0) * np.pi * (r_max**3)
    bulk_density = n_waters / V_bulk

    df = pd.DataFrame(
        {
            "r_inner_nm": r_inner,
            "r_outer_nm": r_outer,
            "r_center_nm": r_center,
            "shell_volume_nm3": shell_volumes,
            "N_avg": N_avg,
            "rho_mol_per_nm3": rho,
        }
    )

    return df, bulk_density


def _average_density_from_replicas(
    replicas: Sequence[ReplicaDistanceData],
    *,
    r_max: float,
    n_bins: int | None,
    bin_width: float | None,
    mode: Literal["equal_dr", "equal_volume"],
) -> Tuple["pd.DataFrame", float]:
    """Average radial densities across replicas using shared histogram edges."""

    if np is None or pd is None:  # pragma: no cover - runtime guard
        raise RuntimeError("NumPy and pandas are required for replica averaging.")
    if mode == "equal_volume" and bin_width is not None:
        raise ValueError("bin_width is only supported with 'equal_dr' mode.")
    if not replicas:
        raise ValueError("At least one replica is required for averaging.")

    n_bins_eff = _resolve_bin_count(r_max, n_bins, bin_width)
    edges = _make_bin_edges(r_max=r_max, n_bins=n_bins_eff, mode=mode)
    r_inner = edges[:-1]
    r_outer = edges[1:]
    r_center = 0.5 * (r_inner + r_outer)
    shell_volumes = (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)

    rho_values: list["np.ndarray"] = []
    N_values: list["np.ndarray"] = []
    n_waters_set = {rep.n_waters for rep in replicas}
    if len(n_waters_set) != 1:
        raise ValueError(
            "Replica trajectories report different numbers of waters; cannot average densities."
        )

    for rep in replicas:
        counts, _ = np.histogram(rep.distances, bins=edges)
        N_avg = counts / rep.n_frames
        rho = N_avg / shell_volumes
        N_values.append(N_avg)
        rho_values.append(rho)

    N_avg_mean = np.mean(N_values, axis=0)
    rho_mean = np.mean(rho_values, axis=0)
    if len(N_values) > 1:
        N_std = np.std(N_values, axis=0, ddof=1)
        rho_std = np.std(rho_values, axis=0, ddof=1)
    else:
        N_std = np.zeros_like(N_avg_mean)
        rho_std = np.zeros_like(rho_mean)
    V_bulk = (4.0 / 3.0) * np.pi * (r_max**3)
    bulk_density = next(iter(n_waters_set)) / V_bulk

    df = pd.DataFrame(
        {
            "r_inner_nm": r_inner,
            "r_outer_nm": r_outer,
            "r_center_nm": r_center,
            "shell_volume_nm3": shell_volumes,
            "N_avg": N_avg_mean,
            "N_std": N_std,
            "rho_mol_per_nm3": rho_mean,
            "rho_std": rho_std,
        }
    )

    return df, bulk_density


def compute_radial_density_general(
    traj: "md.Trajectory",
    water_selection: str = "water and name O",
    r_max: float | None = None,
    n_bins: int | None = 30,
    *,
    bin_width: float | None = None,
    mode: Literal["equal_dr", "equal_volume"] = "equal_dr",
    discard_fraction: float = 0.2,
) -> Tuple["pd.DataFrame", float]:
    """
    Compute radial number density of water oxygens around droplet center.

    Parameters
    ----------
    traj : md.Trajectory
        Trajectory (assumed water droplet).
    water_selection : str
        MDTraj selection string for water oxygens.
    r_max : float, optional
        Maximum radius (nm) to consider. If None, uses max observed distance.
    n_bins : int, optional
        Number of radial shells (ignored if bin_width is provided).
    bin_width : float, optional
        Desired shell thickness in nm (only valid for equal-Δr bins).
    mode : {'equal_dr', 'equal_volume'}
        'equal_dr'      -> shells with constant radial thickness Δr
        'equal_volume'  -> shells with constant volume (same theoretical N_wat)
    discard_fraction : float
        Fraction of frames (from the beginning) discarded as equilibration.

    Returns
    -------
    df : pd.DataFrame
        Columns:
        ['r_inner_nm', 'r_outer_nm', 'r_center_nm',
         'shell_volume_nm3', 'N_avg', 'rho_mol_per_nm3']
    bulk_density : float
        Average bulk density (molecules / nm^3) inside r <= r_max.
    """

    if md is None or np is None or pd is None:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "MDTraj, NumPy, and pandas are required for radial density analysis."
        )
    if mode == "equal_volume" and bin_width is not None:
        raise ValueError("bin_width is only supported with 'equal_dr' mode.")
    dist_flat, n_frames_prod, n_waters = _extract_water_distances(
        traj,
        water_selection=water_selection,
        discard_fraction=discard_fraction,
    )

    if r_max is None:
        r_max_eff = dist_flat.max()
    else:
        r_max_eff = float(r_max)
    if r_max_eff <= 0:
        raise ValueError("r_max must be positive to define radial shells.")

    n_bins_eff = _resolve_bin_count(r_max_eff, n_bins, bin_width)
    edges = _make_bin_edges(r_max=r_max_eff, n_bins=n_bins_eff, mode=mode)

    return _build_density_dataframe_from_edges(
        dist_flat,
        n_frames_prod=n_frames_prod,
        n_waters=n_waters,
        edges=edges,
    )


def plot_radial_density_profiles(
    results: dict[str, dict[str, Any]],
    *,
    ax: "plt.Axes" | None = None,
    title: str | None = None,
    legend: bool = True,
    show_reference: bool = True,
    reference_label: str | None = None,
    expected_density: float | None = DEFAULT_WATER_NUMBER_DENSITY,
    expected_label: str = "expected",
    show_std: bool = True,
    std_alpha: float = 0.2,
) -> "plt.Axes":
    """Plot radial number-density profiles from ``radial_density_profile`` results."""

    if plt is None:
        raise RuntimeError("matplotlib is not available; install the 'plotting' extra to enable plots.")
    if pd is None:
        raise RuntimeError("pandas is required to supply profile data for plotting.")

    axis = ax or plt.gca()
    for label, payload in results.items():
        if reference_label and label == reference_label and not show_reference:
            continue
        profile = payload["profile"]
        (line,) = axis.plot(
            profile["r_center_nm"],
            profile["rho_mol_per_nm3"],
            label=label,
        )
        if show_std and "rho_std" in profile.columns and not profile["rho_std"].isna().all():
            std = profile["rho_std"].to_numpy()
            if np.any(std):
                axis.fill_between(
                    profile["r_center_nm"],
                    profile["rho_mol_per_nm3"] - std,
                    profile["rho_mol_per_nm3"] + std,
                    color=line.get_color(),
                    alpha=std_alpha,
                    linewidth=0,
                )

    if expected_density is not None:
        xmin = min(
            profile["r_center_nm"].min()
            for label, payload in results.items()
            for profile in [payload["profile"]]
        )
        xmax = max(
            profile["r_center_nm"].max()
            for label, payload in results.items()
            for profile in [payload["profile"]]
        )
        axis.hlines(
            expected_density,
            xmin,
            xmax,
            colors="k",
            linestyles="dashed",
            label=expected_label if legend else None,
        )

    axis.set_xlabel("Radius (nm)")
    axis.set_ylabel(r"Number density (molecules / nm$^3$)")
    if title:
        axis.set_title(title)
    if legend:
        axis.legend()
    return axis


@dataclass
class SimulationDataset:
    """Description of a single droplet or reference trajectory bundle."""

    label: str
    topology_path: Path
    trajectory_paths: list[Path]
    metadata: dict[str, Any] = field(default_factory=dict)


class DropletValidationMetrics:
    """Manage droplet simulations and expose stubbed validation metrics.

    The goal is to keep an analysis-friendly view of multiple simulations, where each
    simulation can have an arbitrary number of trajectory files (to support segmented
    production runs) and additional replica trajectories that share the same parameters.
    Each metric method currently raises NotImplementedError, but their docstrings explain
    the calculations we will add once the MDTraj-based I/O utilities are wired in.

    Example:
        >>> metrics = DropletValidationMetrics()
        >>> metrics.register_directory(\"runs_test\")  # auto-discovers variant/replica folders
    """

    def __init__(self) -> None:
        self.simulations: dict[str, SimulationDataset] = {}
        self.replica_groups: dict[str, dict[str, SimulationDataset]] = {}

    # -------------------------------------------------------------------------
    # Simulation registration helpers
    # -------------------------------------------------------------------------
    def add_simulation(
        self,
        label: str,
        topology_path: str | Path,
        trajectory_paths: Sequence[str | Path],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a simulation (e.g., a unique parameter set) for later analysis."""

        dataset = SimulationDataset(
            label=label,
            topology_path=Path(topology_path).expanduser().resolve(),
            trajectory_paths=_coerce_paths(trajectory_paths),
            metadata=dict(metadata or {}),
        )
        self.simulations[label] = dataset

    def add_replica(
        self,
        group_label: str,
        replica_label: str,
        topology_path: str | Path,
        trajectory_paths: Sequence[str | Path],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a replica under a logical group label (e.g., replica set)."""

        dataset = SimulationDataset(
            label=replica_label,
            topology_path=Path(topology_path).expanduser().resolve(),
            trajectory_paths=_coerce_paths(trajectory_paths),
            metadata=dict(metadata or {}),
        )
        self.replica_groups.setdefault(group_label, {})[replica_label] = dataset

    def register_directory(
        self,
        root: str | Path,
        *,
        include_variants: Iterable[str] | None = None,
    ) -> None:
        """Bulk-register simulations discovered under a setup_droplet_jobs directory."""

        from .discovery import discover_simulations

        layout = discover_simulations(root, include_variants=include_variants)
        for variant, replicas in layout.items():
            for replica_label, dataset in replicas.items():
                label = f"{variant}/{replica_label}"
                self.add_simulation(
                    label=label,
                    topology_path=dataset.topology_path,
                    trajectory_paths=dataset.trajectory_paths,
                    metadata=dataset.metadata,
                )

    # -------------------------------------------------------------------------
    # Loading utilities
    # -------------------------------------------------------------------------
    def load_simulation(self, label: str, **kwargs: Any) -> list[Any]:
        """Load the trajectories for a named simulation via MDTraj."""

        dataset = self.simulations.get(label)
        if dataset is None:
            raise KeyError(f"Simulation '{label}' is not registered.")
        return self._load_dataset(dataset, **kwargs)

    def load_replica(self, group_label: str, replica_label: str, **kwargs: Any) -> list[Any]:
        """Load the trajectories for a specific replica."""

        group = self.replica_groups.get(group_label, {})
        dataset = group.get(replica_label)
        if dataset is None:
            raise KeyError(f"Replica '{replica_label}' in group '{group_label}' is not registered.")
        return self._load_dataset(dataset, **kwargs)

    def _load_dataset(self, dataset: SimulationDataset, **kwargs: Any) -> list[Any]:
        """Internal loader that defers to MDTraj once the dependency is available."""

        if md is None:
            raise RuntimeError(
                "MDTraj is not installed. Please install it to enable trajectory loading."
            )
        trajectories = []
        for traj_path in dataset.trajectory_paths:
            trajectories.append(
                md.load(str(traj_path), top=str(dataset.topology_path), **kwargs)
            )
        return trajectories

    def _resolve_simulation_labels(
        self,
        label: str,
        replicas: Sequence[str] | None = None,
    ) -> list[str]:
        """Return fully-qualified simulation labels for a variant or replica name."""

        if label in self.simulations:
            if replicas:
                raise ValueError("replicas can only be provided for variant-level labels.")
            return [label]

        prefix = f"{label}/"
        matches = sorted(name for name in self.simulations if name.startswith(prefix))
        if not matches:
            raise KeyError(f"Simulation '{label}' is not registered.")

        if replicas is not None:
            filtered: list[str] = []
            for replica_label in replicas:
                key = f"{label}/{replica_label}"
                if key not in self.simulations:
                    raise KeyError(f"Replica '{replica_label}' not found under variant '{label}'.")
                filtered.append(key)
            return filtered

        return matches

    def _profile_from_label(
        self,
        label: str,
        *,
        water_selection: str,
        r_max: float | None,
        n_bins: int | None,
        bin_width: float | None,
        mode: Literal["equal_dr", "equal_volume"],
        discard_fraction: float,
        runaway_threshold_override: float | None,
    ) -> dict[str, Any]:
        trajectories = self.load_simulation(label)
        traj = _concatenate_trajectories(trajectories)
        profile, bulk_density = compute_radial_density_general(
            traj,
            water_selection=water_selection,
            r_max=r_max,
            n_bins=n_bins,
            bin_width=bin_width,
            mode=mode,
            discard_fraction=discard_fraction,
        )
        threshold = self._get_runaway_threshold(label, runaway_threshold_override)
        self._warn_runaway_if_needed(label, profile["r_outer_nm"].max(), threshold)
        return {"profile": profile, "bulk_density": bulk_density}

    def _collect_distance_data_for_labels(
        self,
        labels: Sequence[str],
        *,
        water_selection: str,
        discard_fraction: float,
        runaway_threshold_override: float | None,
    ) -> list[ReplicaDistanceData]:
        """Load simulations and collect flattened water distances for averaging."""

        data: list[ReplicaDistanceData] = []
        for label in labels:
            trajectories = self.load_simulation(label)
            traj = _concatenate_trajectories(trajectories)
            dist_flat, n_frames_prod, n_waters = _extract_water_distances(
                traj,
                water_selection=water_selection,
                discard_fraction=discard_fraction,
            )
            data.append(
                ReplicaDistanceData(
                    label=label,
                    distances=dist_flat,
                    n_frames=n_frames_prod,
                    n_waters=n_waters,
                )
            )
            threshold = self._get_runaway_threshold(label, runaway_threshold_override)
            self._warn_runaway_if_needed(label, dist_flat.max(), threshold)
        return data

    def _warn_runaway_if_needed(
        self,
        label: str,
        r_max: float,
        threshold: float | None,
    ) -> None:
        if threshold is None:
            return
        if r_max > threshold:
            LOGGER.warning(
                "Simulation %s observed water distance %.2f nm (threshold %.2f nm)",
                label,
                r_max,
                threshold,
            )

    def _get_expected_radius(self, label: str) -> float | None:
        dataset = self.simulations.get(label)
        if dataset is None:
            return None
        radius = dataset.metadata.get("radius")
        if radius is None:
            return None
        try:
            return float(radius)
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            LOGGER.debug("Unable to parse radius %r for %s", radius, label)
            return None

    def _resolve_metadata_radius(self, labels: Sequence[str]) -> float | None:
        radii = [self._get_expected_radius(label) for label in labels]
        radii = [r for r in radii if r is not None]
        if not radii:
            return None
        unique = {round(r, 6) for r in radii}
        if len(unique) > 1:
            LOGGER.warning(
                "Replica radii differ for %s; using first value %.3f nm",
                labels[0].split("/")[0],
                radii[0],
            )
        return radii[0]

    def _get_runaway_threshold(self, label: str, override: float | None) -> float | None:
        if override is not None:
            return override
        dataset = self.simulations.get(label)
        if dataset is None:
            return None
        radius = dataset.metadata.get("radius")
        if radius is None:
            return None
        try:
            return float(radius)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            LOGGER.debug("Could not coerce radius %r for label %s", radius, label)
            return None

    def _profiles_for_label(
        self,
        label: str,
        *,
        replicas: Sequence[str] | None,
        aggregation: Literal["concatenate", "average"] | None,
        water_selection: str,
        r_max: float | None,
        n_bins: int | None,
        bin_width: float | None,
        mode: Literal["equal_dr", "equal_volume"],
        discard_fraction: float,
        runaway_threshold_override: float | None,
    ) -> dict[str, dict[str, Any]]:
        resolved = self._resolve_simulation_labels(label, replicas=replicas)
        effective_r_max = r_max
        if effective_r_max is None and mode == "equal_volume":
            metadata_radius = self._resolve_metadata_radius(resolved)
            if metadata_radius is not None:
                effective_r_max = metadata_radius
        if len(resolved) == 1 or aggregation is None:
            return {
                sim_label: self._profile_from_label(
                    sim_label,
                    water_selection=water_selection,
                    r_max=effective_r_max,
                    n_bins=n_bins,
                    bin_width=bin_width,
                    mode=mode,
                    discard_fraction=discard_fraction,
                    runaway_threshold_override=runaway_threshold_override,
                )
                for sim_label in resolved
            }

        if aggregation == "concatenate":
            replica_data = self._collect_distance_data_for_labels(
                resolved,
                water_selection=water_selection,
                discard_fraction=discard_fraction,
                runaway_threshold_override=runaway_threshold_override,
            )
            if effective_r_max is not None:
                r_max_eff = float(effective_r_max)
            else:
                r_max_eff = max(rep.r_max for rep in replica_data)
            if r_max_eff <= 0:
                raise ValueError("r_max must be positive to define radial shells.")
            dist_flat = (
                np.concatenate([rep.distances for rep in replica_data])
                if replica_data
                else np.array([])
            )
            if dist_flat.size == 0:
                raise ValueError("Unable to compute distances for the supplied trajectories.")
            n_frames_total = sum(rep.n_frames for rep in replica_data)
            n_waters_set = {rep.n_waters for rep in replica_data}
            if len(n_waters_set) != 1:
                raise ValueError(
                    "Replica trajectories report different numbers of waters; cannot concatenate."
                )
            n_bins_eff = _resolve_bin_count(r_max_eff, n_bins, bin_width)
            edges = _make_bin_edges(r_max=r_max_eff, n_bins=n_bins_eff, mode=mode)
            profile, bulk_density = _build_density_dataframe_from_edges(
                dist_flat,
                n_frames_prod=n_frames_total,
                n_waters=n_waters_set.pop(),
                edges=edges,
            )
            return {label: {"profile": profile, "bulk_density": bulk_density}}

        if aggregation == "average":
            replica_data = self._collect_distance_data_for_labels(
                resolved,
                water_selection=water_selection,
                discard_fraction=discard_fraction,
                runaway_threshold_override=runaway_threshold_override,
            )
            if effective_r_max is not None:
                r_max_eff = float(effective_r_max)
            else:
                r_max_eff = max(rep.r_max for rep in replica_data)
            if r_max_eff <= 0:
                raise ValueError("r_max must be positive to define radial shells.")
            profile, bulk_density = _average_density_from_replicas(
                replica_data,
                r_max=r_max_eff,
                n_bins=n_bins,
                bin_width=bin_width,
                mode=mode,
            )
            return {label: {"profile": profile, "bulk_density": bulk_density}}

        raise ValueError(
            "aggregation must be one of {'concatenate', 'average'} or None for per-replica profiles."
        )

    # -------------------------------------------------------------------------
    # Planned metrics (docstring first, implementation later)
    # -------------------------------------------------------------------------
    def plan_outline(self) -> dict[str, str]:
        """Return a high-level description of each validation metric."""

        return {
            "radial_density_profile": (
                "Compute a center-of-mass-aligned radial histogram of water number density, "
                "compare to a PBC reference profile, and quantify deviations via RMS error."
            ),
            "pressure_temperature_profile": (
                "Project the virial/kinetic terms onto spherical shells to ensure the droplet "
                "maintains near-uniform thermodynamic conditions relative to the reference box."
            ),
            "boundary_stability": (
                "Measure mean bond length and fluctuation of the boundary springs to ensure "
                "the elastic net remains near its target geometry."
            ),
            "water_retention": (
                "Track the number of waters that leave the droplet surface over time to detect "
                "evaporation or instability events."
            ),
            "interface_fluctuations": (
                "Calculate the variance of the droplet radius as a function of polar angle to "
                "understand interfacial roughness introduced by the boundary method."
            ),
            "hydrogen_bond_network": (
                "Compare hydrogen-bond coordination numbers inside the droplet versus a "
                "periodic reference to ensure structural properties are preserved."
            ),
        }

    def radial_density_profile(
        self,
        label: str,
        *,
        reference_label: str | None = None,
        bin_width: float = 0.02,
        n_bins: int | None = 30,
        replicas: Sequence[str] | None = None,
        replica_aggregation: Literal["concatenate", "average"] | None = None,
        reference_replicas: Sequence[str] | None = None,
        reference_replica_aggregation: Literal["concatenate", "average"] | None = None,
        water_selection: str = "water and name O",
        r_max: float | None = None,
        mode: Literal["equal_dr", "equal_volume"] = "equal_dr",
        discard_fraction: float = 0.2,
        runaway_radius_threshold: float | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Compute radial density profiles for a simulation and optional reference.

        Returns a mapping of simulation label (replica-specific or aggregated) to a
        ``{"profile": DataFrame, "bulk_density": float}`` payload. When ``label`` or
        ``reference_label`` refer to a variant (e.g., ``radius-2p0nm``), pass
        ``replica_aggregation`` / ``reference_replica_aggregation`` to combine the
        underlying replicas via concatenation or equal-weight averaging. Leaving the
        aggregation argument as ``None`` returns individual profiles per replica.

        Set ``runaway_radius_threshold`` (nm) to log a warning whenever any replica
        exhibits water distances beyond that radius, which helps flag runaway or
        evaporated waters before aggregating the profiles. When left as ``None``,
        the radius recorded in each replica's ``params.json`` metadata is used.
        """

        if np is None or pd is None:
            raise RuntimeError(
                "NumPy and pandas are required to compute radial density profiles."
            )

        results = self._profiles_for_label(
            label,
            replicas=replicas,
            aggregation=replica_aggregation,
            water_selection=water_selection,
            r_max=r_max,
            n_bins=n_bins,
            bin_width=bin_width,
            mode=mode,
            discard_fraction=discard_fraction,
            runaway_threshold_override=runaway_radius_threshold,
        )

        if reference_label is not None:
            ref_agg = reference_replica_aggregation if reference_replica_aggregation is not None else replica_aggregation
            results.update(
                self._profiles_for_label(
                    reference_label,
                    replicas=reference_replicas,
                    aggregation=ref_agg,
                    water_selection=water_selection,
                    r_max=r_max,
                    n_bins=n_bins,
                    bin_width=bin_width,
                    mode=mode,
                    discard_fraction=discard_fraction,
                    runaway_threshold_override=runaway_radius_threshold,
                )
            )

        return results

    def plot_radial_density(
        self,
        label: str,
        *,
        reference_label: str | None = None,
        bin_width: float = 0.02,
        n_bins: int | None = 30,
        replicas: Sequence[str] | None = None,
        replica_aggregation: Literal["concatenate", "average"] | None = None,
        reference_replicas: Sequence[str] | None = None,
        reference_replica_aggregation: Literal["concatenate", "average"] | None = None,
        water_selection: str = "water and name O",
        r_max: float | None = None,
        mode: Literal["equal_dr", "equal_volume"] = "equal_dr",
        discard_fraction: float = 0.2,
        runaway_radius_threshold: float | None = None,
        ax: "plt.Axes" | None = None,
        title: str | None = None,
        legend: bool = True,
        show_reference: bool = True,
        expected_density: float | None = DEFAULT_WATER_NUMBER_DENSITY,
        expected_label: str = "expected",
        show_std: bool = True,
        std_alpha: float = 0.2,
    ) -> "plt.Axes":
        """Convenience wrapper that computes and plots radial density profiles."""

        results = self.radial_density_profile(
            label,
            reference_label=reference_label,
            bin_width=bin_width,
            n_bins=n_bins,
            replicas=replicas,
            replica_aggregation=replica_aggregation,
            reference_replicas=reference_replicas,
            reference_replica_aggregation=reference_replica_aggregation,
            water_selection=water_selection,
            r_max=r_max,
            mode=mode,
            discard_fraction=discard_fraction,
            runaway_radius_threshold=runaway_radius_threshold,
        )
        return plot_radial_density_profiles(
            results,
            ax=ax,
            title=title,
            legend=legend,
            show_reference=show_reference,
            reference_label=reference_label,
            expected_density=expected_density,
            expected_label=expected_label,
            show_std=show_std,
            std_alpha=std_alpha,
        )

    def pressure_temperature_profile(
        self,
        label: str,
        *,
        shell_thickness: float = 0.1,
    ) -> None:
        """Plan: compute spherical-shell temperature/pressure to detect core-shell gradients."""

        raise NotImplementedError("Requires virial logging support.")

    def boundary_stability(self, label: str) -> None:
        """Plan: monitor stretch statistics for the outer boundary springs/molecules."""

        raise NotImplementedError("Needs bonded-force logging before implementation.")

    def water_retention(self, label: str, *, cutoff: float = 0.2) -> None:
        """Plan: detect evaporation by counting waters that cross a configurable radius."""

        raise NotImplementedError("Requires trajectory post-processing utilities.")

    def interface_fluctuations(self, label: str) -> None:
        """Plan: decompose surface perturbations in spherical harmonics to gauge roughness."""

        raise NotImplementedError("Pending decision on math tooling for harmonic analysis.")

    def hydrogen_bond_network(
        self,
        label: str,
        *,
        reference_label: str | None = None,
    ) -> None:
        """Plan: compare droplet hydrogen-bond coordination against a periodic reference."""

        raise NotImplementedError("Will be implemented alongside the structure metrics.")
