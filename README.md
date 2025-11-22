# openmm-spherical-boundaries

Utilities for trimming solvated systems, building elastic network shells, and
running short OpenMM simulations using spherical boundary conditions.

## Installation

```bash
pip install .
```

This installs the `openmm_spherical_boundaries` package and the accompanying
`openmm-spherical-boundaries` CLI.

## Command-line usage

```bash
# Trim a solvent box, build elastic network bonds, and write system.xml
openmm-spherical-boundaries prepare-md input.pdb --output-pdb multiresPrep.pdb

# Build a spherical droplet directly (no input PDB required)
openmm-spherical-boundaries prepare-droplet --radius 2.5 --output-pdb droplet.pdb

# Build the triangular boundary shell used in the thesis examples
openmm-spherical-boundaries prepare-triangular spce.pdb --output-pdb multiresTriag.pdb

# Serialized run using an existing PDB/system pair
openmm-spherical-boundaries run multiresPrep.pdb system.xml --steps 2000
```

Run `openmm-spherical-boundaries --help` for the full list of options. Defaults
match the parameters in the original scripts so workflows can be migrated before
tuning anything.

The droplet generator currently supports the original AMBER99SB + SPC/E force
field pairing; passing other force-field XML files will raise a clear error until
they can be validated.

### Standalone droplet runner (scripts/run_droplet.py)

If you want a single scriptable droplet run (with warmup/minimization support)
without the CLI, use `scripts/run_droplet.py`:

```bash
python scripts/run_droplet.py --radius 3.0 --equil-ns 1 --prod-ns 5 \
  --warmup-start-k 10 --warmup-duration-ns 0.1 --minimize \
  --equil-traj equil.dcd --prod-traj prod.dcd \
  --equil-log equil.log --prod-log prod.log \
  --warmup-traj warmup.dcd --warmup-log warmup.log
```

Key features: Langevin (300 K, 1 ps^-1, 1 fs), optional energy minimization,
separate warmup phase (default 10 K for 0.1 ns to target temp), split trajectory
and energy reporters for warmup/equil/prod, and DCD-only trajectories.

### Preparing multiple jobs (setup_simulations)

Use `openmm_spherical_boundaries.setup_simulations.setup_droplet_jobs` to emit
ready-to-run commands and per-job folders:

```python
from openmm_spherical_boundaries.setup_simulations import setup_droplet_jobs

commands = setup_droplet_jobs(
    simulation_folder="runs",
    variants=[{"radius": r} for r in (2.0, 2.5, 3.0)],
    base_params={"equil_ns": 1.0, "prod_ns": 5.0},
)
for cmd in commands:
    print(cmd)
```

Behavior: creates a `script/run_droplet.py` copy under `simulation_folder`,
creates one subfolder per variant (named from parameters), writes `params.json`,
and returns shell commands that run in a subshell so your CWD is restored.
Existing folders are reused when `overwrite=True` (default) rather than deleted.

## Python API

```python
from openmm_spherical_boundaries import (
    prepare_water_droplet,
    prepare_md_system,
    prepare_reference_system,
    prepare_triangular_boundary,
    run_serialized_simulation,
)

prepare_water_droplet(radius=2.5, output_pdb="droplet.pdb")
prepare_md_system("input.pdb", output_pdb="multiresPrep.pdb")
prepare_triangular_boundary("input.pdb")
prepare_reference_system("spce_9822.pdb")
run_serialized_simulation(
    "multiresPrep.pdb",
    "system.xml",
    steps=500,
    traj_path="trajectory.dcd",
    trajectory_format="dcd",
)
```

Each preparation helper writes a PDB/XML pair and returns diagnostic data that
can be inspected or plotted for validation.
### Quick droplet example

```python
from openmm_spherical_boundaries import (
    prepare_water_droplet,
    run_serialized_simulation,
)

# Build a 2.5 nm water droplet (writes droplet.pdb/droplet.xml)
prepare_water_droplet(radius=2.5, output_pdb="droplet.pdb")

# Run a short MD trajectory, saving coordinates to trajectory.dcd
run_serialized_simulation("droplet.pdb", "droplet.xml", steps=5000)
```

`prepare_water_droplet` seeds a cube with SPC/E water via `Modeller.addSolvent`,
trims molecules whose oxygen lies outside the requested radius, and writes the
trimmed system plus a serialized `OpenMM` `System`. The subsequent call to
`run_serialized_simulation` deserializes that system, initializes an integrator,
and runs 5000 steps while recording a DCD trajectory. Adjust `steps`, `traj_path`,
or `trajectory_format` to suit your workflow.

When building droplets the outer layer of oxygen atoms is turned into an elastic
network so the droplet maintains its spherical boundary. By default a 0.3 nm
shell, 0.6 nm cutoff, and 1000 kJ/mol/nm^2 springs are used; tweak these through
`shell_thickness`, `shell_cutoff`, and `force_constant` (or the analogous CLI
flags) if you need softer or harder confinement.
