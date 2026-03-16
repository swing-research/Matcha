# Matcha: Fast Volume Alignment by Frequency-Marched Newton

> Official implementation of **"Fast Volume Alignment by Frequency-Marched Newton"**

Matcha aligns subtomograms against a reference template via continuous SO(3) optimisation using a ball harmonic decomposition and a frequency-marched Newton solver. It runs as a stand-alone CLI tool or directly as a **RELION External job**.

---

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Benchmark example](#benchmark-example)
- [Full alignment](#full-alignment)
- [RELION External job](#relion-external-job)
- [Tips](#tips)
- [Project layout](#project-layout)

---

## Requirements

|              |                                                     |
| ------------ | --------------------------------------------------- |
| Python       | ≥ 3.12                                              |
| GPU          | CUDA-capable (required for alignment and benchmark) |
| Dependencies | `pyproject.toml` / `requirements.txt`               |

---

## Installation

**From a local clone** (recommended for development):

```bash
git clone https://github.com/swing-research/Matcha matcha
cd matcha
pip install -e .
```

**Directly from the repository:**

```bash
pip install "git+https://github.com/swing-research/Matcha.git"
```

After installation, `matcha` and `matcha-example` are available on your `PATH`.

**Without installation** — use the bundled launchers, which set `PYTHONPATH` automatically:

```bash
./bin/matcha         --config configs/config.yaml --align
./bin/matcha-example --config configs/config_example.yaml
```

---

## Benchmark example

Benchmarks the orientation-search back-end on synthetically rotated and noise-corrupted copies of a template volume.

### Step 1 — download the template

The example config uses a ribosome map from [EMPIAR-10045](https://www.ebi.ac.uk/empiar/EMPIAR-10045/)
(Bharat & Scheres, _Nature Protocols_ 2016):

```bash
wget -P data/ https://ftp.ebi.ac.uk/empiar/world_availability/10045/data/ribosomes/AnticipatedResults/Refine3D/run2_class001.mrc
```

The config already points to `data/run2_class001.mrc` — no further edits needed.

### Step 2 — run

```bash
# without installation:
./bin/matcha-example --config configs/config_example.yaml

# installed CLI:
matcha-example --config configs/config_example.yaml

# directly:
python src/run.py --example --config configs/config_example.yaml
```

Optionally write metrics to JSON:

```bash
./bin/matcha-example --config configs/config_example.yaml --metrics_out results.json
```

### Expected output

```
Results of 1000 volumes at snr 0.0dB:
Mean distance (deg):              X.XX°
Median distance (deg):            X.XX°
90th percentile distance (deg):   X.XX°
search_orientations timing: mean=XX.XX ms, std=XX.XX ms, total=XX.XX s for 1000 volumes, ...
```

---

## Full alignment

Edit `configs/config.yaml` to point to your data:

| Field                | Description                               |
| -------------------- | ----------------------------------------- |
| `path_templates`     | List of two half-map `.mrc` files         |
| `path_template_mask` | Optional mask `.mrc` (leave `""` to skip) |
| `path_subtomograms`  | Directory containing subtomogram files    |
| `run_data_path`      | RELION STAR file with particle metadata   |
| `gpu_ids`            | List of GPU indices to use                |
| `voxel_size`         | Voxel size in Å                           |
| `box_size`           | Box size in pixels                        |

Then run:

```bash
# without installation:
./bin/matcha --config configs/config.yaml --align

# installed CLI:
matcha --config configs/config.yaml --align

# directly:
python src/run.py --config configs/config.yaml --align
```

---

## RELION External job

Matcha can be used directly as a RELION External executable:

```bash
matcha --o External/jobXXX/       \
       --in_parts  <particles.star>      \
       --in_3dref  <half1_or_half2.mrc>  \
       --in_mask   <mask.mrc>            \
       --j         <threads>
```

| Behaviour       | Details                                                                                                 |
| --------------- | ------------------------------------------------------------------------------------------------------- |
| I/O override    | `--o`, `--in_parts`, `--in_3dref`, `--in_mask`, `--j`, GPU flags override the config                    |
| Half-map pair   | If `--in_3dref` contains `half1`/`half2`, the counterpart is required at the same location              |
| Single map      | If neither tag is present, the same map is used for both halves                                         |
| Particle split  | Particles are split randomly into two halves from the input STAR                                        |
| Output          | Particles STAR written to `<--o>/matcha_particles.star`                                                 |
| Lifecycle files | `RELION_JOB_EXIT_SUCCESS` / `RELION_JOB_EXIT_FAILURE` and `RELION_OUTPUT_NODES.star` written to `<--o>` |

---

## Tips

**Numba cache path** — on HPC systems with restricted home directories, set:

```bash
export NUMBA_CACHE_DIR=/tmp/numba_cache
```

**Orientation-search back-end** — select via `orientation_search.method` in `config_example.yaml`:

| Method     | Description                                                                      |
| ---------- | -------------------------------------------------------------------------------- |
| `"matcha"` | Frequency-marched Newton solver — fast, recommended                              |
| `"sofft"`  | SO(3) FFT exhaustive grid search — slower or less accurate, useful as a baseline |

---

## Project layout

```
matcha/
├── bin/                       # Shell launchers (no installation needed)
│   ├── matcha
│   └── matcha-example
├── configs/
│   ├── config.yaml            # Config template for full alignment
│   └── config_example.yaml   # Config for the benchmark example
├── data/                      # Pre-computed FLE lookup tables
├── src/
│   ├── run.py                 # Unified CLI entry point
│   ├── example.py             # Benchmark example
│   ├── align_subtomograms.py
│   ├── core/                  # Matcha, SOFFT, CrossCorrelationMatcher
│   └── utils/                 # I/O, rotation ops, volume ops
└── pyproject.toml
```
