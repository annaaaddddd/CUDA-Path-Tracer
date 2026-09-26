# Performance Data

This file records how every number in the README was measured and what the raw readings were. Interpretation lives in the README.

## Setup

- GPU: NVIDIA GeForce RTX 3090 Ti, 24 GB, driver 616.56
- CPU: AMD Ryzen 9 5950X, 128 GB, Windows 11 Home
- Resolution 800x800 in every run
- Timing comes from `PERF_LOG` in `src/pathtrace.cu`: host-side chrono around `pathtrace()`, printed as an average over each block of 100 iterations
- Paths-alive counts are printed once, from iteration 1

Every script runs from the repo root, writes a CSV to `analysis/data/` and the raw console output to `analysis/logs/` (not committed). Scripts that edit `src/pathtrace.cu` or a scene file restore it on exit, including on Ctrl-C, but they do not rebuild afterwards, so the binary left in `build/` is the last configuration that ran. Rebuild before rendering.

| Script | Varies | Rebuilds | Output |
|---|---|---|---|
| `run-perf.sh` | compaction on/off, open/closed box | yes | `paths.csv`, `timing.csv` |
| `run-aa.sh` | sorting, anti-aliasing | yes | `aa-timing.csv` |
| `run-aabb.sh` | mesh AABB culling, triangle count | yes | `aabb-timing.csv` |
| `run-refraction.sh` | sphere material, open/closed box | no | `refraction-timing.csv` |
| `run-depth.sh` | trace depth, open/closed box | no | `depth-timing.csv` |

## run-perf.sh: compaction sweep

```bash
bash analysis/scripts/run-perf.sh
```

- Scenes: `cornell.json` (open), `cornell_closed.json` (closed)
- 5000 iterations, depth 8, sorting off, anti-aliasing off
- Run on 2026-09-22; 50 readings per row

| Scene | Compaction | avg ms/iteration | range | fps |
|---|---|---|---|---|
| open | on | 20.40 | 19.65 - 21.19 | 49 |
| open | off | 17.09 | 16.73 - 17.58 | 59 |
| closed | on | 35.97 | 35.10 - 37.16 | 28 |
| closed | off | 19.74 | 19.21 - 21.05 | 51 |

Paths alive after each bounce, iteration 1:

| Bounce | open, compaction on | closed, compaction on | compaction off, both |
|---|---|---|---|
| 1 | 523,164 | 613,916 | 640,000 |
| 2 | 362,150 | 601,084 | 640,000 |
| 3 | 279,680 | 590,284 | 640,000 |
| 4 | 222,899 | 580,254 | 640,000 |
| 5 | 180,366 | 570,499 | 640,000 |
| 6 | 146,600 | 560,637 | 640,000 |
| 7 | 119,912 | 551,106 | 640,000 |

`analysis/scripts/plot-paths.py` turns `paths.csv` into `img/paths_alive_per_bounce.png`:

```bash
python analysis/scripts/plot-paths.py
```

Note: an earlier 500-iteration pass of the same four runs agreed to within a few tenths of a millisecond, so ms/iteration does not depend on the iteration count. 5000 is kept because it also produces a usable render.

## run-aa.sh: sorting and anti-aliasing

```bash
bash analysis/scripts/run-aa.sh
```

- Scene: `cornell.json`, 5000 iterations, depth 8, compaction on in every row
- Run on 2026-09-22; 50 readings per row
- The baseline row is the open, compaction-on row of the sweep above

| Config | sort | AA | avg ms/iteration | range | fps |
|---|---|---|---|---|---|
| sweep baseline | off | off | 20.40 | 19.65 - 21.19 | 49 |
| aa-only | off | on | 20.11 | 19.50 - 21.31 | 50 |
| all-on | on | on | 36.03 | 35.27 - 37.07 | 28 |

The aa-only and all-on rows are the single-variable pair for sorting.

Note: a first all-on measurement read 40.5 - 45.9 ms. Two later runs of the same configuration gave 36.0 ms, so the first was discarded.

## run-aabb.sh: mesh bounding-box culling

```bash
bash analysis/scripts/run-aabb.sh
```

- Scene: `suzanne.json` with `models/suzanne_4k.gltf` and `models/suzanne_16k.gltf`
- 100 iterations, depth 8, compaction, sorting and anti-aliasing on
- Toggle: `MESH_AABB_CULL`
- Run on 2026-09-25; one reading per row

| Model | Triangles | Culling | ms/iteration | fps |
|---|---|---|---|---|
| suzanne_4k | 3,936 | on | 177.49 | 5.6 |
| suzanne_4k | 3,936 | off | 226.58 | 4.4 |
| suzanne_16k | 15,744 | on | 622.71 | 1.6 |
| suzanne_16k | 15,744 | off | 843.14 | 1.2 |

Note: a longer manual run of the 16k model with culling off drifted from 843 to 901 ms over 600 iterations, so the script reads the first block only.

## run-refraction.sh: mirror versus glass

```bash
bash analysis/scripts/run-refraction.sh
```

- Scenes: `cornell.json` and `glass_open.json`, `cornell_closed.json` and `glass_closed.json`. Each pair differs only in the sphere's material
- 300 iterations, depth 8, all toggles on
- Run on 2026-09-26; three readings per row

| Scene | Material | Box | readings, ms/iteration | avg |
|---|---|---|---|---|
| cornell | mirror | open | 36.80, 37.22, 36.87 | 36.96 |
| glass_open | glass | open | 37.41, 37.37, 37.37 | 37.38 |
| cornell_closed | mirror | closed | 64.75, 64.70, 64.99 | 64.81 |
| glass_closed | glass | closed | 64.69, 64.46, 67.25 | 65.47 |

## run-depth.sh: trace depth

```bash
bash analysis/scripts/run-depth.sh
```

- Scenes: `glass_open.json`, `glass_closed.json`, with `DEPTH` overridden
- 300 iterations, all toggles on
- Run on 2026-09-26; three readings per row

| Scene | Depth | readings, ms/iteration | avg |
|---|---|---|---|
| glass_open | 8 | 37.06, 37.03, 37.12 | 37.07 |
| glass_open | 32 | 54.78, 58.61, 53.68 | 55.69 |
| glass_closed | 8 | 64.96, 64.72, 65.38 | 65.02 |
| glass_closed | 32 | 218.20, 218.39, 220.85 | 219.15 |

## Image comparisons

`analysis/scripts/compare-images.py` reports per-pixel differences between two renders:

```bash
python analysis/scripts/compare-images.py img/cornell_diffuse_5000samp.png img/REFERENCE_cornell.5000samp.png
python analysis/scripts/compare-images.py img/cornell_specular_5000samp.png img/cornell_specular_5000samp_no_compaction.png
```

| Pair | mean abs difference | other |
|---|---|---|
| diffuse render vs course reference | 2.13 / 255 | RMS 3.21, worst 4x4 region mean within 0.07% |
| compaction on vs off | 1.51 / 255 | global means equal to three decimals |

## One-off measurements

The toggles are the `#define`s at the top of [`src/pathtrace.cu`](../src/pathtrace.cu). Iteration count and trace depth are `ITERATIONS` and `DEPTH` in the scene JSON.
