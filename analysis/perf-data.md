# Performance Data

This file records how every number in the README was measured and what the raw readings were. Interpretation lives in the README.

## Setup

- GPU: NVIDIA GeForce RTX 3090 Ti, 24 GB, driver 616.56
- CPU: AMD Ryzen 9 5950X, 128 GB, Windows 11 Home
- Resolution 800x800 in every run
- Timing comes from `PERF_LOG` in `src/pathtrace.cu`: host-side chrono around `pathtrace()`, printed as an average over each block of 100 iterations
- Paths-alive counts are printed once, from iteration 1

Times are comparable inside one script run, not between runs. `texture/tiles_procedural.json` read 39.86 ms on 2026-09-27, then 46.77 ms and 44.13 ms in two runs an hour apart on 2026-09-28 with the same binary. The code changed between the two days (tangents, bump, the below-surface check), but the last two readings show the machine alone moves the number by a few percent. The first scene of a run is also often the slowest. Effects under about 5% are below what this method can resolve.

Every script runs from the repo root, writes a CSV to `analysis/data/` and the raw console output to `analysis/logs/` (not committed). Scripts that edit `src/pathtrace.cu` or a scene file restore it on exit, including on Ctrl-C, but they do not rebuild afterwards, so the binary left in `build/` is the last configuration that ran. Rebuild before rendering.

| Script | Varies | Rebuilds | Output |
|---|---|---|---|
| `run-perf.sh` | compaction on/off, open/closed box | yes | `paths.csv`, `timing.csv` |
| `run-aa.sh` | sorting, anti-aliasing | yes | `aa-timing.csv` |
| `run-aabb.sh` | mesh AABB culling, triangle count | yes | `aabb-timing.csv` |
| `run-refraction.sh` | sphere material, open/closed box | no | `refraction-timing.csv` |
| `run-depth.sh` | trace depth, open/closed box | no | `depth-timing.csv` |
| `run-texture.sh` | wall material: plain, image, procedural | no | `texture-timing.csv` |
| `run-bump.sh` | bump on/off on procedural tiles | no | `bump-timing.csv` |
| `run-direct.sh` | direct lighting on/off, trace depth | yes | `direct-timing.csv` |

## run-perf.sh: compaction sweep

```bash
bash analysis/scripts/run-perf.sh
```

- Scenes: `core/cornell.json` (open), `core/cornell_closed.json` (closed)
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

- Scene: `core/cornell.json`, 5000 iterations, depth 8, compaction on in every row
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

- Scene: `mesh/suzanne.json` with `suzanne_4k.gltf` and `suzanne_16k.gltf` from `scenes/models/`
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

- Scenes: `core/cornell.json` and `refraction/glass_open.json`, `core/cornell_closed.json` and `refraction/glass_closed.json`. Each pair differs only in the sphere's material
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

- Scenes: `refraction/glass_open.json`, `refraction/glass_closed.json`, with `DEPTH` overridden
- 300 iterations, all toggles on
- Run on 2026-09-26; three readings per row

| Scene | Depth | readings, ms/iteration | avg |
|---|---|---|---|
| glass_open | 8 | 37.06, 37.03, 37.12 | 37.07 |
| glass_open | 32 | 54.78, 58.61, 53.68 | 55.69 |
| glass_closed | 8 | 64.96, 64.72, 65.38 | 65.02 |
| glass_closed | 32 | 218.20, 218.39, 220.85 | 219.15 |

## run-texture.sh: plain versus image versus procedural

```bash
bash analysis/scripts/run-texture.sh
```

- Scenes: `texture/tiles_plain.json`, `texture/tiles_image.json`, `texture/tiles_procedural.json`. The open Cornell box with all five walls sharing one material, so most hits on every bounce sample it
- The image is `scenes/textures/tiles_8x8.png`, 2048 x 2048, generated with the same tile count, grout width and colors as the procedural material
- 300 iterations, depth 8, all toggles on
- Three readings per row
- Run twice. The README quotes the second run, taken on the code that includes bump mapping

Run on 2026-09-28, bilinear sampling:

| Scene | Wall material | readings, ms/iteration | avg |
|---|---|---|---|
| tiles_plain | flat color | 45.02, 45.10, 44.61 | 44.91 |
| tiles_image | image lookup | 43.82, 44.31, 43.61 | 43.91 |
| tiles_procedural | computed tiles | 43.84, 44.53, 44.02 | 44.13 |

Run on 2026-09-27, nearest-texel sampling, before tangents and bump were added:

| Scene | Wall material | readings, ms/iteration | avg |
|---|---|---|---|
| tiles_plain | flat color | 39.38, 39.88, 39.55 | 39.60 |
| tiles_image | image lookup | 39.56, 40.28, 39.44 | 39.76 |
| tiles_procedural | computed tiles | 39.46, 40.15, 39.97 | 39.86 |

`analysis/data/texture-timing.csv` holds the second run; the script overwrites it.

## run-bump.sh: bump on versus off

```bash
bash analysis/scripts/run-bump.sh
```

- Scenes: `texture/tiles_procedural.json` and `bump/tiles_bump.json`, identical except for `"BUMP": 1.0`
- 300 iterations, depth 8, all toggles on
- Run on 2026-09-28, after the below-surface fix; three readings per row

| Scene | Bump | readings, ms/iteration | avg |
|---|---|---|---|
| tiles_procedural | off | 45.59, 45.77, 48.94 | 46.77 |
| tiles_bump | on | 47.23, 46.98, 45.59 | 46.60 |

## measure-bump.py: pixel values across the grout

```bash
python analysis/scripts/measure-bump.py
```

- Images: `img/bump_off_5000samp.png`, `img/blooper/bump_light_leak.png` (before the below-surface fix), `img/bump_on_5000samp.png` (after). All from `bump/bump_on.json` or `bump/bump_off.json`, 800x800, 5000 spp
- Values are the 0 to 255 numbers stored in the PNG, averaged over a strip 30 px wide. The light itself is clipped at 255
- Noise is about 1 at 5000 spp

Across the first horizontal grout line, mean of R, G and B, by image row:

| row | 322 | 324 | 326 | 328 | 330 | 332 | 334 | 336 | 338 | 340 | 342 | 344 | 346 | 348 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bump off | 48 | 48 | 49 | 48 | 49 | 31 | 31 | 30 | 30 | 50 | 50 | 51 | 51 | 51 |
| before fix | 47 | 47 | 48 | 37 | 24 | 31 | 29 | 30 | 29 | 49 | 48 | 50 | 50 | 50 |
| after fix | 48 | 48 | 48 | 41 | 31 | 31 | 30 | 30 | 30 | 53 | 52 | 50 | 50 | 51 |

Rows 328 to 330 are the bottom edge of the upper tile, facing away from the light. Rows 340 to 342 are the top edge of the lower tile, facing it.

Across the first vertical grout line, by image column:

| column | 324 | 326 | 328 | 330 | 332 | 334 | 336 | 338 | 340 | 342 | 344 | 346 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bump off R | 56 | 57 | 57 | 58 | 36 | 36 | 36 | 39 | 60 | 61 | 61 | 62 |
| bump off G | 49 | 50 | 50 | 51 | 31 | 32 | 32 | 35 | 54 | 55 | 56 | 56 |
| before fix R | 55 | 55 | 48 | 53 | 35 | 35 | 36 | 38 | 34 | 55 | 60 | 60 |
| before fix G | 48 | 48 | 52 | 54 | 31 | 31 | 32 | 33 | 21 | 46 | 55 | 54 |
| after fix R | 55 | 56 | 53 | 56 | 35 | 35 | 36 | 38 | 42 | 57 | 61 | 60 |
| after fix G | 48 | 49 | 62 | 58 | 31 | 31 | 32 | 33 | 24 | 47 | 55 | 55 |

Column 328 is the right edge of the left tile, facing the green wall. Column 340 is the left edge of the right tile, facing the red wall.

Mean pixel value of the whole frame: bump off 27.27, before fix 26.43, after fix 26.95.

## run-direct.sh: direct lighting on versus off

```bash
bash analysis/scripts/run-direct.sh
```

- Scenes: `lighting/small_light_depth2.json` and `lighting/small_light_depth8.json`. `core/cornell_diffuse.json` with the light scaled from 3 x 0.3 x 3 to 1 x 0.3 x 1 and its emittance raised from 5 to 45
- Toggle: `DIRECT_LIGHTING`
- 300 iterations, all other toggles on
- Run on 2026-09-28; three readings per row

| Scene | Direct lighting | readings, ms/iteration | avg |
|---|---|---|---|
| small_light_depth2 | on | 20.28, 19.90, 20.14 | 20.11 |
| small_light_depth2 | off | 20.79, 20.14, 20.25 | 20.39 |
| small_light_depth8 | on | 45.41, 45.58, 44.74 | 45.24 |
| small_light_depth8 | off | 45.24, 44.62, 44.24 | 44.70 |

Paths alive after each bounce at depth 8, iteration 1, same with the toggle on or off: 529,113, 371,972, 292,150, 236,567, 194,431, 161,321, 133,863.

## measure-noise.py: brightness and noise of the direct lighting renders

```bash
python analysis/scripts/measure-noise.py img/direct_off_depth2_100samp.png img/direct_on_depth2_100samp.png
python analysis/scripts/measure-noise.py img/direct_off_depth8_100samp.png img/direct_on_depth8_100samp.png
```

- Images rendered by hand from the two scenes above at 100 spp, once with `DIRECT_LIGHTING` 0 and once with 1
- Values are the 0 to 255 numbers stored in the PNG
- Noise is the mean absolute difference between horizontally adjacent pixels inside a region that lies on one flat surface

| Image | Region | mean | noise |
|---|---|---|---|
| depth 2, off | whole frame | 13.73 | |
| | floor | 29.77 | 46.77 |
| | back wall | 33.05 | 46.88 |
| | ceiling | 1.33 | 2.61 |
| depth 2, on | whole frame | 13.57 | |
| | floor | 30.12 | 4.92 |
| | back wall | 33.83 | 5.62 |
| | ceiling | 0.84 | 0.67 |
| depth 8, off | whole frame | 38.04 | |
| | floor | 64.51 | 77.75 |
| | back wall | 74.67 | 81.05 |
| | ceiling | 34.05 | 49.06 |
| depth 8, on | whole frame | 37.78 | |
| | floor | 64.25 | 77.43 |
| | back wall | 74.37 | 80.47 |
| | ceiling | 33.73 | 48.47 |

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
