# Performance Data

Measured with `PERF_LOG` in `pathtrace.cu`: host-side chrono timing of `pathtrace()` per iteration, averaged per 100 iterations over a full 5000-iteration run, at 800x800 and trace depth 8. Scenes are `scenes/cornell.json` and `scenes/cornell_closed.json`. Runs dated 2026-09-21 and 2026-09-22; each section says which.

GPU: NVIDIA GeForce RTX 3090 Ti, 24 GB, driver 616.56. CPU: AMD Ryzen 9 5950X, 128 GB. Windows 11 Home.

## How to reproduce

`analysis/run-perf.sh` runs the whole compaction sweep. It builds and runs the open and the closed Cornell box with compaction on and off, with sorting and anti-aliasing forced off in all four so that compaction is the only variable, and it writes the results as CSV.

```bash
bash analysis/run-perf.sh
```

| Output | Columns |
|---|---|
| `analysis/paths.csv` | scene, compaction, bounce, paths_alive |
| `analysis/timing.csv` | scene, compaction, ms_per_iter, fps |
| `analysis/logs/*.log` | raw console output of each run |

`analysis/run-aa.sh` does the two anti-aliasing runs on the open box and writes `analysis/aa-timing.csv` (config, sort, aa, ms_per_iter, fps):

```bash
bash analysis/run-aa.sh
```

`analysis/plot-paths.py` turns `paths.csv` into `img/paths_alive_per_bounce.png`, the chart used in the README:

```bash
python analysis/plot-paths.py
```

The iteration count is set by `ITERS` at the top of `run-perf.sh`, currently 5000, which is what every published number was measured at. Timing does not actually need that many: the per-100-iteration average is flat from the first block onward and the paths-alive counts come from iteration 1 either way, so an earlier 500-iteration pass agreed to within a few tenths of a millisecond. 5000 is kept because it also produces a usable render.

The script edits `src/pathtrace.cu` and the scene files in place and restores them on exit, including on Ctrl-C. Two things to know about that. The restore writes through the existing files rather than replacing them, because Visual Studio holding a file open will otherwise block it, and if a restore still fails the script says so and leaves the originals in its temp directory. Separately, `sed -i` rewrites those files with LF endings while the sweep runs, which only matters if the trap gets skipped entirely.

Each run's PNG is moved into `build/` as `<scene>-compaction-<0|1>-<ITERS>samp.png`. Rename the keepers into `img/`.

For a one-off measurement outside the sweep, the toggles are the four `#define`s at the top of [`src/pathtrace.cu`](../src/pathtrace.cu) and the iteration count is `ITERATIONS` in the scene JSON.

## Material sort ON vs OFF (open Cornell box)

Authoritative pair, from `analysis/run-aa.sh` (2026-09-22). Compaction and AA are on in both rows, so sorting is the only variable.

| Config | avg ms/iteration | range | ~FPS |
|---|---|---|---|
| Sort OFF | 20.11 | 19.50 - 21.31 | 50 |
| Sort ON (`thrust::sort_by_key`, intersections as keys, paths as values) | 36.03 | 35.27 - 37.07 | 28 |

Sorting costs 15.92 ms/iteration, a 79% slowdown. The shading kernel branches three ways (emitter, scatter, miss) and `scatterRay` splits again into diffuse and specular, none of them expensive, so there is very little divergence for sorting to remove. What it does cost in full is a `thrust::sort_by_key` over two struct arrays, 20-byte intersections as keys and 44-byte path segments as values, up to 640k elements, at each of the 8 bounces. Sorting should only pay off with many materials or genuinely expensive per-material shading.

An earlier standalone run (2026-09-21, AA off, compaction on) put sort-on at ~36.0-38.7 ms against a ~20.6 ms baseline. It agrees, but its two rows were not matched to each other as cleanly, so the pair above is the one quoted in the README.

Sorting also shifts the paths-alive counts very slightly, for example 362,755 against 362,150 at bounce 2, because it moves a path into a different array slot while the RNG is seeded per index. The runs stay statistically equivalent and converge to the same image.

## Anti-aliasing (2026-09-22, resolved)

Two runs from `analysis/run-aa.sh`, 5000 iterations on the open Cornell box, compaction on in both. Raw output in `analysis/aa-timing.csv` and `analysis/logs/aa-*.log`.

| Config | sort | AA | avg ms/iteration | range | ~FPS |
|---|---|---|---|---|---|
| sweep baseline | off | off | 20.40 | 19.65 - 21.19 | 49 |
| aa-only | off | on | 20.11 | 19.50 - 21.31 | 50 |
| all-on | on | on | 36.03 | 35.27 - 37.07 | 28 |

AA costs -0.29 ms against the baseline, i.e. the jittered run came out marginally faster and the two ranges overlap almost entirely over 50 blocks each. The cost is smaller than the run-to-run spread of either run, so this measurement cannot separate it from zero; it bounds AA at well under 1.5% of a frame rather than pinning a value. That matches the mechanism: two random draws in the ray generation kernel, which runs once per iteration, against an eight-deep bounce loop AA never enters.

The aa-only and all-on rows also give the single-variable sorting measurement quoted in the section above.

Images: `img/cornell_specular_5000samp_aa.png` (full render, all toggles on, 36.0 ms/iter), `img/aa_{on,off}_{sphere_edge,light_edge}.png` (4x nearest-neighbor close-ups; the sphere-edge pair clearly shows stair-stepping vs smooth silhouette).

## Discrepancy resolved

The earlier anti-aliasing entry recorded 40.5-45.9 ms/iter (typ. 42) for compaction + sort + AA all on. Re-running that exact configuration gave 36.03 ms, matching an independent 36.0 ms measurement on `cornell_diffuse.json`. Two runs agree against one, so the 42 figure was contaminated and is discarded. Nothing in the README depends on it any more.

## Open vs closed Cornell box, all toggles on (2026-09-22, superseded)

Superseded by the sweep below for every open-vs-closed claim; kept because its 36.0 ms figure is one half of the anti-aliasing discrepancy. Both runs had compaction, sorting and anti-aliasing all on, 5000 iterations, 800x800, depth 8. The open scene is `scenes/cornell_diffuse.json` (diffuse sphere) and the closed one is `scenes/cornell_closed.json` (specular sphere, front wall at z=+5, camera moved inside to [0,5,4.9]). These are not the clean compaction-only comparison -- that needs the four runs described above -- but the paths-alive contrast is already decisive.

| Bounce | Open, paths alive | Open, % of 640k | Closed, paths alive | Closed, % of 640k |
|---|---|---|---|---|
| 1 | 522,877 | 82% | 613,895 | 96% |
| 2 | 360,502 | 56% | 600,904 | 94% |
| 3 | 278,065 | 43% | 590,148 | 92% |
| 4 | 221,486 | 35% | 579,851 | 91% |
| 5 | 179,596 | 28% | 569,868 | 89% |
| 6 | 146,587 | 23% | 560,132 | 88% |
| 7 | 119,897 | 19% | 550,769 | 86% |

| Scene | avg ms/iteration | ~FPS |
|---|---|---|
| Open | 35.0 - 36.7, typ. 36.0 | 28 |
| Closed | 62.0 - 63.3, typ. 62.5 | 16 |

The open box loses 18-31% of its surviving paths per bounce; the closed box loses 4.1% at bounce 1 and 1.7% thereafter. Sealing the front wall removes escape-to-background, leaving a light hit as the only way to terminate early, and a light hit is rare. By bounce 7 the closed scene still has 86% of its paths alive against 19% for the open one, which is exactly the situation where compaction has almost nothing left to remove while still paying its full per-bounce cost.

The closed scene is also 1.7x slower per iteration in absolute terms, for a reason that has nothing to do with compaction: paths that used to fly out of the open wall at bounce 1 now keep bouncing and keep paying for intersection tests.

Open-scene image: `img/cornell_diffuse_5000samp.png`. Verified against `img/REFERENCE_cornell.5000samp.png` by per-pixel comparison with `analysis/compare-images.py` -- mean absolute difference 2.13/255, RMS 3.21, worst 4x4 region mean within 0.07%, which is Monte Carlo noise rather than a systematic difference. The same script produces the compaction-pair comparison (MAD 1.51/255, global means equal to three decimals):

```bash
python analysis/compare-images.py img/cornell_diffuse_5000samp.png img/REFERENCE_cornell.5000samp.png
python analysis/compare-images.py img/cornell_specular_5000samp.png img/cornell_specular_5000samp_no_compaction.png
```

## Compaction sweep (2026-09-22) -- source of record

Every compaction number in the README comes from here. Four runs from `analysis/run-perf.sh` at 5000 iterations, sorting and anti-aliasing off in all four so compaction is the only variable. An earlier 500-iteration pass of the same four runs agreed with this one to within a few tenths of a millisecond, confirming that ms/iteration does not depend on the iteration count; its output is kept locally but not committed.

| Scene | Compaction | avg ms/iteration | range | ~FPS |
|---|---|---|---|---|
| open | on | 20.40 | 19.65 - 21.19 | 49 |
| open | off | 17.09 | 16.73 - 17.58 | 59 |
| closed | on | 35.97 | 35.10 - 37.16 | 28 |
| closed | off | 19.74 | 19.21 - 21.05 | 51 |

| Bounce | open, compaction on | closed, compaction on | compaction off, both scenes |
|---|---|---|---|
| 1 | 523,164 | 613,916 | 640,000 |
| 2 | 362,150 | 601,084 | 640,000 |
| 3 | 279,680 | 590,284 | 640,000 |
| 4 | 222,899 | 580,254 | 640,000 |
| 5 | 180,366 | 570,499 | 640,000 |
| 6 | 146,600 | 560,637 | 640,000 |
| 7 | 119,912 | 551,106 | 640,000 |

Compaction costs 3.3 ms/iteration in the open box, a 19% slowdown, and 16.2 ms in the closed box, an 82% slowdown. The same optimization is roughly five times more expensive once the room is sealed.

`thrust::partition` pays for the length of the array it scans and moves. The open box collapses from 640k live paths to 120k over eight bounces, so its later partitions are cheap. The closed box never falls below 551k, so every bounce moves nearly the whole array and removes almost nothing.

With compaction off the two scenes sit close together, 17.09 against 19.74, because neither shortens its launches and the sealed room only does a little more real intersection work. Most of the closed box's 35.97 ms is therefore compaction's own overhead rather than the scene being intrinsically harder. The margin inverts the ranking outright: the closed box with compaction off beats the open box with compaction on, 19.74 against 20.40.

## TODO (for README charts)

- [x] Closed Cornell box variant: paths-per-bounce done (see above)
- [x] Closed + open ms/iter with compaction ON vs OFF at 5000 iterations, written up in the README
- [x] Re-measure AA cost on cornell.json -- done, AA is free within noise
- [x] Material sort ON vs OFF — done, see above
- [x] AA on/off close-up crops — done, in img/
- [ ] Re-run compaction comparison on a heavy scene (after mesh loading)
- [x] Diffuse-sphere cornell render, verified against the course reference
