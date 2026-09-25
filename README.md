CUDA Path Tracer
================

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 3**

* Anna Dai
* Tested on: Windows 11 Home, AMD Ryzen 9 5950X @ 3.40GHz 128GB, NVIDIA GeForce RTX 3090 Ti 24GB, personal machine


This is a physically-based path tracer that runs entirely on the GPU. Instead of giving each pixel a thread and looping the whole path inside one kernel, the renderer parallelizes over *path segments*.


> TODO: replace this hero image with the custom scene

![](img/cornell_specular_5000samp_aa.png)

*Cornell box with a perfect mirror sphere. 800x800, 5000 samples per pixel, trace depth 8, compaction, material sorting and stochastic anti-aliasing all on. 36.0 ms/iteration, so the full render takes just over three minutes.*

## Overview

The renderer parallelizes over *path segments*: the depth loop lives on the host, and each kernel launch advances every live path by exactly one bounce.
Path state sits in device-memory arrays between launches, which is what makes it possible to compact dead paths away and reorder live ones by material before shading.

One iteration of the pipeline looks like this:

```
generateRayFromCamera            one ray per pixel, jittered inside the pixel for AA
for depth in 0 .. 7:
    computeIntersections         closest hit for every live path
    sort by materialId           toggleable, groups same-material paths together
    shadeMaterial                evaluate BSDF, update throughput, spawn next ray
    stream compaction            thrust::partition, drop terminated paths
    break if no paths remain
finalGather                      accumulate each path into its pixel via pixelIndex
```

Because compaction and sorting both shuffle the path array, every `PathSegment` carries its own `pixelIndex` — that is the only thing that lets the final gather find its way home.

## Features

Shading is done with a single kernel that branches on material type. 
- Ideal diffuse surfaces: use cosine-weighted hemisphere sampling, where the cosine term in the estimator and the cosine in the PDF cancel, so throughput is just multiplied by the surface albedo
- Perfect specular surfaces: reflect the incoming direction about the normal with no randomness at all


- Rays that miss the scene entirely, or that run out of bounces before finding the light -> black
- Rays that land on an emitter settle up immediately -> the accumulated throughput is multiplied by the emitted radiance and the path terminates

Stream compaction, material sorting, anti-aliasing and the performance logging are each behind a `#define` at the top of [`src/pathtrace.cu`](src/pathtrace.cu), so any combination can be built and measured without touching the rest of the code.

## Renders

### Diffuse only, against the course reference

| Mine, 5000 spp | Course reference, 5000 spp |
|---|---|
| ![](img/cornell_diffuse_5000samp.png) | ![](img/REFERENCE_cornell.5000samp.png) |

The render matches the reference on the two things that would break first if the diffuse BSDF or the throughput bookkeeping were wrong, which are the color bleeding off the red and green walls and the soft contact shadow under the sphere. A per-pixel comparison puts the mean absolute difference at 2.1 out of 255 and every region mean within 0.4%, which is the Monte Carlo noise left at 5000 samples rather than a systematic difference.

### Specular

![](img/cornell_specular_5000samp.png)

*Same scene, sphere switched to `Specular` with roughness 0. 800x800, 5000 spp, depth 8.*

- The mirror picks up the red and green walls and reflects the light straight back at the camera
- The dark disk in the middle is the camera's own line of sight leaving through the open front of the box -> nothing to hit -> black

### Anti-aliasing

Each camera ray is jittered by a uniform random offset inside its pixel footprint, so over 5000 iterations every pixel integrates its whole area instead of one fixed point. 

Cost: seeding one RNG per ray in the generation kernel and drawing from it twice, once for the x offset and once for the y.

Note that pictures below are 4x nearest-neighbor blowups, so the pixel grid stays visible:

| | AA off | AA on |
|---|---|---|
| Sphere silhouette | ![](img/aa_off_sphere_edge.png) | ![](img/aa_on_sphere_edge.png) |
| Light fixture edge | ![](img/aa_off_light_edge.png) | ![](img/aa_on_light_edge.png) |

The sphere pair clearly shows that without jitter the silhouette is a hard staircase of fully-lit and fully-dark pixels; with jitter the boundary pixels land in between, in proportion to how much of the sphere actually covers them.

## Performance analysis

Timing is a host-side `std::chrono` measurement around the whole `pathtrace()` call, averaged per 100 iterations over a full 5000-iteration run with `PERF_LOG` on, at 800x800 and trace depth 8, which means 640,000 paths are born per iteration.

The compaction numbers come from one set of four runs, `analysis/scripts/run-perf.sh`, with sorting and anti-aliasing off throughout so that compaction is the only variable. The sorting and anti-aliasing numbers come from a second pair of runs on the same scene, `analysis/scripts/run-aa.sh`. Raw CSVs and logs are described in [`analysis/perf-data.md`](analysis/perf-data.md), and the per-pixel image comparisons quoted in this README come from [`analysis/scripts/compare-images.py`](analysis/scripts/compare-images.py).

The three optimizations below were each measured on the same open Cornell box, a scene with seven primitives and two BSDFs.

| Feature | Effect on this scene |
|---|---|
| Stream compaction | +3.3 ms/iteration, 19% slower |
| Material sorting | +15.9 ms/iteration, 79% slower |
| Anti-aliasing | -0.29 ms, within measurement noise |

Two of the three cost more than they save here. That looks like a property of the scene rather than of the implementations: compaction and sorting target idle warp lanes and branch divergence, and a scene this simple has little of either, while the bandwidth they spend moving path segments around is the same either way. The sections below take each one in turn.

### Paths actually survive each bounce

Path counts are taken from iteration 1 of the open Cornell box:

| Bounce | Paths alive | Fraction of the original 640k |
|---|---|---|
| start | 640,000 | 100% |
| 1 | 523,164 | 82% |
| 2 | 362,150 | 57% |
| 3 | 279,680 | 44% |
| 4 | 222,899 | 35% |
| 5 | 180,366 | 28% |
| 6 | 146,600 | 23% |
| 7 | 119,912 | 19% |

- Between 18% and 31% of the surviving paths die at every bounce, either escaping the open front of the box or landing on the light. Bounce 2 is the outlier at 31%; the rest sit between 18% and 23%
- Without compaction all 640,000 threads launch at every depth anyway -> by the last bounce four out of five do nothing but read `remainingBounces`, see zero, and return

### Stream compaction, on versus off

| Configuration | avg ms/iteration | ~FPS |
|---|---|---|
| Compaction on (`thrust::partition`) | 20.40 | 49 |
| Compaction off | 17.09 | 59 |

Compaction is a net *loss* of 3.3 ms/iteration here, a 19% slowdown, which is the opposite of what the idle-thread argument predicts. That argument only counts the work saved, not the work added.

- Work saved is small: 7 primitives in the scene, so one bounce of intersection + shading is a handful of ray-box tests. A warp full of dead threads costs very little
- Work added is not: `thrust::partition` runs over the path array 8 times per iteration, physically moving 44-byte `PathSegment` structs through global memory
- The bandwidth spent shuffling outweighs the arithmetic it saves

The two configurations converge to the same image without being byte-identical. Measured over the 5000-sample pair, the mean absolute difference is 1.5 out of 255 and the two global means agree to three decimals, which looks more like the residual Monte Carlo noise rather than a disagreement. They are not bit-exact because compaction moves a path into a different array slot and the RNG is seeded per index, so the path draws a different random number than it would have uncompacted.

The balance should (in theory) flip once per-ray work gets expensive, as it will with arbitrary meshes, thousands of primitives and deeper traces. In that setting the cost of an idle warp slot grows while the cost of a partition stays where it is, so this comparison is one I plan to re-measure if mesh loading and a BVH are in place.

### Open versus closed scene

`scenes/cornell_closed.json` is the same room with a wall across the open front at z=+5 and the camera moved inside to sit just in front of it, so nothing can leave the scene.

![](img/cornell_closed_5000samp.png)

*The closed box at 5000 spp, 800x800, depth 8. The centre of the mirror sphere used to be a black disk where the camera's own line of sight escaped through the open front; it now reflects the blue wall standing behind the camera, which is the quickest visual confirmation that the room is really sealed.*

![](img/paths_alive_per_bounce.png)

*Unterminated paths after each bounce of a single iteration, plotted from `analysis/data/paths.csv` by `analysis/scripts/plot-paths.py`. The dashed line is what runs when compaction is off, which is every thread at every depth in both scenes.*

| Bounce | Paths alive, open box | Paths alive, closed box |
|---|---|---|
| start | 640,000 (100%) | 640,000 (100%) |
| 1 | 523,164 (82%) | 613,916 (96%) |
| 2 | 362,150 (57%) | 601,084 (94%) |
| 3 | 279,680 (44%) | 590,284 (92%) |
| 4 | 222,899 (35%) | 580,254 (91%) |
| 5 | 180,366 (28%) | 570,499 (89%) |
| 6 | 146,600 (23%) | 560,637 (88%) |
| 7 | 119,912 (19%) | 551,106 (86%) |

| Configuration | Open box | Closed box |
|---|---|---|
| Compaction on | 20.40 ms, 49 FPS | 35.97 ms, 28 FPS |
| Compaction off | 17.09 ms, 59 FPS | 19.74 ms, 51 FPS |

Sealing the box removes the only cheap way for a path to die, and that turns compaction from a mild loss into a severe one.

- Compaction costs 3.3 ms/iteration in the open box and 16.2 ms in the closed one, a 19% slowdown against an 82% slowdown -> the same optimization is roughly five times more expensive once the room is sealed
- The path counts say why: `thrust::partition` pays for the length of the array it scans and moves. The open box collapses from 640k to 120k over eight bounces, so its later partitions are cheap. The closed box never falls below 551k, so every bounce moves nearly the whole array and deletes almost nothing
- With compaction off the two scenes are close, 17.09 against 19.74, because neither one shortens its launches and the sealed room only does a little more real intersection work -> most of the closed box's 35.97 ms is compaction's own overhead rather than the scene being intrinsically harder. One caveat on comparing the two scenes' absolute times: the closed box also moves the camera inside the room and widens the FOV from 45° to 60°, so its ray distribution differs; the on/off comparisons within each scene are unaffected
- The margin is wide enough to invert the ranking: the closed box with compaction off (19.74 ms) beats the open box with compaction on (20.40 ms)

### Sorting by material

| Configuration | avg ms/iteration | ~FPS |
|---|---|---|
| Sort off | 20.11 | 50 |
| Sort on (`thrust::sort_by_key`) | 36.03 | 28 |

Both rows have compaction and anti-aliasing on, so sorting is the only variable. It costs 15.9 ms per iteration, a 79% slowdown.

- What it targets: warp divergence in the shading kernel, by making a warp's 32 threads hit the same material branch
- Why there is nothing to win here: `shadeMaterial` branches three ways (emitter, scatter, miss) and `scatterRay` splits again into diffuse and specular, none of them expensive -> almost no divergence to cure
- What it costs: a full `thrust::sort_by_key` over 20-byte intersection keys and 44-byte path values, up to 640k elements, at every one of the 8 bounces
- When it would pay: many materials with genuinely different shading costs, so an unsorted warp stalls on its slowest thread — refraction with Fresnel, texture lookups, a microfacet BSDF

One detail is worth flagging. The paths-alive counts shift slightly when sorting is on, for instance 362,755 against 362,150 at bounce 2. This is most likely because sorting moves a path into a different array slot while the RNG is seeded per index, so the path draws a different random number than it would have unsorted. The two runs are statistically equivalent and converge to the same image.

### Cost of anti-aliasing

| Configuration | avg ms/iteration | range over 50 blocks | ~FPS |
|---|---|---|---|
| Compaction on, sorting off, AA off | 20.40 | 19.65 – 21.19 | 49 |
| Compaction on, sorting off, AA on | 20.11 | 19.50 – 21.31 | 50 |

The difference is -0.29 ms, meaning the run with jitter came out marginally *faster*, and the two ranges overlap almost entirely. The cost of anti-aliasing is smaller than the run-to-run spread of either run, so this measurement cannot separate it from zero. What it does establish is a bound: whatever AA costs, it is well under 1.5% of a frame.

That is where the work sits, too. The jitter is two random draws in the ray generation kernel, which runs once per iteration, while frame time is dominated by the eight-deep bounce loop that anti-aliasing never touches. It buys a visibly better silhouette for nothing measurable.

An earlier run had put the cost at 3 to 9 ms. That measurement was contaminated: re-running the same three toggles gave 36.03 ms against the 40.5 – 45.9 ms first recorded, so the earlier figure is discarded rather than reported.

### Versus a hypothetical CPU version

A single-threaded CPU tracer would take minutes per frame instead of milliseconds, and two of the three optimizations above would be pointless on it.

- Stream compaction: exists because GPU threads run in lockstep warps and an idle lane is wasted silicon. A CPU loop skips a dead path with a branch that costs nothing
- Material sorting: exists because a diverging warp serializes its branches. A CPU core takes the branch it needs and moves on
- Anti-aliasing: transfers unchanged, same cost on either side

## Build notes

`CMakeLists.txt` has one change beyond the source file list: MSVC gets `/Zc:preprocessor` for both C++ and CUDA, which enables the conforming preprocessor.

```powershell
cmake --build build --config Release
& ".\build\bin\Release\cis565_path_tracer.exe" "scenes/cornell.json"
```

- Esc saves the image and exits
- S saves the image without exiting, and the filename is printed to the console

## Bloopers

> TODO: add any debug renders worth keeping here

## References
- Path tracing background and the BSDF formulation follow [Physically Based Rendering, 4th edition](https://pbr-book.org/4ed/Reflection_Models/Diffuse_Reflection)
- Staged-kernel pipeline follows the CIS 5650 [path tracing primer recitation](https://docs.google.com/presentation/d/1rr6zFbpVkdMEkxBK4QLN4_tBRo168SJA_bMi2GkBB6I/edit?usp=drive_link)
- Compaction and sorting use [Thrust](https://nvidia.github.io/cccl/thrust/)
