#!/usr/bin/env bash
# Times the mirror sphere against the glass sphere in the open and the closed
# Cornell box. Writes analysis/data/refraction-timing.csv, no rebuild needed

set -euo pipefail
cd "$(dirname "$0")/../.."

EXE=build/bin/Release/cis565_path_tracer.exe
ITERS=300
# scene file, material label, box label
RUNS=(
    "cornell,mirror,open"
    "glass_open,glass,open"
    "cornell_closed,mirror,closed"
    "glass_closed,glass,closed"
)

mkdir -p analysis/logs
echo "scene,material,box,ms_per_iter,fps" > analysis/data/refraction-timing.csv

for run in "${RUNS[@]}"; do
    IFS=, read -r scene material box <<< "$run"
    tmp="scenes/tmp_$scene.json"
    sed -E "s/\"ITERATIONS\": *[0-9]+/\"ITERATIONS\":$ITERS/; s/\"FILE\":\"[^\"]+\"/\"FILE\":\"tmp_$scene\"/" "scenes/$scene.json" > "$tmp"

    log="analysis/logs/refraction-$scene.log"
    echo "=== $scene ($material, $box box) ==="
    "$EXE" "$tmp" | tee "$log"
    rm -f "$tmp" tmp_"$scene".*samp.png
    awk -v s="$scene" -v m="$material" -v b="$box" '
        /ms\/iteration/ { gsub(/[()]/,""); print s","m","b","$5","$7 >> "analysis/data/refraction-timing.csv" }
    ' "$log"
done

echo "wrote analysis/data/refraction-timing.csv"
cat analysis/data/refraction-timing.csv
