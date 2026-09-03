#!/bin/bash
# Regenerates docs/demo.gif from a fresh headless recording of a GEPA run.
#
#   npm run demo:record          # with `npm run dev` up and Ollama running
#
# Needs a full ffmpeg on PATH (Homebrew: brew install ffmpeg). The one bundled
# with Playwright has no GIF muxer. The waiting phase is sped up 10x, the
# setup and the results are shown at (near) real speed.

set -euo pipefail
cd "$(dirname "$0")/../Web"

command -v ffmpeg >/dev/null || { echo "ffmpeg not found (brew install ffmpeg)"; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe not found (brew install ffmpeg)"; exit 1; }

VIDEO=$(node e2e/record-demo.mjs | tail -1)
[ -f "$VIDEO" ] || { echo "recording failed"; exit 1; }

T=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO")
SETUP_END=13                                   # seconds of typing and picking options, real speed
RESULTS_START=$(python3 -c "print(max($SETUP_END + 1, $T - 12))")
OUT=../docs/demo.gif

ffmpeg -y -loglevel error -i "$VIDEO" -filter_complex \
  "[0:v]trim=0:$SETUP_END,setpts=PTS-STARTPTS[a];\
   [0:v]trim=$SETUP_END:$RESULTS_START,setpts=(PTS-STARTPTS)/10[b];\
   [0:v]trim=$RESULTS_START,setpts=(PTS-STARTPTS)/1.4[c];\
   [a][b][c]concat=n=3:v=1:a=0,fps=8,scale=880:-1:flags=lanczos,split[s0][s1];\
   [s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" \
  "$OUT"

echo "wrote $OUT ($(du -h "$OUT" | cut -f1), $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT" | cut -c1-5)s)"
rm -rf e2e/video
