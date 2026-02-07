#!/bin/bash
# im2pc-pipeline: Run the full 3D reconstruction pipeline
#
# Usage:
#   ./run.sh <video_file>
#   ./run.sh data/input/mov_makersmark.mp4
#
# The video file should be in data/input/ (or provide an absolute path inside /data/).

set -e

VIDEO="${1:?Usage: ./run.sh <video_file>}"

# If path doesn't start with /data, assume it's relative to data/input
if [[ "$VIDEO" != /data/* ]]; then
    VIDEO="/data/input/$(basename "$VIDEO")"
fi

echo "=== im2pc-pipeline ==="
echo "Input: $VIDEO"
echo "Output: /data/output/"
echo ""

docker compose run --rm \
    --service-ports \
    pipeline \
    /app/scripts/pipeline.py "$VIDEO" --output-dir /data/output
