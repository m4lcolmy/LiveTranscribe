#!/bin/bash

# LiveTranscribe - Launcher
# Activates the conda environment and runs the app with whatever arguments
# it was given (./run.sh --file clip.mp4, ./run.sh --record, ...).

# Run from the project, wherever the launcher was called from.
cd "$(dirname "$(readlink -f "$0")")" || exit 1

CONDA_PATH=$(which conda 2>/dev/null)
if [ -z "$CONDA_PATH" ]; then
    CONDA_PATH="$HOME/anaconda3/bin/conda"
fi
CONDA_BASE=$(dirname "$(dirname "$CONDA_PATH")")

if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate livetranscribe
else
    echo "Conda not found; running with the current python."
fi
exec python app.py "$@"
