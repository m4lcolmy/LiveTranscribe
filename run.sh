#!/bin/bash

# LiveTranscribe - launcher
#
# Runs the app with whatever arguments it was given (./run.sh --console,
# ./run.sh --reset-settings, ...), in the first Python environment it finds:
#   1. .venv next to this script   (made by ./install.sh)
#   2. the conda environment "livetranscribe"
#   3. python3 as it is

# Run from the project, wherever the launcher was called from — the app menu
# starts it in $HOME, and app.py is a relative path.
cd "$(dirname "$(readlink -f "$0")")" || exit 1

if [ -x ".venv/bin/python" ]; then
    exec .venv/bin/python app.py "$@"
fi

CONDA_PATH=$(command -v conda 2>/dev/null)
[ -z "$CONDA_PATH" ] && [ -x "$HOME/anaconda3/bin/conda" ] && CONDA_PATH="$HOME/anaconda3/bin/conda"
[ -z "$CONDA_PATH" ] && [ -x "$HOME/miniconda3/bin/conda" ] && CONDA_PATH="$HOME/miniconda3/bin/conda"
if [ -n "$CONDA_PATH" ]; then
    CONDA_BASE=$(dirname "$(dirname "$CONDA_PATH")")
    if [ -d "$CONDA_BASE/envs/livetranscribe" ]; then
        exec "$CONDA_BASE/envs/livetranscribe/bin/python" app.py "$@"
    fi
fi

echo "No LiveTranscribe environment found; trying python3. Run ./install.sh to set one up." >&2
exec python3 app.py "$@"
