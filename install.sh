#!/usr/bin/env bash
#
# LiveTranscribe installer — for Ubuntu / GNOME (and other PipeWire desktops).
#
#   ./install.sh               install: Python environment, model, app menu entry
#   ./install.sh --yes         the same, answering yes to every question
#   ./install.sh --uninstall   remove the app menu entry and icons
#   ./install.sh --purge       ...and the .venv and the saved settings too
#
# Nothing here needs root. Anything that does (a missing system package) is
# printed as a command for you to run, never run for you.

set -euo pipefail

APP_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICONS="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor"
DESKTOP="$APPS/livetranscribe.desktop"
SETTINGS="${XDG_CONFIG_HOME:-$HOME/.config}/LiveTranscribe"
MODEL_REPO="Systran/faster-whisper-small"

YES=0
MODE=install
for arg in "$@"; do
    case "$arg" in
        --yes|-y) YES=1 ;;
        --uninstall) MODE=uninstall ;;
        --purge) MODE=purge ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg (try --help)"; exit 2 ;;
    esac
done

bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
ok()    { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn()  { printf '  \033[33m!\033[0m %s\n' "$*"; }
fail()  { printf '  \033[31m✗\033[0m %s\n' "$*"; }
ask() {  # ask "question" → 0 for yes
    [ "$YES" = 1 ] && return 0
    read -r -p "  $1 [Y/n] " reply
    [ -z "$reply" ] || [[ "$reply" =~ ^[Yy] ]]
}

# ── Uninstall ──────────────────────────────────────────────────────────

remove_entry() {
    rm -f "$DESKTOP"
    find "$ICONS" -name 'livetranscribe*.png' -path '*/apps/*' -delete 2>/dev/null || true
    command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" 2>/dev/null || true
    ok "app menu entry and icons removed"
}

if [ "$MODE" != install ]; then
    bold "Removing LiveTranscribe"
    remove_entry
    if [ "$MODE" = purge ]; then
        # The Deepgram key is in the system keyring, not in the settings file.
        for py in "$APP_DIR/.venv/bin/python" python3; do
            if "$py" -c "import keyring; keyring.delete_password('LiveTranscribe', 'deepgram')" 2>/dev/null; then
                ok "removed the Deepgram API key from the keyring"
                break
            fi
        done
        rm -rf "$APP_DIR/.venv" "$SETTINGS"
        ok "removed .venv and saved settings ($SETTINGS)"
        echo "  Downloaded models stay in ~/.cache/huggingface (other apps may use them)."
    fi
    echo "  The program folder itself is left where it is: $APP_DIR"
    exit 0
fi

bold "LiveTranscribe installer"
echo "  folder: $APP_DIR"

# ── 1. What the system must provide ────────────────────────────────────

bold "1. System"
missing=()
if command -v pw-record >/dev/null && command -v pw-metadata >/dev/null && command -v pw-dump >/dev/null; then
    ok "PipeWire tools (pw-record, pw-metadata, pw-dump)"
else
    fail "PipeWire tools are missing — they capture what the computer plays"
    missing+=(pipewire-bin)
fi
# (No `grep -q` after a pipe here: under pipefail it stops reading, the writer
# gets SIGPIPE, and a match is reported as a failure.)
has_lib() {
    local ldc
    ldc=$(command -v ldconfig || echo /sbin/ldconfig)
    "$ldc" -p 2>/dev/null | grep "$1" >/dev/null && return 0
    compgen -G "/usr/lib/*/$1*" >/dev/null || compgen -G "/usr/lib/$1*" >/dev/null
}
if has_lib libxcb-cursor.so.0; then
    ok "libxcb-cursor0 (Qt's X11 support, which keeps the window on top in GNOME)"
else
    fail "libxcb-cursor0 is missing — Qt cannot open its window without it"
    missing+=(libxcb-cursor0)
fi
if fc-list :lang=ar family 2>/dev/null | grep -i "noto" >/dev/null; then
    ok "Arabic fonts (Noto)"
else
    warn "no Noto Arabic font found — text will still show, in whatever font has Arabic"
    missing+=(fonts-noto-core)
fi
GPU=0
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    GPU=1
    ok "NVIDIA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    warn "no working NVIDIA GPU — LiveTranscribe will run on the CPU (slower, still works)"
fi
if [ ${#missing[@]} -gt 0 ]; then
    echo
    echo "  Install what is missing, then run this again:"
    echo "      sudo apt install ${missing[*]}"
    [[ " ${missing[*]} " == *" pipewire-bin "* || " ${missing[*]} " == *" libxcb-cursor0 "* ]] && exit 1
fi

# ── 2. Python ──────────────────────────────────────────────────────────

bold "2. Python environment"
PY=""
CONDA=$(command -v conda 2>/dev/null || true)
[ -z "$CONDA" ] && [ -x "$HOME/anaconda3/bin/conda" ] && CONDA="$HOME/anaconda3/bin/conda"
[ -z "$CONDA" ] && [ -x "$HOME/miniconda3/bin/conda" ] && CONDA="$HOME/miniconda3/bin/conda"
if [ -x "$APP_DIR/.venv/bin/python" ]; then
    PY="$APP_DIR/.venv/bin/python"
    ok "using $APP_DIR/.venv"
elif [ -n "$CONDA" ] && [ -x "$(dirname "$(dirname "$CONDA")")/envs/livetranscribe/bin/python" ]; then
    PY="$(dirname "$(dirname "$CONDA")")/envs/livetranscribe/bin/python"
    ok "using the conda environment 'livetranscribe'"
else
    if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
        fail "Python 3.10 or newer is needed (found: $(python3 --version 2>&1))"
        exit 1
    fi
    if ! python3 -m venv --help >/dev/null 2>&1; then
        fail "python3-venv is missing:  sudo apt install python3-venv"
        exit 1
    fi
    echo "  creating $APP_DIR/.venv"
    python3 -m venv "$APP_DIR/.venv"
    PY="$APP_DIR/.venv/bin/python"
    ok "created .venv ($("$PY" --version))"
fi
echo "  installing packages (a few minutes the first time)…"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r "$APP_DIR/requirements.txt"
ok "packages from requirements.txt"
if [ "$GPU" = 1 ]; then
    if "$PY" -c "import nvidia.cublas, nvidia.cudnn" 2>/dev/null; then
        ok "CUDA libraries already there"
    elif ask "Install NVIDIA's CUDA libraries for the GPU (~1.4 GB)?"; then
        "$PY" -m pip install --quiet -r "$APP_DIR/requirements-gpu.txt"
        ok "CUDA libraries (requirements-gpu.txt)"
    else
        warn "skipped — the app will use the CPU"
    fi
fi

# ── 3. The speech model ────────────────────────────────────────────────

bold "3. Whisper model"
if "$PY" - "$MODEL_REPO" <<'PYEOF' 2>/dev/null
import sys
from huggingface_hub import try_to_load_from_cache
sys.exit(0 if isinstance(try_to_load_from_cache(sys.argv[1], "model.bin"), str) else 1)
PYEOF
then
    ok "$MODEL_REPO is already downloaded"
else
    echo "  This is the default model, fetched now, once. Others can be downloaded later in ⚙ Settings."
    echo "  (Only using Deepgram, in the cloud? Then this can be skipped.)"
    if ask "Download $MODEL_REPO (~480 MB)?"; then
        "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_REPO')"
        ok "model downloaded to ~/.cache/huggingface"
    else
        warn "skipped — the app will say the model is missing until it is downloaded"
    fi
fi

# ── 4. The app menu ────────────────────────────────────────────────────

bold "4. App menu"
chmod +x "$APP_DIR/run.sh"
for png in "$APP_DIR"/packaging/icons/livetranscribe-[0-9]*.png; do
    size=$(basename "$png" .png | sed 's/livetranscribe-//')
    mkdir -p "$ICONS/${size}x${size}/apps"
    cp "$png" "$ICONS/${size}x${size}/apps/livetranscribe.png"
done
mkdir -p "$APPS"
sed "s|@APP_DIR@|$APP_DIR|g" "$APP_DIR/packaging/livetranscribe.desktop.in" > "$DESKTOP"
chmod +x "$DESKTOP"
command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" 2>/dev/null || true
command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q -f -t "$ICONS" 2>/dev/null || true
ok "LiveTranscribe is in the app menu ($DESKTOP)"

echo
bold "Done."
echo "  Open it: press the Super key, type LiveTranscribe, press Enter."
echo "  Or from here:  $APP_DIR/run.sh"
echo "  Remove it:     $APP_DIR/install.sh --uninstall"
