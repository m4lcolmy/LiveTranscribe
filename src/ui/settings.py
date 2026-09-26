"""Settings: what the user chose, kept between sessions, and the dialog to choose it.

Everything here is saved with QSettings (~/.config/LiveTranscribe/LiveTranscribe.conf)
the moment the dialog is saved, and read back at the next start, so the app
opens the way it was left. A command-line flag overrides a saved value for that
run only; it is never written back.

The model list is the models actually in the Hugging Face cache: the app never
downloads anything, so offering a model that is not there would only offer an
error. Any other CTranslate2 Whisper folder can be picked by hand.
"""

import json
import os
import subprocess
from dataclasses import asdict, dataclass, fields

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QLabel,
    QSlider, QSpinBox, QVBoxLayout, QWidget, QHBoxLayout,
)

from src.config import (
    DEVICE, OVERLAY_FONT_MAX_PX, OVERLAY_FONT_MIN_PX, OVERLAY_FONT_PX, OVERLAY_OPACITY,
    WHISPER_MODEL,
)

SETTINGS_ORG = "LiveTranscribe"
SETTINGS_APP = "LiveTranscribe"


def open_store() -> QSettings:
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


@dataclass
class AppSettings:
    model: str = WHISPER_MODEL        # a size name from the cache, or a model folder
    device: str = DEVICE              # auto | cuda | cpu
    precision: str = "auto"           # auto | float16 | int8_float16 | int8
    step_s: float = 0.0               # seconds between passes; 0 = by device
    sink: str = ""                    # an output's node.name; "" follows the default
    font_px: int = OVERLAY_FONT_PX
    opacity: int = OVERLAY_OPACITY    # background, percent
    show_tentative: bool = True       # show words that may still change
    arabic_only: bool = True          # drop speech Whisper hears as another language
    click_through: bool = False

    @classmethod
    def load(cls, store: QSettings) -> "AppSettings":
        values = {}
        for f in fields(cls):
            default = f.default
            raw = store.value(f"settings/{f.name}", default)
            try:
                if isinstance(default, bool):
                    values[f.name] = raw if isinstance(raw, bool) else str(raw).lower() == "true"
                else:
                    values[f.name] = type(default)(raw)
            except (TypeError, ValueError):
                values[f.name] = default           # a corrupt value falls back alone
        return cls(**values)

    def save(self, store: QSettings):
        for name, value in asdict(self).items():
            store.setValue(f"settings/{name}", value)
        store.sync()

    # What a change needs: a new engine, a new pipeline, or just the window.
    ENGINE = ("model", "device", "precision")
    PIPELINE = ("step_s", "sink", "arabic_only")

    def needs(self, other: "AppSettings") -> str:
        """'engine', 'pipeline' or 'window': the least that must restart to apply `other`."""
        if any(getattr(self, k) != getattr(other, k) for k in self.ENGINE):
            return "engine"
        if any(getattr(self, k) != getattr(other, k) for k in self.PIPELINE):
            return "pipeline"
        return "window"


# ── What there is to choose from ───────────────────────────────────────

MODEL_NOTES = {
    "small": "fast; the tuned default",
    "large-v3": "most accurate; several times slower, int8 on a 4 GB GPU",
    "large-v3-turbo": "near large-v3 accuracy at a fraction of its cost",
    "medium": "between small and large",
    "base": "fastest, least accurate",
}


def cached_models() -> list[tuple[str, str]]:
    """(name, label) for every faster-whisper model in the Hugging Face cache."""
    try:
        from faster_whisper.utils import _MODELS
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return [(WHISPER_MODEL, WHISPER_MODEL)]
    seen, out = set(), []
    for name, repo in _MODELS.items():
        if repo in seen or name.endswith(".en"):
            continue
        path = try_to_load_from_cache(repo, "model.bin")
        if not isinstance(path, str):
            continue
        seen.add(repo)
        size = os.path.getsize(os.path.realpath(path)) / 1e9
        note = MODEL_NOTES.get(name, "")
        out.append((name, f"{name} — {size:.1f} GB" + (f" — {note}" if note else "")))
    return out or [(WHISPER_MODEL, WHISPER_MODEL)]


def audio_outputs() -> list[tuple[str, str]]:
    """(node.name, description) for every PipeWire output, via pw-dump."""
    try:
        raw = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=5).stdout
        objects = json.loads(raw)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return []
    out = []
    for obj in objects:
        props = ((obj.get("info") or {}).get("props") or {})
        if props.get("media.class") == "Audio/Sink" and props.get("node.name"):
            out.append((props["node.name"], props.get("node.description") or props["node.name"]))
    return out


def cuda_usable() -> bool:
    try:
        from src.core.device import cuda_available
        return cuda_available()
    except Exception:
        return False


# ── The dialog ─────────────────────────────────────────────────────────

STEPS = [(0.0, "Automatic (1 s on GPU, 2.5 s on CPU)"), (0.5, "0.5 s — fastest, more flicker"),
         (1.0, "1 s"), (1.5, "1.5 s"), (2.0, "2 s"), (3.0, "3 s — for slow machines")]
PRECISIONS = [("auto", "Automatic"), ("float16", "float16 (GPU)"),
              ("int8_float16", "int8 + float16 (GPU, half the memory)"), ("int8", "int8 (CPU)")]
DEVICES = [("auto", "Automatic — GPU when usable"), ("cuda", "GPU (CUDA)"), ("cpu", "CPU")]


class SettingsDialog(QDialog):
    def __init__(self, current: AppSettings, parent: QWidget | None = None,
                 click_through_allowed: bool = True):
        super().__init__(parent)
        self.setWindowTitle("LiveTranscribe settings")
        # The transcript window stays on top of everything; so must its dialog.
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setMinimumWidth(520)
        self._current = current

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        # Speech recognition
        self.model = QComboBox()
        for name, label in cached_models():
            self.model.addItem(label, name)
        if self.model.findData(current.model) < 0:      # a folder chosen earlier
            self.model.addItem(f"{os.path.basename(current.model.rstrip('/'))} — folder", current.model)
        self.model.addItem("Other model folder…", "__browse__")
        self.model.setCurrentIndex(max(0, self.model.findData(current.model)))
        self.model.activated.connect(self._maybe_browse)
        form.addRow("Whisper model", self.model)

        self.device = QComboBox()
        for value, label in DEVICES:
            self.device.addItem(label, value)
        if not cuda_usable():
            self.device.model().item(1).setEnabled(False)
            self.device.setItemText(1, "GPU (CUDA) — not available now")
        self.device.setCurrentIndex(max(0, self.device.findData(current.device)))
        form.addRow("Run on", self.device)

        self.precision = QComboBox()
        for value, label in PRECISIONS:
            self.precision.addItem(label, value)
        self.precision.setCurrentIndex(max(0, self.precision.findData(current.precision)))
        form.addRow("Precision", self.precision)

        self.step = QComboBox()
        for value, label in STEPS:
            self.step.addItem(label, value)
        self.step.setCurrentIndex(max(0, self.step.findData(current.step_s)))
        form.addRow("Update every", self.step)

        self.sink = QComboBox()
        self.sink.addItem("Default output — follows when it changes", "")
        for name, description in audio_outputs():
            self.sink.addItem(description, name)
        if current.sink and self.sink.findData(current.sink) < 0:
            self.sink.addItem(f"{current.sink} (not connected now)", current.sink)
        self.sink.setCurrentIndex(max(0, self.sink.findData(current.sink)))
        form.addRow("Listen to", self.sink)

        self.arabic_only = QCheckBox("Ignore speech that is not Arabic")
        self.arabic_only.setChecked(current.arabic_only)
        form.addRow("", self.arabic_only)

        # The window
        self.font_px = QSpinBox()
        self.font_px.setRange(OVERLAY_FONT_MIN_PX, OVERLAY_FONT_MAX_PX)
        self.font_px.setSuffix(" px")
        self.font_px.setValue(current.font_px)
        form.addRow("Text size", self.font_px)

        opacity_row = QHBoxLayout()
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(20, 100)
        self.opacity.setValue(current.opacity)
        self.opacity_label = QLabel(f"{current.opacity}%")
        self.opacity.valueChanged.connect(lambda v: self.opacity_label.setText(f"{v}%"))
        opacity_row.addWidget(self.opacity)
        opacity_row.addWidget(self.opacity_label)
        form.addRow("Background", opacity_row)

        self.show_tentative = QCheckBox("Show words that may still change (dimmed)")
        self.show_tentative.setChecked(current.show_tentative)
        form.addRow("", self.show_tentative)

        self.click_through = QCheckBox("Click-through — the window ignores the mouse")
        self.click_through.setChecked(current.click_through and click_through_allowed)
        self.click_through.setEnabled(click_through_allowed)
        self.click_through.setToolTip("Turn it off again from the tray icon's menu"
                                      if click_through_allowed else
                                      "Needs a tray icon to turn it back off")
        form.addRow("", self.click_through)

        note = QLabel("A new model, device or precision reloads the model; a new source or "
                      "interval restarts listening. Saved settings are used at every start.")
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(mid);")

        # Long model descriptions must not squeeze the labels: the drop-downs
        # size to a fixed number of characters and elide the rest.
        for combo in (self.model, self.device, self.precision, self.step, self.sink):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(34)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)

    def _maybe_browse(self, index: int):
        if self.model.itemData(index) != "__browse__":
            return
        folder = QFileDialog.getExistingDirectory(self, "A CTranslate2 Whisper model folder")
        if folder and os.path.exists(os.path.join(folder, "model.bin")):
            self.model.insertItem(index, f"{os.path.basename(folder)} — folder", folder)
            self.model.setCurrentIndex(index)
        else:
            self.model.setCurrentIndex(max(0, self.model.findData(self._current.model)))

    def result_settings(self) -> AppSettings:
        return AppSettings(
            model=self.model.currentData(),
            device=self.device.currentData(),
            precision=self.precision.currentData(),
            step_s=float(self.step.currentData()),
            sink=self.sink.currentData(),
            font_px=self.font_px.value(),
            opacity=self.opacity.value(),
            show_tentative=self.show_tentative.isChecked(),
            arabic_only=self.arabic_only.isChecked(),
            click_through=self.click_through.isChecked(),
        )
