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

from PyQt6.QtCore import QSettings, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QToolButton, QVBoxLayout, QWidget,
)

from src.config import (
    ASR_STEP_S_CPU, ASR_STEP_S_GPU, DEVICE, OVERLAY_FONT_MAX_PX, OVERLAY_FONT_MIN_PX,
    OVERLAY_FONT_PX, OVERLAY_OPACITY, WHISPER_MODEL,
)
from src.ui import icons
from src.ui.theme import TEXT, TEXT_2, dialog_palette, dialog_stylesheet
from src.ui.translate import LANGUAGES, MODES, default_target

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
    translate: str = "button"         # off | button | auto — Google Translate on selection
    translate_to: str = default_target()

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
    """(name, size and note) for every faster-whisper model in the Hugging Face cache."""
    try:
        from faster_whisper.utils import _MODELS
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return [(WHISPER_MODEL, "")]
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
        out.append((name, f"{size:.1f} GB" + (f" — {note}" if note else "")))
    return out or [(WHISPER_MODEL, "")]


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

# Each choice is (value, label, hint); the hint shows under the drop-down.
STEPS = [(0.0, "Automatic", f"{ASR_STEP_S_GPU:g} s on the GPU, {ASR_STEP_S_CPU:g} s on the CPU"),
         (0.5, "0.5 s", "Fastest; the newest words change more often"),
         (1.0, "1 s", ""), (1.5, "1.5 s", ""), (2.0, "2 s", ""),
         (3.0, "3 s", "For slow machines")]
PRECISIONS = [("auto", "Automatic", "float16 on the GPU (int8 + float16 for large models), "
                                    "int8 on the CPU"),
              ("float16", "float16", "GPU; on the CPU it runs as int8"),
              ("int8_float16", "int8 + float16", "GPU, half the memory; on the CPU it runs as int8"),
              ("int8", "int8", "The CPU's precision; runs on the GPU too")]
DEVICES = [("auto", "Automatic", "The GPU when it is usable, else the CPU"),
           ("cuda", "GPU (CUDA)", ""),
           ("cpu", "CPU", "Several times slower than a GPU")]
TRANSLATE_LABELS = {"off": "Off", "button": "Button beside selected text",
                    "auto": "As soon as text is selected"}
LABEL_PX = 104          # the label column, the same width in every section


class _Section:
    """Rows of one section: a label, its field, and under the field a hint when it has one."""

    def __init__(self, grid: QGridLayout):
        self.grid = grid
        self.row = 0

    def add(self, label: str, field: QWidget, hint: QLabel | None = None):
        if label:
            self.grid.addWidget(QLabel(label), self.row, 0)
        self.grid.addWidget(field, self.row, 1)
        self.row += 1
        if hint is not None:
            self.grid.addWidget(hint, self.row, 1)
            self.row += 1


def _hint(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "hint")
    label.setWordWrap(True)
    return label


class SettingsDialog(QDialog):
    # The look, as it is being chosen: the window shows it before Save.
    appearance_changed = pyqtSignal(object)

    def __init__(self, current: AppSettings, parent: QWidget | None = None,
                 click_through_allowed: bool = True):
        super().__init__(parent)
        self.setWindowTitle("LiveTranscribe settings")
        # The transcript window stays on top of everything; so must its dialog.
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        dialog_palette(self)
        self.setStyleSheet(dialog_stylesheet())
        self.setMinimumWidth(500)
        self._current = current

        self._body = QVBoxLayout(self)
        self._body.setContentsMargins(20, 16, 20, 16)
        self._body.setSpacing(0)

        # Speech recognition
        recognition = self._section("Recognition")
        self.model = self._combo([])
        for name, detail in cached_models():
            self._add_choice(self.model, name, name, detail)
        if self.model.findData(current.model) < 0:      # a folder chosen earlier
            self._add_choice(self.model, current.model,
                             os.path.basename(current.model.rstrip("/")), current.model)
        self._add_choice(self.model, "__browse__", "Other model folder…", "")
        self.model.activated.connect(self._maybe_browse)
        self._select(self.model, current.model)
        recognition.add("Model", self.model, self._hint_for(self.model))

        self.device = self._combo(DEVICES)
        if not cuda_usable():
            self.device.model().item(1).setEnabled(False)
            self.device.setItemText(1, "GPU (CUDA) — not available now")
            self.device.setItemData(0, "No usable GPU now, so the CPU", Qt.ItemDataRole.ToolTipRole)
        self._select(self.device, current.device)
        # Said under the field only when it matters: no GPU to run on.
        recognition.add("Run on", self.device,
                        None if cuda_usable() else self._hint_for(self.device))
        self._tip_for(self.device)

        # Audio
        audio = self._section("Audio")
        self.sink = self._combo([("", "Default output", "Follows the default output when it changes")])
        for name, description in audio_outputs():
            self._add_choice(self.sink, name, description, "")
        if current.sink and self.sink.findData(current.sink) < 0:
            self._add_choice(self.sink, current.sink, f"{current.sink} (not connected now)", "")
        self._select(self.sink, current.sink)
        audio.add("Listen to", self.sink)
        self._tip_for(self.sink)
        self.arabic_only = QCheckBox("Ignore speech that is not Arabic")
        self.arabic_only.setChecked(current.arabic_only)
        audio.add("", self.arabic_only)

        # The window
        window = self._section("Window")
        self.font_px = QSlider(Qt.Orientation.Horizontal)
        self.font_px.setRange(OVERLAY_FONT_MIN_PX, OVERLAY_FONT_MAX_PX)
        self.font_px.setValue(current.font_px)
        window.add("Text size", self._with_value(self.font_px, "{} px"))
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(20, 100)
        self.opacity.setValue(current.opacity)
        window.add("Background", self._with_value(self.opacity, "{}%"))
        self.show_tentative = QCheckBox("Show words that may still change, dimmed")
        self.show_tentative.setChecked(current.show_tentative)
        window.add("", self.show_tentative)
        self.click_through = QCheckBox("Click-through: clicks pass through the text")
        self.click_through.setChecked(current.click_through and click_through_allowed)
        self.click_through.setEnabled(click_through_allowed)
        self.click_through.setToolTip("The top bar stays clickable; turn it off with its lock button."
                                      if click_through_allowed else
                                      "Needs a tray icon to turn it back off")
        window.add("", self.click_through)

        # Google Translate
        translation = self._section("Translation")
        self.translate = self._combo([(value, TRANSLATE_LABELS.get(value, label), "")
                                      for value, label in MODES])
        self._select(self.translate, current.translate)
        translation.add("Translate", self.translate)
        self.translate_to = self._combo([(code, name, "") for code, name in LANGUAGES])
        self._select(self.translate_to, current.translate_to)
        translation.add("Into", self.translate_to,
                        _hint("Only text you select is sent to Google."))

        # Advanced: tuning most people never touch, folded away unless in use.
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setObjectName("disclosure")
        self.advanced_toggle.setText("Advanced")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_toggle.setIconSize(QSize(14, 14))
        self.advanced_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._heading(self.advanced_toggle)
        self.advanced = QWidget()
        advanced = _Section(self._grid(self.advanced))
        self.precision = self._combo(PRECISIONS)
        self._select(self.precision, current.precision)
        advanced.add("Precision", self.precision)
        self._tip_for(self.precision)
        self.step = self._combo(STEPS)
        self._select(self.step, current.step_s)
        advanced.add("Update every", self.step)
        self._tip_for(self.step)
        self._body.addWidget(self.advanced)
        self.advanced_toggle.toggled.connect(self._show_advanced)
        in_use = current.precision != "auto" or current.step_s != 0.0
        self.advanced_toggle.setChecked(in_use)
        self._show_advanced(in_use)

        # What Save will do, then the buttons.
        self.note = QLabel("")
        self.note.setProperty("role", "note")
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Save")
        save.setProperty("primary", True)
        save.setDefault(True)
        save.clicked.connect(self.accept)
        footer = QHBoxLayout()
        footer.setSpacing(8)
        footer.addWidget(self.note, 1)
        footer.addWidget(cancel)
        footer.addWidget(save)
        self._body.addSpacing(24)
        self._body.addLayout(footer)
        save.setFocus()             # no field lit up as if being edited

        # Long output names must not widen the dialog: the drop-downs size to
        # a fixed number of characters and elide the rest.
        for combo in (self.model, self.device, self.precision, self.step, self.sink,
                      self.translate, self.translate_to):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(28)
            combo.currentIndexChanged.connect(self._changed)
        for box in (self.arabic_only, self.click_through):
            box.toggled.connect(self._changed)
        self.font_px.valueChanged.connect(self._appearance)
        self.opacity.valueChanged.connect(self._appearance)
        self.show_tentative.toggled.connect(self._appearance)

    # ── Building ───────────────────────────────────────────────────────

    def _grid(self, parent: QWidget | None = None) -> QGridLayout:
        grid = QGridLayout(parent)
        if parent is not None:
            grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)
        grid.setColumnMinimumWidth(0, LABEL_PX)
        grid.setColumnStretch(1, 1)
        return grid

    def _heading(self, title: QWidget):
        """A section's title, and a rule from it to the right edge."""
        if self._body.count():
            self._body.addSpacing(20)
        rule = QFrame()
        rule.setProperty("role", "rule")
        rule.setFixedHeight(1)
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(title)
        row.addWidget(rule, 1, Qt.AlignmentFlag.AlignVCenter)
        self._body.addLayout(row)
        self._body.addSpacing(12)

    def _section(self, title: str) -> _Section:
        heading = QLabel(title)
        heading.setProperty("role", "section")
        self._heading(heading)
        grid = self._grid()
        self._body.addLayout(grid)
        return _Section(grid)

    def _combo(self, choices) -> QComboBox:
        combo = QComboBox()
        for value, label, hint in choices:
            self._add_choice(combo, value, label, hint)
        return combo

    @staticmethod
    def _add_choice(combo: QComboBox, value, label: str, hint: str):
        combo.addItem(label, value)
        if hint:
            combo.setItemData(combo.count() - 1, hint, Qt.ItemDataRole.ToolTipRole)

    @staticmethod
    def _select(combo: QComboBox, value):
        combo.setCurrentIndex(max(0, combo.findData(value)))

    @staticmethod
    def _hint_for(combo: QComboBox) -> QLabel:
        """A hint that follows the drop-down: the chosen item's own, or nothing."""
        hint = _hint()

        def follow():
            text = combo.currentData(Qt.ItemDataRole.ToolTipRole) or ""
            hint.setText(text)
            hint.setVisible(bool(text))
        combo.currentIndexChanged.connect(follow)
        follow()
        return hint

    @staticmethod
    def _tip_for(combo: QComboBox):
        """The chosen item's hint as the drop-down's tooltip, not a line under it."""
        def follow():
            combo.setToolTip(combo.currentData(Qt.ItemDataRole.ToolTipRole) or "")
        combo.currentIndexChanged.connect(follow)
        follow()

    @staticmethod
    def _with_value(slider: QSlider, fmt: str) -> QWidget:
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        value = QLabel(fmt.format(slider.value()))
        value.setProperty("role", "value")
        value.setMinimumWidth(44)
        value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        slider.valueChanged.connect(lambda v: value.setText(fmt.format(v)))
        row.addWidget(slider, 1)
        row.addWidget(value)
        return box

    def _show_advanced(self, on: bool):
        self.advanced.setVisible(on)
        self.advanced_toggle.setIcon(icons.icon(icons.CHEVRON_DOWN if on else icons.CHEVRON_RIGHT,
                                                14, rest=TEXT_2, hover=TEXT))
        if self.isVisible():
            QTimer.singleShot(0, self.adjustSize)

    # ── Changes ────────────────────────────────────────────────────────

    def _changed(self, *_):
        needs = self._current.needs(self.result_settings())
        self.note.setText({"engine": "Saving reloads the model",
                           "pipeline": "Saving restarts listening"}.get(needs, ""))

    def _appearance(self, *_):
        self._changed()
        self.appearance_changed.emit(self.result_settings())

    def _maybe_browse(self, index: int):
        if self.model.itemData(index) != "__browse__":
            return
        folder = QFileDialog.getExistingDirectory(self, "A CTranslate2 Whisper model folder")
        if folder and os.path.exists(os.path.join(folder, "model.bin")):
            self.model.insertItem(index, os.path.basename(folder), folder)
            self.model.setItemData(index, folder, Qt.ItemDataRole.ToolTipRole)
            self.model.setCurrentIndex(index)
        else:
            self._select(self.model, self._current.model)

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
            translate=self.translate.currentData(),
            translate_to=self.translate_to.currentData(),
        )
