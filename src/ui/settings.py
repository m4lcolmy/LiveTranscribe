"""Settings: what the user chose, kept between sessions, and the dialog to choose it.

Everything here is saved with QSettings (~/.config/LiveTranscribe/LiveTranscribe.conf)
the moment the dialog is saved, and read back at the next start, so the app
opens the way it was left. A command-line flag overrides a saved value for that
run only; it is never written back.

The model list is the Whisper models the app knows (src/core/models.py) and
any other in the Hugging Face cache. One that is not on this computer says so
and its size, and shows a Download button under it, then a progress bar; the
offline translator the same, under Translation. Nothing downloads unless that
button is pressed, and Save waits until the chosen model is here — a model
that is not there would only be an error. Any other CTranslate2 Whisper folder
can be picked by hand. The one model that is not local is Deepgram, with the
user's own API key. That key is not a setting: it lives in the system keyring
(src/core/keystore.py), and the dialog can replace or remove it but never
shows it again.
"""

import json
import os
import subprocess
import threading
from dataclasses import asdict, dataclass, fields

from PyQt6.QtCore import QPoint, QSettings, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSlider, QToolButton, QVBoxLayout, QWidget,
)

from src.config import (
    ASR_STEP_S_CPU, ASR_STEP_S_GPU, DEEPGRAM_LANGUAGE, DEEPGRAM_MODEL_NAME, DEVICE,
    OVERLAY_FONT_MAX_PX, OVERLAY_FONT_MIN_PX, OVERLAY_FONT_PX, OVERLAY_OPACITY, WHISPER_MODEL,
)
from src.core import keystore, models
from src.ui import icons
from src.ui.downloads import DownloadRow
from src.ui.theme import TEXT, TEXT_2, dialog_palette, dialog_stylesheet
from src.ui.translate import ENGINES, LANGUAGES, MODES, default_target

SETTINGS_ORG = "LiveTranscribe"
SETTINGS_APP = "LiveTranscribe"


def open_store() -> QSettings:
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


@dataclass
class AppSettings:
    model: str = WHISPER_MODEL        # a size name from the cache, a model folder, or "deepgram"
    device: str = DEVICE              # auto | cuda | cpu
    precision: str = "auto"           # auto | float16 | int8_float16 | int8
    step_s: float = 0.0               # seconds between passes; 0 = by device
    sink: str = ""                    # an output's node.name; "" follows the default
    font_px: int = OVERLAY_FONT_PX
    opacity: int = OVERLAY_OPACITY    # background, percent
    show_tentative: bool = True       # show words that may still change
    arabic_only: bool = True          # drop speech Whisper hears as another language
    click_through: bool = False
    translate: str = "button"         # off | button | auto — translate selected text
    translate_to: str = default_target()
    translate_with: str = "google"    # google | offline (NLLB-200, src/core/nllb.py)
    deepgram_language: str = DEEPGRAM_LANGUAGE   # ar, or one of Nova-3's Arabic dialects

    @classmethod
    def load(cls, store: QSettings) -> "AppSettings":
        _move_key_to_keyring(store)
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
    DEEPGRAM = ("deepgram_language",)       # an engine change only while Deepgram is in use
    PIPELINE = ("step_s", "sink", "arabic_only")

    def needs(self, other: "AppSettings") -> str:
        """'engine', 'pipeline' or 'window': the least that must restart to apply `other`."""
        engine = self.ENGINE + (self.DEEPGRAM if other.model == DEEPGRAM_MODEL_NAME else ())
        if any(getattr(self, k) != getattr(other, k) for k in engine):
            return "engine"
        if any(getattr(self, k) != getattr(other, k) for k in self.PIPELINE):
            return "pipeline"
        return "window"


def _move_key_to_keyring(store: QSettings):
    """A key the first Deepgram version saved here, in plain text: into the keyring, out of the file.

    A key already in the keyring is the newer one and is kept. If the keyring
    will not take it, the old key stays where it was rather than being lost —
    Deepgram shows a key only once, when it is made.
    """
    if not store.contains("settings/deepgram_key"):
        return
    old = str(store.value("settings/deepgram_key", "") or "").strip()
    if old and not keystore.load():
        try:
            keystore.save(old)
        except keystore.KeystoreError:
            return
    store.remove("settings/deepgram_key")
    store.sync()


# ── What there is to choose from ───────────────────────────────────────

DEEPGRAM_HINT = "Nova-3, in the cloud: speech is sent to Deepgram"
# Nova-3's Arabic: general, and the dialects it was trained for (2026-10).
DEEPGRAM_LANGUAGES = [
    ("ar", "Arabic (general)"), ("ar-EG", "Egyptian"), ("ar-SA", "Saudi"), ("ar-AE", "Emirati"),
    ("ar-QA", "Qatari"), ("ar-KW", "Kuwaiti"), ("ar-IQ", "Iraqi"), ("ar-SY", "Syrian"),
    ("ar-LB", "Lebanese"), ("ar-JO", "Jordanian"), ("ar-PS", "Palestinian"), ("ar-SD", "Sudanese"),
    ("ar-TD", "Chadian"), ("ar-MA", "Moroccan"), ("ar-DZ", "Algerian"), ("ar-TN", "Tunisian"),
    ("ar-IR", "Iranian Arabic"),
]


def whisper_models() -> list[tuple[str, str, str]]:
    """(name, label, hint): the models the app offers, then any other faster-whisper
    model in the Hugging Face cache. One not downloaded says so in its label."""
    out, repos = [], set()
    for m in models.WHISPER:
        repos.add(m.repo)
        here = models.is_downloaded(m)
        label = m.name if here else f"{m.name}  ·  download {m.size}"
        out.append((m.name, label, f"{m.size} — {m.note}" if m.note else m.size))
    try:
        from faster_whisper.utils import _MODELS
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return out
    for name, repo in _MODELS.items():
        if repo in repos or name.endswith(".en"):
            continue
        path = try_to_load_from_cache(repo, "model.bin")
        if not isinstance(path, str):
            continue
        repos.add(repo)
        size = os.path.getsize(os.path.realpath(path))
        out.append((name, name, models.size_text(size)))
    return out


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

    def add(self, label: str, field: QWidget, hint: QLabel | None = None) -> list[QWidget]:
        """The row's widgets, so it can be hidden when it does not apply."""
        row = [field]
        if label:
            row.insert(0, QLabel(label))
            self.grid.addWidget(row[0], self.row, 0)
        self.grid.addWidget(field, self.row, 1)
        self.row += 1
        if hint is not None:
            self.grid.addWidget(hint, self.row, 1)
            self.row += 1
            row.append(hint)
        return row


def _hint(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "hint")
    label.setWordWrap(True)
    return label


class SettingsDialog(QDialog):
    """Returns the chosen settings; the Deepgram key, which is not one, it saves itself on Save.

    `key_changed` then says whether the key was replaced or removed.
    """

    # The look, as it is being chosen: the window shows it before Save.
    appearance_changed = pyqtSignal(object)
    # The Test button's answer, from its thread: (text, the key works).
    key_checked = pyqtSignal(str, bool)

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
        for name, label, detail in whisper_models():
            self._add_choice(self.model, name, label, detail)
        self._add_choice(self.model, DEEPGRAM_MODEL_NAME, "Deepgram (cloud)", DEEPGRAM_HINT)
        if self.model.findData(current.model) < 0:      # a folder chosen earlier
            self._add_choice(self.model, current.model,
                             os.path.basename(current.model.rstrip("/")), current.model)
        self._add_choice(self.model, "__browse__", "Other model folder…", "")
        self.model.activated.connect(self._maybe_browse)
        self._select(self.model, current.model)
        recognition.add("Model", self.model, self._hint_for(self.model))
        self.model_download = DownloadRow()
        self.model_download.ready.connect(self._downloaded)
        recognition.add("", self.model_download)

        self.device = self._combo(DEVICES)
        if not cuda_usable():
            self.device.model().item(1).setEnabled(False)
            self.device.setItemText(1, "GPU (CUDA) — not available now")
            self.device.setItemData(0, "No usable GPU now, so the CPU", Qt.ItemDataRole.ToolTipRole)
        self._select(self.device, current.device)
        # Said under the field only when it matters: no GPU to run on.
        self._local_rows = recognition.add("Run on", self.device,
                                           None if cuda_usable() else self._hint_for(self.device))
        self._tip_for(self.device)

        # Deepgram's rows, in place of the local model's. The key field is
        # write-only: it starts empty, and a saved key shows as its last four
        # characters in the placeholder — the key itself is never read into it.
        self.key_changed = False
        self._remove_key = False
        saved = keystore.load()
        self._saved_ending = keystore.ending(saved) if saved else ""
        del saved
        self.deepgram_key = QLineEdit()
        self.deepgram_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.show_key = self.deepgram_key.addAction(icons.icon(icons.EYE, 16),
                                                    QLineEdit.ActionPosition.TrailingPosition)
        self.show_key.setCheckable(True)
        self.show_key.setToolTip("Show what is typed here")
        self.show_key.toggled.connect(self._reveal_key)
        self.test_key = QPushButton("Test")
        self.test_key.clicked.connect(self._test_key)
        self.forget_key = QPushButton("Remove")
        self.forget_key.setToolTip("Remove the saved key from the system keyring")
        self.forget_key.clicked.connect(self._forget_key)
        for button in (self.test_key, self.forget_key):     # small: the field needs the room
            button.setStyleSheet("padding: 5px 10px; min-width: 0;")
        key_row = QWidget()
        row = QHBoxLayout(key_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(self.deepgram_key, 1)
        row.addWidget(self.test_key)
        row.addWidget(self.forget_key)
        self.key_hint = _hint()
        self.key_checked.connect(self._key_checked)
        self._cloud_rows = recognition.add("API key", key_row, self.key_hint)
        self._show_key_state()
        self.deepgram_language = self._combo([(code, name, "") for code, name in DEEPGRAM_LANGUAGES])
        self._select(self.deepgram_language, current.deepgram_language)
        self._cloud_rows += recognition.add("Dialect", self.deepgram_language)

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

        # Translation: Google, or NLLB on this computer
        translation = self._section("Translation")
        self.translate = self._combo([(value, TRANSLATE_LABELS.get(value, label), "")
                                      for value, label in MODES])
        self._select(self.translate, current.translate)
        translation.add("Translate", self.translate)
        self.translate_with = self._combo(ENGINES)
        self._select(self.translate_with, current.translate_with)
        self._label_engines()
        translation.add("With", self.translate_with, self._hint_for(self.translate_with))
        self.translate_download = DownloadRow()
        self.translate_download.ready.connect(self._downloaded)
        translation.add("", self.translate_download)
        self.translate_to = self._combo([(code, name, "") for code, name in LANGUAGES])
        self._select(self.translate_to, current.translate_to)
        translation.add("Into", self.translate_to)

        # Advanced: tuning most people never touch, folded away unless in use.
        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setObjectName("disclosure")
        self.advanced_toggle.setText("Advanced")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_toggle.setIconSize(QSize(14, 14))
        self.advanced_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.advanced_heading = self._heading(self.advanced_toggle)
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
        self._show_engine_rows()

        # What Save will do, then the buttons.
        self.note = QLabel("")
        self.note.setProperty("role", "note")
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Save")
        save.setProperty("primary", True)
        save.setDefault(True)
        save.clicked.connect(self.accept)
        self.save_button = save
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
                      self.translate, self.translate_with, self.translate_to,
                      self.deepgram_language):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(28)
            combo.currentIndexChanged.connect(self._changed)
        for box in (self.arabic_only, self.click_through):
            box.toggled.connect(self._changed)
        self.deepgram_key.textChanged.connect(self._changed)
        self.deepgram_key.textChanged.connect(self._show_key_state)
        self.model.currentIndexChanged.connect(self._show_engine_rows)
        self.model.currentIndexChanged.connect(self._show_downloads)
        self.translate.currentIndexChanged.connect(self._show_downloads)
        self.translate_with.currentIndexChanged.connect(self._show_downloads)
        self._show_downloads()
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

    def _heading(self, title: QWidget) -> QWidget:
        """A section's title, and a rule from it to the right edge — one widget, spacing and all."""
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        if self._body.count():
            column.addSpacing(20)
        rule = QFrame()
        rule.setProperty("role", "rule")
        rule.setFixedHeight(1)
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(title)
        row.addWidget(rule, 1, Qt.AlignmentFlag.AlignVCenter)
        column.addLayout(row)
        column.addSpacing(12)
        self._body.addWidget(box)
        return box

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

    @property
    def _cloud(self) -> bool:
        return self.model.currentData() == DEEPGRAM_MODEL_NAME

    def _show_advanced(self, on: bool):
        middle = self.frameGeometry().center()
        self.advanced.setVisible(on and not self._cloud)
        self.advanced_toggle.setIcon(icons.icon(icons.CHEVRON_DOWN if on else icons.CHEVRON_RIGHT,
                                                14, rest=TEXT_2, hover=TEXT))
        if self.isVisible():
            QTimer.singleShot(0, lambda: self._refit(middle))

    def _refit(self, middle: QPoint):
        """Fit to the rows now shown, about the middle it had before they
        changed: grown down from the top edge it slips under the transcript window."""
        self.adjustSize()
        frame = self.frameGeometry()
        frame.moveCenter(middle)
        screen = self.screen()
        if screen is not None:
            room = screen.availableGeometry()
            frame.moveBottom(min(frame.bottom(), room.bottom()))
            frame.moveTop(max(frame.top(), room.top()))
        self.move(frame.topLeft())

    def _show_engine_rows(self, *_):
        """Deepgram's key and dialect, or the local model's device and Advanced tuning."""
        middle = self.frameGeometry().center()
        cloud = self._cloud
        for widget in self._cloud_rows:
            widget.setVisible(cloud)
        for widget in self._local_rows:
            empty_hint = widget.property("role") == "hint" and not widget.text()
            widget.setVisible(not cloud and not empty_hint)
        self.advanced_heading.setVisible(not cloud)
        self.advanced.setVisible(not cloud and self.advanced_toggle.isChecked())
        self.arabic_only.setEnabled(not cloud)
        self.arabic_only.setToolTip("Whisper only: it relies on Whisper's own language detection"
                                    if cloud else "")
        if self.isVisible():
            QTimer.singleShot(0, lambda: self._refit(middle))

    # ── Downloads ──────────────────────────────────────────────────────

    def _label_engines(self):
        """Offline says its size while it is not on this computer."""
        i = self.translate_with.findData("offline")
        name = dict((v, label) for v, label, _ in ENGINES)["offline"]
        here = models.is_downloaded(models.NLLB)
        self.translate_with.setItemText(i, name if here else f"{name}  ·  download {models.NLLB.size}")

    def _show_downloads(self, *_):
        """A download row under each choice that is not on this computer."""
        middle = self.frameGeometry().center()
        self.model_download.show_for(self.model.currentData())
        on = self.translate.currentData() != "off"
        self.translate_with.setEnabled(on)
        self.translate_to.setEnabled(on)
        self.translate_download.show_for(
            models.NLLB_NAME if on and self.translate_with.currentData() == "offline" else None)
        self._changed()
        if self.isVisible():
            QTimer.singleShot(0, lambda: self._refit(middle))

    def _downloaded(self, name: str):
        """A model arrived: its label loses the size, and Save may be pressed."""
        if name == models.NLLB_NAME:
            self._label_engines()
        else:
            i = self.model.findData(name)
            if i >= 0:
                self.model.setItemText(i, name)
        self._changed()

    def missing(self) -> str | None:
        """What must be downloaded before Save: the chosen model, or the offline translator."""
        if self.model_download.missing:
            return self.model.currentData()
        if self.translate_download.missing:
            return "the offline translation model"
        return None

    # ── The key ────────────────────────────────────────────────────────

    def key_change(self) -> str | None:
        """'replace' (a key is typed), 'remove', or None."""
        if self.deepgram_key.text().strip():
            return "replace"
        return "remove" if self._remove_key else None

    def _show_key_state(self, *_):
        """The placeholder says what is saved; the hint, what to do about it."""
        saved = bool(self._saved_ending) and not self._remove_key
        self.forget_key.setVisible(saved)
        if not keystore.available():
            self.deepgram_key.setEnabled(False)
            self.deepgram_key.setPlaceholderText("No system keyring")
            hint = "Start the app with DEEPGRAM_API_KEY set instead"
        elif saved:
            self.deepgram_key.setPlaceholderText(f"Saved key {self._saved_ending}")
            hint = "Paste a new key to replace it"
        else:
            self.deepgram_key.setPlaceholderText("Paste your Deepgram API key")
            hint = ("The saved key is removed when you save" if self._remove_key
                    else "From console.deepgram.com → API Keys")
        if keystore.from_environment():
            hint = "DEEPGRAM_API_KEY is set, and used instead"
        if self.deepgram_key.text().strip():
            hint = "Saved in the system keyring when you save"
        self.key_hint.setText(hint)

    def _forget_key(self):
        self._remove_key = True
        self.deepgram_key.clear()
        self._show_key_state()
        self._changed()

    def _reveal_key(self, shown: bool):
        self.deepgram_key.setEchoMode(QLineEdit.EchoMode.Normal if shown
                                      else QLineEdit.EchoMode.Password)
        self.show_key.setIcon(icons.icon(icons.EYE_OFF if shown else icons.EYE, 16))
        self.show_key.setToolTip("Hide what is typed here" if shown else "Show what is typed here")

    def _test_key(self):
        """The typed key, else the one in use — read in the thread, never shown."""
        from src.audio.deepgram import DeepgramError, check_key
        typed = self.deepgram_key.text().strip()
        removed = self._remove_key
        self.test_key.setEnabled(False)
        self.key_hint.setText("Asking Deepgram…")

        def ask():
            key = typed or (keystore.from_environment() if removed else keystore.key())
            try:
                check_key(key)
                answer = (f"✓ Deepgram accepts {'this' if typed else 'the saved'} key", True)
            except DeepgramError as e:
                answer = (f"✕ {e}", False)
            try:
                self.key_checked.emit(*answer)
            except RuntimeError:            # the dialog closed meanwhile
                pass
        threading.Thread(target=ask, name="deepgram-key", daemon=True).start()

    def _key_checked(self, text: str, _ok: bool):
        self.key_hint.setText(text)
        self.test_key.setEnabled(True)

    def accept(self):
        """Save or remove the key first; if the keyring refuses, say so and stay open."""
        if self.missing() is not None:
            return                          # Enter pressed while Save is greyed out
        change = self.key_change()
        try:
            if change == "replace":
                keystore.save(self.deepgram_key.text())
            elif change == "remove":
                keystore.remove()
        except keystore.KeystoreError as e:
            self.key_hint.setText(f"✕ {e}")
            return
        self.key_changed = change is not None
        self.deepgram_key.clear()
        super().accept()

    # ── Changes ────────────────────────────────────────────────────────

    def _changed(self, *_):
        if not hasattr(self, "save_button"):
            return                          # still being built
        missing = self.missing()
        self.save_button.setEnabled(missing is None)
        if missing is not None:
            self.note.setText(f"Download {missing} to save, or choose another")
            return
        needs = self._current.needs(self.result_settings())
        if self._cloud and self.key_change() is not None:
            needs = "engine"
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
            translate_with=self.translate_with.currentData(),
            deepgram_language=self.deepgram_language.currentData(),
        )
