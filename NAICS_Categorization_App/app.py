"""
app.py - NAICS Classifier — unified Train + Predict application.

Run with:
    python app.py
"""

import html
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def _setup_qt_compat():
    """
    Wire PyQt6 as a transparent drop-in for PyQt5.

    PyQt6 has pip wheels for every platform (including ARM64 Linux).
    The two API differences we patch here:
      • Qt enum values moved into sub-namespaces  (Qt.Alignment → Qt.AlignmentFlag.*)
      • QHeaderView resize-mode constants moved    (QHeaderView.Stretch → .ResizeMode.Stretch)
    exec_() and Qt.AA_* are handled inline where they're called.
    """
    import importlib.util
    if importlib.util.find_spec("PyQt5") is not None:
        return  # native PyQt5 installed — nothing to do

    import sys as _sys
    from PyQt6 import QtCore, QtGui, QtWidgets

    _sys.modules.setdefault("PyQt5", _sys.modules["PyQt6"])
    _sys.modules["PyQt5.QtCore"]    = QtCore
    _sys.modules["PyQt5.QtGui"]     = QtGui
    _sys.modules["PyQt5.QtWidgets"] = QtWidgets

    Qt = QtCore.Qt
    _aliases = {
        # scroll bars
        "ScrollBarAlwaysOff":  Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        "ScrollBarAlwaysOn":   Qt.ScrollBarPolicy.ScrollBarAlwaysOn,
        "ScrollBarAsNeeded":   Qt.ScrollBarPolicy.ScrollBarAsNeeded,
        # cursors
        "ArrowCursor":         Qt.CursorShape.ArrowCursor,
        "WaitCursor":          Qt.CursorShape.WaitCursor,
        "PointingHandCursor":  Qt.CursorShape.PointingHandCursor,
        "WhatsThisCursor":     Qt.CursorShape.WhatsThisCursor,
        # focus
        "NoFocus":             Qt.FocusPolicy.NoFocus,
        "StrongFocus":         Qt.FocusPolicy.StrongFocus,
        # item flags
        "ItemIsEditable":      Qt.ItemFlag.ItemIsEditable,
        "ItemIsSelectable":    Qt.ItemFlag.ItemIsSelectable,
        "ItemIsEnabled":       Qt.ItemFlag.ItemIsEnabled,
        # alignment
        "AlignLeft":           Qt.AlignmentFlag.AlignLeft,
        "AlignRight":          Qt.AlignmentFlag.AlignRight,
        "AlignCenter":         Qt.AlignmentFlag.AlignCenter,
        "AlignHCenter":        Qt.AlignmentFlag.AlignHCenter,
        "AlignVCenter":        Qt.AlignmentFlag.AlignVCenter,
        "AlignTop":            Qt.AlignmentFlag.AlignTop,
        "AlignBottom":         Qt.AlignmentFlag.AlignBottom,
        # orientation
        "Horizontal":          Qt.Orientation.Horizontal,
        "Vertical":            Qt.Orientation.Vertical,
        # text
        "RichText":            Qt.TextFormat.RichText,
        "PlainText":           Qt.TextFormat.PlainText,
        "TextWrapAnywhere":    Qt.TextFlag.TextWrapAnywhere,
        # image
        "KeepAspectRatio":     Qt.AspectRatioMode.KeepAspectRatio,
        "SmoothTransformation":Qt.TransformationMode.SmoothTransformation,
    }
    for name, val in _aliases.items():
        setattr(Qt, name, val)

    # QHeaderView resize modes moved to QHeaderView.ResizeMode.*
    QHV = QtWidgets.QHeaderView
    QHV.Stretch          = QHV.ResizeMode.Stretch
    QHV.ResizeToContents = QHV.ResizeMode.ResizeToContents
    QHV.Interactive      = QHV.ResizeMode.Interactive
    QHV.Fixed            = QHV.ResizeMode.Fixed

    # QFrame shape/shadow constants moved to QFrame.Shape.* / QFrame.Shadow.*
    QF = QtWidgets.QFrame
    QF.NoFrame  = QF.Shape.NoFrame
    QF.HLine    = QF.Shape.HLine
    QF.VLine    = QF.Shape.VLine
    QF.Box      = QF.Shape.Box
    QF.Panel    = QF.Shape.Panel
    QF.Plain    = QF.Shadow.Plain
    QF.Raised   = QF.Shadow.Raised
    QF.Sunken   = QF.Shadow.Sunken

    # QTableWidget / QAbstractItemView edit triggers and selection modes
    QTW = QtWidgets.QTableWidget
    QTW.NoEditTriggers      = QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
    QTW.AllEditTriggers     = QtWidgets.QAbstractItemView.EditTrigger.AllEditTriggers
    QTW.NoSelection         = QtWidgets.QAbstractItemView.SelectionMode.NoSelection
    QTW.SingleSelection     = QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
    QTW.MultiSelection      = QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
    QTW.ExtendedSelection   = QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
    QTW.SelectRows          = QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows

    # QPlainTextEdit wrap mode
    QPT = QtWidgets.QPlainTextEdit
    QPT.WidgetWidth = QPT.LineWrapMode.WidgetWidth
    QPT.NoWrap      = QPT.LineWrapMode.NoWrap

    # QLineEdit echo mode
    QLE = QtWidgets.QLineEdit
    QLE.Password   = QLE.EchoMode.Password
    QLE.Normal     = QLE.EchoMode.Normal
    QLE.NoEcho     = QLE.EchoMode.NoEcho

    # QMessageBox icon / button constants moved to sub-enums
    QMB = QtWidgets.QMessageBox
    QMB.Warning      = QMB.Icon.Warning
    QMB.Critical     = QMB.Icon.Critical
    QMB.Information  = QMB.Icon.Information
    QMB.Question     = QMB.Icon.Question
    QMB.Ok           = QMB.StandardButton.Ok
    QMB.Yes          = QMB.StandardButton.Yes
    QMB.No           = QMB.StandardButton.No
    QMB.Cancel       = QMB.StandardButton.Cancel
    QMB.Close        = QMB.StandardButton.Close


_setup_qt_compat()

import pandas as pd
from PyQt5.QtCore import (
    QThread, Qt, QTimer, pyqtSignal, pyqtSlot,
)
from PyQt5.QtGui import QColor, QFont, QIcon, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QButtonGroup, QComboBox, QDialog, QFileDialog, QGroupBox,
    QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListView,
    QMainWindow, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QScrollArea,
    QSpinBox, QSplitter, QStackedWidget, QStatusBar,
    QTableWidget, QTableWidgetItem,
    QTextEdit, QToolTip, QVBoxLayout, QWidget, QFrame, QCheckBox,
)

# Resolve src/ so support modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from pipeline import TrainWorker, PredictWorker, detect_columns, clean_text, build_prompt, _norm_code
from model_bundle import ModelBundle
from styles import make_stylesheet, LOG_COLORS


APP_DIR     = os.path.dirname(os.path.abspath(__file__))   # project root
CONFIG_PATH = os.path.join(APP_DIR, "config.json")
ASSETS_DIR  = os.path.join(APP_DIR, "assets")
DEFAULT_PROMPT = "NAICS code for item. Supplier: {Supplier}, Description: {Description}"



# ═══════════════════════════════════════════════════════════════════════════
#  Thread wrappers
# ═══════════════════════════════════════════════════════════════════════════

class _WorkerThread(QThread):
    progress = pyqtSignal(dict)
    finished = pyqtSignal(object)   # str path or None

    def __init__(self, worker_class, config: dict, parent=None):
        super().__init__(parent)
        self._config = config
        self._worker_class = worker_class
        self._stop = False
        self._skip_model = False

    def stop(self):
        self._stop = True

    def skip_model(self):
        self._skip_model = True

    def run(self):
        def _skip_checker():
            if self._skip_model:
                self._skip_model = False  # auto-reset so next model isn't skipped
                return True
            return False

        worker = self._worker_class(
            self._config,
            progress_cb=lambda d: self.progress.emit(d),
            stop_checker=lambda: self._stop,
            skip_checker=_skip_checker,
        )
        result = worker.run()
        self.finished.emit(result)


# ═══════════════════════════════════════════════════════════════════════════
#  Reusable widgets
# ═══════════════════════════════════════════════════════════════════════════

class _PathEdit(QPlainTextEdit):
    """Editable multi-line path field — always tall enough to show full content."""
    _MIN_H = 32

    def __init__(self, placeholder: str = "", parent=None):
        super().__init__(parent)
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setPlaceholderText(placeholder)
        self.setFixedHeight(self._MIN_H)
        self.document().contentsChanged.connect(self._adjust)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._adjust()

    def _adjust(self):
        # Use self.width() NOT viewport().width() — viewport is resized
        # AFTER resizeEvent fires, so viewport().width() is stale/zero.
        # fontMetrics().boundingRect is synchronous and never wrong.
        w = self.width() - self.frameWidth() * 2 - 4
        if w <= 0:
            return
        text = self.toPlainText()
        fm = self.fontMetrics()
        if text:
            br = fm.boundingRect(0, 0, w, 1_000_000,
                                 Qt.TextWrapAnywhere, text)
            content_h = br.height()
        else:
            content_h = fm.lineSpacing()
        h = max(self._MIN_H, content_h + self.frameWidth() * 2 + 8)
        if self.height() != h:
            self.setFixedHeight(h)


class _FilePicker(QWidget):
    """Editable multi-line path field + Browse button on the right."""
    def __init__(self, label: str = "", placeholder: str = "", filters: str = "",
                 pick_dir: bool = False, parent=None):
        super().__init__(parent)
        self._filters  = filters
        self._pick_dir = pick_dir

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self._edit = _PathEdit(placeholder or "")
        lay.addWidget(self._edit, 1)

        btn_col = QVBoxLayout()
        btn_col.setContentsMargins(0, 0, 0, 0)
        btn_col.setSpacing(0)
        btn = QPushButton("Browse…")
        btn.setObjectName("browseBtn")
        btn.setFixedWidth(96)
        btn.clicked.connect(self._browse)
        btn_col.addWidget(btn)
        btn_col.addStretch()
        lay.addLayout(btn_col)

    def _browse(self):
        start = self._edit.toPlainText().strip() or APP_DIR
        if self._pick_dir:
            p = QFileDialog.getExistingDirectory(self, "Select folder", start)
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Open file", start, self._filters)
        if p:
            self.set_path(p)

    def path(self) -> str:
        return self._edit.toPlainText().strip()

    def set_path(self, p: str):
        self._edit.setPlainText(p or "")


class _LabeledBar(QWidget):
    """Progress bar with a bold label above it."""
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.lbl = QLabel(label)
        self.lbl.setStyleSheet("font-weight:600; font-size:14px;")
        row.addWidget(self.lbl)
        row.addStretch()
        self.pct_lbl = QLabel("—")
        self.pct_lbl.setStyleSheet(f"color:#333333; font-size:14px;")
        row.addWidget(self.pct_lbl)
        lay.addLayout(row)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(10)
        lay.addWidget(self.bar)

    def set_value(self, v: int, label: str = ""):
        self.bar.setValue(int(v))
        self.pct_lbl.setText(label or f"{v}%")

    def update_label(self, text: str):
        self.lbl.setText(text)

    def reset(self, label="—"):
        self.bar.setValue(0)
        self.pct_lbl.setText(label)

    def set_success(self):
        self.bar.setObjectName("successBar")
        # Force Qt to re-evaluate the app stylesheet for the new objectName
        self.bar.style().unpolish(self.bar)
        self.bar.style().polish(self.bar)


class _StepIndicator(QWidget):
    """Five-step pipeline status display."""
    STEPS = [
        "Load & Clean",
        "Encode Labels",
        "Embeddings",
        "Train XGBoost",
        "Evaluate",
    ]
    _IDLE    = "●"
    _ACTIVE  = "▶"
    _DONE    = "✓"
    _WAIT    = "○"

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)
        self._rows = []  # type: List[QLabel]
        for i, name in enumerate(self.STEPS):
            lbl = QLabel(f"  {self._WAIT}  Step {i+1}: {name}")
            lbl.setStyleSheet("font-size:14px; color:#555555;")
            lay.addWidget(lbl)
            self._rows.append(lbl)

    def reset(self):
        for i, lbl in enumerate(self._rows):
            lbl.setText(f"  {self._WAIT}  Step {i+1}: {self.STEPS[i]}")
            lbl.setStyleSheet("font-size:14px; color:#555555;")

    def set_active(self, step: int):  # 1-indexed
        for i, lbl in enumerate(self._rows):
            if i + 1 < step:
                lbl.setText(f"  {self._DONE}  Step {i+1}: {self.STEPS[i]}")
                lbl.setStyleSheet("font-size:14px; color:#16A34A; font-weight:600;")
            elif i + 1 == step:
                lbl.setText(f"  {self._ACTIVE}  Step {i+1}: {self.STEPS[i]}")
                lbl.setStyleSheet("font-size:14px; color:#990000; font-weight:700;")
            else:
                lbl.setText(f"  {self._WAIT}  Step {i+1}: {self.STEPS[i]}")
                lbl.setStyleSheet("font-size:14px; color:#555555;")

    def set_all_done(self):
        for i, lbl in enumerate(self._rows):
            lbl.setText(f"  {self._DONE}  Step {i+1}: {self.STEPS[i]}")
            lbl.setStyleSheet("font-size:14px; color:#16A34A; font-weight:600;")


# ═══════════════════════════════════════════════════════════════════════════
#  Log view helper
# ═══════════════════════════════════════════════════════════════════════════

def _warn_save_key(parent_widget):
    """Show a one-time security acknowledgment when the user enables key saving."""
    msg = QMessageBox(parent_widget)
    msg.setWindowTitle("Security Notice")
    msg.setIcon(QMessageBox.Warning)
    msg.setText(
        "<b>Your API key will be stored in plain text.</b><br><br>"
        "The key will be saved in <code>config.json</code> located in the same "
        "folder as this application.<br><br>"
        "<span style='color:#DC2626;'>⚠ Never share this file with others or "
        "commit it to version control.</span>"
    )
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec()


def _fl(parent_lay, label_text: str, widget, spacing: int = 2):
    """Add a label-above-field pair into parent_lay (QVBoxLayout)."""
    lbl = QLabel(label_text)
    lbl.setStyleSheet(
        "font-size:13px; font-weight:600; color:#333333; margin-bottom:1px;"
    )
    parent_lay.addWidget(lbl)
    parent_lay.addWidget(widget)
    parent_lay.addSpacing(spacing)


class _TipIcon(QPushButton):
    """ⓘ info icon — shows its tooltip immediately on hover (zero OS delay)."""

    _SS = (
        "QPushButton { background: transparent; border: none; color: #990000;"
        " font-size: 13px; font-weight: 700; padding: 0; margin: 0;"
        " min-height: 0; max-height: 9999px; }"
        "QPushButton:hover { color: #CC3333; }"
    )

    def __init__(self, tooltip: str, parent=None):
        super().__init__("ⓘ", parent)
        self._tip = tooltip
        self.setFixedSize(15, 15)
        self.setCursor(Qt.WhatsThisCursor)
        self.setFocusPolicy(Qt.NoFocus)
        self.setFlat(True)
        self.setStyleSheet(self._SS)

    def enterEvent(self, event):
        QToolTip.showText(
            self.mapToGlobal(self.rect().bottomLeft()),
            self._tip,
            self,
        )
        super().enterEvent(event)

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)


def _tip_icon(tooltip: str) -> _TipIcon:
    """Factory — returns a hover-only ⓘ icon with zero-delay tooltip."""
    return _TipIcon(tooltip)


class _StyledComboBox(QComboBox):
    """QComboBox with a non-native popup view and mouse tracking.

    combobox-popup:0 in QSS forces Qt to use QAbstractItemView for the
    dropdown instead of the platform-native popup, so QSS :hover rules
    are respected on all platforms including macOS.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        view = QListView(self)
        view.setMouseTracking(True)
        self.setView(view)


def _append_log(log_widget: QTextEdit, msg: str, level: str = "info"):
    color = LOG_COLORS.get(level, LOG_COLORS["info"])
    ts    = datetime.now().strftime("%H:%M:%S")
    _html = (
        f'<span style="color:#555555">[{ts}]</span> '
        f'<span style="color:{color}">{html.escape(msg)}</span>'
    )
    log_widget.append(_html)
    log_widget.verticalScrollBar().setValue(log_widget.verticalScrollBar().maximum())


# ═══════════════════════════════════════════════════════════════════════════
#  API key tester
# ═══════════════════════════════════════════════════════════════════════════

class _ApiKeyTester(QThread):
    """Tests an OpenAI API key by calling models.list() (no billing)."""

    success = pyqtSignal(str)   # brief success description
    failure = pyqtSignal(str)   # error message

    def __init__(self, api_key: str, parent=None):
        super().__init__(parent)
        self._key = api_key

    def run(self):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self._key)
            models = list(client.models.list())
            self.success.emit(f"Valid  ✓   ({len(models)} models accessible)")
        except Exception as e:
            msg = str(e)
            # Trim verbose OpenAI error wrappers
            if "Error code:" in msg:
                msg = msg.split("\n")[0]
            self.failure.emit(msg[:120])


# ═══════════════════════════════════════════════════════════════════════════
#  Column selector widget (auto-populated from file)
# ═══════════════════════════════════════════════════════════════════════════

class _ColSelector(QWidget):
    """Browse file → auto-detect headers → populate column dropdowns.

    predict_mode=True hides the NAICS code / description columns (not present
    in unlabelled prediction data).
    """

    columns_changed = pyqtSignal(list)   # emitted with the new column list whenever headers load

    def __init__(self, predict_mode: bool = False, parent=None):
        super().__init__(parent)
        self._predict_mode = predict_mode
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # File picker — editable multi-line path field + browse button on the right
        file_row = QHBoxLayout()
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.setSpacing(6)
        self._file_edit = _PathEdit("Select input file (CSV or Excel)…")
        file_row.addWidget(self._file_edit, 1)
        _fb_col = QVBoxLayout()
        _fb_col.setContentsMargins(0, 0, 0, 0)
        _fb_col.setSpacing(0)
        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.setObjectName("browseBtn")
        self._browse_btn.setFixedWidth(96)
        self._browse_btn.clicked.connect(self._browse)
        _fb_col.addWidget(self._browse_btn)
        _fb_col.addStretch()
        file_row.addLayout(_fb_col)
        lay.addLayout(file_row)

        # Sheet name (shown only for Excel)
        self._sheet_row = QHBoxLayout()
        self._sheet_row.setSpacing(6)
        self._sheet_row.addWidget(QLabel("Sheet:"))
        self.sheet_edit = QLineEdit("Sheet1")
        self.sheet_edit.setMaximumWidth(140)
        self._sheet_row.addWidget(self.sheet_edit)
        self._sheet_row.addStretch()
        self._sheet_widget = QWidget()
        self._sheet_widget.setLayout(self._sheet_row)
        self._sheet_widget.setVisible(False)
        lay.addWidget(self._sheet_widget)

        # Load-headers button
        self._load_btn = QPushButton("⟳  Load / Refresh Headers")
        self._load_btn.clicked.connect(self._load_headers)
        lay.addWidget(self._load_btn)

        # Column dropdowns — label above each combo, full-width
        self.desc_combo = _StyledComboBox()
        self.desc_combo.addItem("")
        _fl(lay, "Description column *", self.desc_combo)

        # Supplier — with "Not in my data" checkbox
        self.supplier_combo = _StyledComboBox()
        self.supplier_combo.addItem("")
        self.no_supplier_chk = QCheckBox("Not in my data")
        self.no_supplier_chk.setStyleSheet("font-size:13px; color:#555555;")
        self.no_supplier_chk.toggled.connect(
            lambda c: self.supplier_combo.setEnabled(not c)
        )
        _supp_hdr = QHBoxLayout()
        _supp_hdr.setContentsMargins(0, 0, 0, 0)
        _supp_lbl = QLabel("Supplier column (optional)")
        _supp_lbl.setStyleSheet("font-size:13px; font-weight:600; color:#333333;")
        _supp_hdr.addWidget(_supp_lbl)
        _supp_hdr.addStretch()
        _supp_hdr.addWidget(self.no_supplier_chk)
        lay.addLayout(_supp_hdr)
        lay.addWidget(self.supplier_combo)
        lay.addSpacing(2)

        if not predict_mode:
            self.label_combo = _StyledComboBox()
            self.label_combo.addItem("")
            _fl(lay, "NAICS/EEIO column *", self.label_combo)

            # NAICS description — with "Not in my data" checkbox
            self.naics_desc_combo = _StyledComboBox()
            self.naics_desc_combo.addItem("")
            self.no_naics_desc_chk = QCheckBox("Not in my data")
            self.no_naics_desc_chk.setStyleSheet("font-size:13px; color:#555555;")
            self.no_naics_desc_chk.toggled.connect(
                lambda c: self.naics_desc_combo.setEnabled(not c)
            )
            _ndesc_hdr = QHBoxLayout()
            _ndesc_hdr.setContentsMargins(0, 0, 0, 0)
            _ndesc_lbl = QLabel("NAICS description column (optional)")
            _ndesc_lbl.setStyleSheet("font-size:13px; font-weight:600; color:#333333;")
            _ndesc_hdr.addWidget(_ndesc_lbl)
            _ndesc_hdr.addStretch()
            _ndesc_hdr.addWidget(self.no_naics_desc_chk)
            lay.addLayout(_ndesc_hdr)
            lay.addWidget(self.naics_desc_combo)
            lay.addSpacing(2)
        else:
            self.label_combo      = _StyledComboBox()
            self.naics_desc_combo = _StyledComboBox()
            self.no_naics_desc_chk = QCheckBox()  # stub (never shown)

        self._columns = []  # type: list

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Open training file",
            self._file_edit.toPlainText().strip() or APP_DIR,
            "Data files (*.csv *.xlsx *.xls);;All files (*)",
        )
        if p:
            self._file_edit.setPlainText(p)
            ext = Path(p).suffix.lower()
            self._sheet_widget.setVisible(ext in (".xlsx", ".xls"))
            self._load_headers()

    def _load_headers(self):
        path = self._file_edit.toPlainText().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "File not found", "Please select a valid file first.")
            return
        try:
            ext = Path(path).suffix.lower()
            if ext in (".xlsx", ".xls"):
                sheet = self.sheet_edit.text().strip() or 0
                df = pd.read_excel(path, sheet_name=sheet, nrows=3)
            else:
                try:
                    df = pd.read_csv(path, nrows=3, encoding="utf-8-sig")
                except UnicodeDecodeError:
                    df = pd.read_csv(path, nrows=3, encoding="latin-1")
            self._set_columns(list(df.columns), df)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot read file:\n{e}")

    def _set_columns(self, cols, df=None):
        self._columns = cols
        suggestions = detect_columns(df) if df is not None else {}

        def _populate(combo, suggested):
            combo.clear()
            combo.addItem("")
            for c in cols:
                combo.addItem(c)
            if suggested and suggested in cols:
                combo.setCurrentText(suggested)

        _populate(self.desc_combo,       suggestions.get("desc_col"))
        _populate(self.supplier_combo,   suggestions.get("supplier_col"))
        _populate(self.label_combo,      suggestions.get("label_col"))
        _populate(self.naics_desc_combo, suggestions.get("naics_desc_col"))
        self.columns_changed.emit(cols)

    def get_values(self) -> dict:
        def _val(combo):
            return combo.currentText().strip()
        no_supp    = self.no_supplier_chk.isChecked()
        no_naics_d = self.no_naics_desc_chk.isChecked()
        return {
            "input_file":        self._file_edit.toPlainText().strip(),
            "sheet_name":        self.sheet_edit.text().strip() or 0,
            "desc_col":          _val(self.desc_combo),
            "supplier_col":      "" if no_supp    else _val(self.supplier_combo),
            "supplier_absent":   no_supp,
            "label_col":         _val(self.label_combo),
            "naics_desc_col":    "" if no_naics_d else _val(self.naics_desc_combo),
            "naics_desc_absent": no_naics_d,
        }

    def set_values(self, d: dict):
        p = d.get("input_file", "")
        self._file_edit.setPlainText(p or "")
        sheet = d.get("sheet_name", "")
        self.sheet_edit.setText(str(sheet) if sheet else "Sheet1")
        # Restore absent-field checkboxes before loading headers
        self.no_supplier_chk.setChecked(d.get("supplier_absent", False))
        if not self._predict_mode:
            self.no_naics_desc_chk.setChecked(d.get("naics_desc_absent", False))
        # Reload columns if file exists
        if p and os.path.isfile(p):
            self._load_headers()
            for combo, key in [
                (self.desc_combo,       "desc_col"),
                (self.supplier_combo,   "supplier_col"),
                (self.label_combo,      "label_col"),
                (self.naics_desc_combo, "naics_desc_col"),
            ]:
                val = d.get(key, "")
                if val and combo.findText(val) >= 0:
                    combo.setCurrentText(val)


# ═══════════════════════════════════════════════════════════════════════════
#  Train Tab
# ═══════════════════════════════════════════════════════════════════════════

class TrainTab(QWidget):
    status_message = pyqtSignal(str)
    model_ready    = pyqtSignal(str)   # emits saved model path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None           # type: Optional[_WorkerThread]
        self._api_key_tester = None   # type: Optional[_ApiKeyTester]
        self._train_ckpt_auto_path = ""
        self._build_ui()

    # ── Build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        root.addWidget(splitter)

        # ─── Left: Settings ───────────────────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_inner = QWidget()
        left_lay   = QVBoxLayout(left_inner)
        left_lay.setSpacing(12)
        left_lay.setContentsMargins(0, 0, 8, 0)
        left_scroll.setWidget(left_inner)

        # Output directory + model name
        out_grp = QGroupBox("Model Output")
        out_lay = QVBoxLayout(out_grp)
        out_lay.setContentsMargins(0, 0, 0, 0)
        out_lay.setSpacing(4)
        self.model_dir_picker = _FilePicker("", "Path to store model & temp files…", pick_dir=True)
        self.model_dir_picker.set_path(APP_DIR)
        self.model_name_edit  = QLineEdit("MyModel")
        _fl(out_lay, "Output directory  (choose a folder)", self.model_dir_picker)
        _fl(out_lay, "Model name  (optional)",             self.model_name_edit)
        left_lay.addWidget(out_grp)

        # Data & column selection
        data_grp = QGroupBox("Training Dataframe  (choose a file below)")
        data_lay  = QVBoxLayout(data_grp)
        data_lay.setContentsMargins(0, 0, 0, 0)
        data_lay.setSpacing(6)
        self.col_selector = _ColSelector()
        data_lay.addWidget(self.col_selector)
        _privacy_note = QLabel(
            "🔒  Only the Description and Supplier column values are sent to OpenAI "
            "for embedding generation. All other columns stay on your machine."
        )
        _privacy_note.setWordWrap(True)
        _privacy_note.setStyleSheet(
            "font-size:12px; color:#333333; background:#FFF0F0;"
            " border-radius:5px; padding:5px 8px; margin-top:4px;"
        )
        data_lay.addWidget(_privacy_note)
        left_lay.addWidget(data_grp)

        # Category-Based Training
        cat_grp = QGroupBox("Category-Based Training (optional)")
        cat_lay = QVBoxLayout(cat_grp)
        cat_lay.setContentsMargins(0, 0, 0, 0)
        cat_lay.setSpacing(6)

        self.use_cat_chk = QCheckBox(
            "Use a category column to train separate models per category"
        )
        self.use_cat_chk.setStyleSheet("font-size:13px; color:#333333;")
        cat_lay.addWidget(self.use_cat_chk)

        # Explanation label (always visible)
        _cat_info = QLabel(
            "When enabled, one specialised model is trained per category "
            "(using an auto-generated category-specific prompt) plus a general "
            "model on all rows.  All models are saved into one .naics_model file."
        )
        _cat_info.setWordWrap(True)
        _cat_info.setStyleSheet("font-size:12px; color:#555555; margin-top:2px;")
        cat_lay.addWidget(_cat_info)

        # Hidden sub-section shown when checkbox is checked
        self._cat_col_widget = QWidget()
        _cat_col_lay = QVBoxLayout(self._cat_col_widget)
        _cat_col_lay.setContentsMargins(0, 4, 0, 0)
        _cat_col_lay.setSpacing(4)
        _cat_col_hdr = QHBoxLayout()
        _cat_col_hdr.setContentsMargins(0, 0, 0, 0)
        _cat_col_lbl = QLabel("Category column *")
        _cat_col_lbl.setStyleSheet("font-size:13px; font-weight:600; color:#333333;")
        _cat_col_hdr.addWidget(_cat_col_lbl)
        _cat_col_hdr.addWidget(_tip_icon(
            "The column that identifies the procurement category for each row "
            "(e.g. 'IT Hardware', 'Office Supplies').\n"
            "One model is trained per distinct category value (requires ≥2 rows "
            "and ≥2 unique NAICS codes per category)."
        ))
        _cat_col_hdr.addStretch()
        _cat_col_lay.addLayout(_cat_col_hdr)
        self._cat_col_combo = _StyledComboBox()
        self._cat_col_combo.addItem("")
        _cat_col_lay.addWidget(self._cat_col_combo)
        self._cat_col_widget.setVisible(False)
        cat_lay.addWidget(self._cat_col_widget)

        self.use_cat_chk.toggled.connect(self._cat_col_widget.setVisible)
        self.col_selector.columns_changed.connect(self._on_col_headers_changed)
        left_lay.addWidget(cat_grp)

        # API Key
        api_grp = QGroupBox("API Key")
        api_lay = QVBoxLayout(api_grp)
        api_lay.setContentsMargins(0, 0, 0, 0)
        api_lay.setSpacing(4)
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("sk-…")
        self.api_key_edit.textChanged.connect(self._reset_api_key_status)
        self._api_key_status = QLabel("")
        self._api_key_status.setStyleSheet("font-size:13px; font-weight:600;")
        _api_test_btn = QPushButton("Test")
        _api_test_btn.setObjectName("browseBtn")
        _api_test_btn.setFixedWidth(52)
        _api_test_btn.setCursor(Qt.PointingHandCursor)
        _api_test_btn.clicked.connect(self._test_api_key)
        _key_row = QHBoxLayout()
        _key_row.setContentsMargins(0, 0, 0, 0)
        _key_row.setSpacing(6)
        _key_row.addWidget(self.api_key_edit)
        _key_row.addWidget(self._api_key_status)
        _key_row.addWidget(_api_test_btn)
        _key_lbl = QLabel("OpenAI API Key")
        _key_lbl.setStyleSheet("font-size:13px; font-weight:600; color:#333333; margin-bottom:1px;")
        api_lay.addWidget(_key_lbl)
        api_lay.addLayout(_key_row)
        api_lay.addSpacing(2)

        self.save_key_chk = QCheckBox("Save API key to config file")
        self.save_key_chk.setStyleSheet("font-size:13px; color:#333333; margin-top:2px;")
        self.save_key_chk.toggled.connect(lambda checked: _warn_save_key(self) if checked else None)
        _save_key_warn = QLabel("⚠ Do this on your own computer only. Do not share the config file with others.")
        _save_key_warn.setStyleSheet("font-size:12px; color:#DC2626;")
        api_lay.addWidget(self.save_key_chk)
        api_lay.addWidget(_save_key_warn)
        _cost_warn = QLabel(
            "⚠ You are responsible for all charges on your OpenAI account. "
            "Do not load more credits than you need, never enable auto-recharge, "
            "and monitor usage at platform.openai.com/usage while the app runs. "
            "Always check openai.com/api/pricing/ for current rates before starting a run."
        )
        _cost_warn.setWordWrap(True)
        _cost_warn.setStyleSheet("font-size:12px; color:#B45309;")
        api_lay.addWidget(_cost_warn)
        _config_note = QLabel(f"ℹ Settings are auto-saved to config.json in the app folder:\n{CONFIG_PATH}")
        _config_note.setWordWrap(True)
        _config_note.setStyleSheet("font-size:11px; color:#6B7280; margin-top:2px;")
        api_lay.addWidget(_config_note)
        left_lay.addWidget(api_grp)

        # Embedding Checkpoint
        ckpt_grp = QGroupBox("Embedding Checkpoint")
        ckpt_lay = QVBoxLayout(ckpt_grp)
        ckpt_lay.setContentsMargins(0, 0, 0, 0)
        ckpt_lay.setSpacing(6)

        _ckpt_save_hdr = QLabel("Save Embeddings")
        _ckpt_save_hdr.setStyleSheet("font-size:13px; font-weight:600; color:#333333;")
        ckpt_lay.addWidget(_ckpt_save_hdr)
        self._train_save_ckpt_chk = QCheckBox("Save embeddings to checkpoint file after embedding step")
        self._train_save_ckpt_chk.setStyleSheet("font-size:13px; color:#333333;")
        ckpt_lay.addWidget(self._train_save_ckpt_chk)
        _tsave_row = QHBoxLayout(); _tsave_row.setContentsMargins(0, 0, 0, 0); _tsave_row.setSpacing(6)
        self._train_save_ckpt_edit = _PathEdit("Path to save .naics_embed file…")
        _tsave_row.addWidget(self._train_save_ckpt_edit, 1)
        _tsave_btn = QPushButton("Browse…"); _tsave_btn.setObjectName("browseBtn"); _tsave_btn.setFixedWidth(96)
        _tsave_btn.clicked.connect(self._browse_save_train_ckpt)
        _tsb = QVBoxLayout(); _tsb.setContentsMargins(0, 0, 0, 0); _tsb.setSpacing(0)
        _tsb.addWidget(_tsave_btn); _tsb.addStretch()
        _tsave_row.addLayout(_tsb)
        ckpt_lay.addLayout(_tsave_row)

        _ckpt_div = QFrame(); _ckpt_div.setFrameShape(QFrame.HLine)
        _ckpt_div.setStyleSheet("color:#E2E8F0; margin-top:4px; margin-bottom:4px;")
        ckpt_lay.addWidget(_ckpt_div)

        _ckpt_load_hdr = QLabel("Load Embeddings")
        _ckpt_load_hdr.setStyleSheet("font-size:13px; font-weight:600; color:#333333;")
        ckpt_lay.addWidget(_ckpt_load_hdr)
        self._train_load_ckpt_chk = QCheckBox("Load from checkpoint — skips embedding step (API key not required)")
        self._train_load_ckpt_chk.setStyleSheet("font-size:13px; color:#333333;")
        ckpt_lay.addWidget(self._train_load_ckpt_chk)
        _tload_row = QHBoxLayout(); _tload_row.setContentsMargins(0, 0, 0, 0); _tload_row.setSpacing(6)
        self._train_load_ckpt_edit = _PathEdit("Path to .naics_embed file…")
        _tload_row.addWidget(self._train_load_ckpt_edit, 1)
        _tload_btn = QPushButton("Browse…"); _tload_btn.setObjectName("browseBtn"); _tload_btn.setFixedWidth(96)
        _tload_btn.clicked.connect(self._browse_load_train_ckpt)
        _tlb = QVBoxLayout(); _tlb.setContentsMargins(0, 0, 0, 0); _tlb.setSpacing(0)
        _tlb.addWidget(_tload_btn); _tlb.addStretch()
        _tload_row.addLayout(_tlb)
        ckpt_lay.addLayout(_tload_row)

        self._train_ckpt_info_lbl = QLabel("")
        self._train_ckpt_info_lbl.setWordWrap(True)
        self._train_ckpt_info_lbl.setStyleSheet(
            "font-size:12px; color:#555555; padding:4px 6px;"
            " background:#F8F9FA; border-radius:4px; border:1px solid #E2E8F0;"
        )
        self._train_ckpt_info_lbl.setVisible(False)
        ckpt_lay.addWidget(self._train_ckpt_info_lbl)
        left_lay.addWidget(ckpt_grp)

        # Hyperparameters — three columns side by side
        hp_grp = QGroupBox("Hyperparameters")
        hp_lay = QHBoxLayout(hp_grp)
        hp_lay.setContentsMargins(0, 0, 0, 0)
        hp_lay.setSpacing(16)

        self.batch_spin  = QSpinBox(); self.batch_spin.setRange(10, 2000);  self.batch_spin.setValue(500)
        self.depth_spin  = QSpinBox(); self.depth_spin.setRange(1, 800);    self.depth_spin.setValue(100)
        self.rounds_spin = QSpinBox(); self.rounds_spin.setRange(10, 5000); self.rounds_spin.setValue(50)

        _HP_TIPS = {
            "batch_spin":  "How many items are sent to the AI in one go.\nLarger = faster but uses more memory. 200 is a safe default.",
            "depth_spin":  "How deep the decision tree can grow.\nHigher = model can learn more complex patterns, but risks overfitting. 8 is a good starting point.",
            "rounds_spin": "How many learning iterations the model runs.\nMore rounds = better accuracy (up to a point). 600 works well for most datasets.",
        }
        for lbl_text, spin, attr in [
            ("Embedding batch size",       self.batch_spin,  "batch_spin"),
            ("ML Model (XGBoost) depth",   self.depth_spin,  "depth_spin"),
            ("Training rounds",            self.rounds_spin, "rounds_spin"),
        ]:
            col = QVBoxLayout()
            col.setSpacing(3)
            hdr = QHBoxLayout()
            hdr.setContentsMargins(0, 0, 0, 0)
            hdr.setSpacing(4)
            lbl = QLabel(lbl_text)
            lbl.setStyleSheet("font-size:13px; font-weight:600; color:#333333;")
            hdr.addWidget(lbl)
            hdr.addWidget(_tip_icon(_HP_TIPS[attr]))
            hdr.addStretch()
            spin.setMinimumWidth(80)
            col.addLayout(hdr)
            col.addWidget(spin)
            hp_lay.addLayout(col)

        left_lay.addWidget(hp_grp)

        left_lay.addStretch()
        splitter.addWidget(left_scroll)

        # ─── Right: Progress & Log ────────────────────────────────────────
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_inner = QWidget()
        right_lay   = QVBoxLayout(right_inner)
        right_lay.setSpacing(12)
        right_lay.setContentsMargins(8, 0, 0, 0)
        right_scroll.setWidget(right_inner)

        # Step indicators
        step_grp = QGroupBox("Pipeline Steps")
        step_lay = QVBoxLayout(step_grp)
        step_lay.setContentsMargins(0, 0, 0, 0)
        self.step_indicator = _StepIndicator()
        step_lay.addWidget(self.step_indicator)
        right_lay.addWidget(step_grp)

        # Progress bars
        prog_grp = QGroupBox("Progress")
        prog_lay = QVBoxLayout(prog_grp)
        prog_lay.setContentsMargins(0, 0, 0, 0)
        prog_lay.setSpacing(8)

        self.overall_bar = _LabeledBar("Overall")
        prog_lay.addWidget(self.overall_bar)

        # ── Embeddings ────────────────────────────────────────────────────────
        _emb_sep = QLabel("Embeddings")
        _emb_sep.setStyleSheet(
            "font-size:13px; font-weight:700; color:#555555;"
            " border-top: 1px solid #E2E8F0; padding-top:6px; margin-top:2px;"
        )
        prog_lay.addWidget(_emb_sep)

        self.embedding_bar       = _LabeledBar("Overall Embedding")
        self.embedding_model_bar = _LabeledBar("Embedding Model")
        prog_lay.addWidget(self.embedding_bar)
        prog_lay.addWidget(self.embedding_model_bar)

        # Embedding metrics — single row
        _EMB_TIPS = {
            "Model Batches":   "Batches sent to the AI for the current model only\n(done / total for this model).",
            "Overall Batches": "Cumulative batches across all models\n(done / total for the entire embedding run).",
            "Total Tokens":    "Cumulative tokens sent to the OpenAI API across all models.\nOne token ≈ 3–4 characters of text.",
            "Est. Cost":       "Cumulative estimated API cost for this run.\n$0.13 per 1M tokens (text-embedding-3-large, Feb 2026).",
            "Model ETA":       "Estimated time remaining to finish embedding the current model.",
            "Overall ETA":     "Estimated time remaining to finish embedding all models.",
        }
        emb_stats = QHBoxLayout(); emb_stats.setSpacing(20)
        for attr, label in [
            ("_emb_lbl_batch",         "Model Batches"),
            ("_emb_lbl_overall_batch", "Overall Batches"),
            ("_emb_lbl_tokens",        "Total Tokens"),
            ("_emb_lbl_cost",          "Est. Cost"),
            ("_emb_lbl_eta",           "Model ETA"),
            ("_emb_lbl_eta_overall",   "Overall ETA"),
        ]:
            _col = QVBoxLayout(); _col.setSpacing(2)
            _hdr = QHBoxLayout(); _hdr.setContentsMargins(0, 0, 0, 0); _hdr.setSpacing(3)
            _title = QLabel(label); _title.setStyleSheet("font-size:13px; color:#333333;")
            _hdr.addWidget(_title)
            if label in _EMB_TIPS:
                _hdr.addWidget(_tip_icon(_EMB_TIPS[label]))
            _col.addLayout(_hdr)
            _val = QLabel("—"); _val.setStyleSheet("font-size:18px; font-weight:700;")
            _col.addWidget(_val)
            setattr(self, attr, _val)
            emb_stats.addLayout(_col)
        emb_stats.addStretch()
        prog_lay.addLayout(emb_stats)

        _cost_note = QLabel(
            '* Estimated cost · $0.13 / 1M tokens (text-embedding-3-large, as of Feb 20, 2026) · '
            '<a href="https://platform.openai.com/docs/pricing" style="color:#990000;">'
            'Check current rates</a>'
            ' · For reference only — actual cost varies depending on your data.'
        )
        _cost_note.setOpenExternalLinks(True)
        _cost_note.setStyleSheet("font-size:12px; color:#555555; margin-top:1px;")
        _cost_note.setWordWrap(True)
        prog_lay.addWidget(_cost_note)

        # ── Training ─────────────────────────────────────────────────────────
        _trn_sep = QLabel("Training")
        _trn_sep.setStyleSheet(
            "font-size:13px; font-weight:700; color:#555555;"
            " border-top: 1px solid #E2E8F0; padding-top:6px; margin-top:2px;"
        )
        prog_lay.addWidget(_trn_sep)

        self.training_bar       = _LabeledBar("Overall Training")
        self.training_model_bar = _LabeledBar("Training Model")
        prog_lay.addWidget(self.training_bar)
        prog_lay.addWidget(self.training_model_bar)

        # Training metrics — single row: model round, overall round, loss, accuracy, ETAs
        _TRN_TIPS = {
            "Model Round":   "Current round / total rounds for the current model only.",
            "Overall Round": "Cumulative round count across all models.",
            "Loss":          "Training loss — lower is better.",
            "Accuracy":      "Training accuracy for the current model — measured on the training set.",
            "Model ETA":     "Estimated time until the current model finishes training.",
            "Overall ETA":   "Estimated time until all models finish training.",
        }
        trn_stats = QHBoxLayout(); trn_stats.setSpacing(20)
        for attr, label in [
            ("_lbl_model_round", "Model Round"),
            ("_lbl_round",       "Overall Round"),
            ("_lbl_loss",        "Loss"),
            ("_lbl_acc",         "Accuracy"),
            ("_lbl_eta",         "Model ETA"),
            ("_lbl_eta_overall", "Overall ETA"),
        ]:
            col = QVBoxLayout(); col.setSpacing(2)
            _hdr2 = QHBoxLayout(); _hdr2.setContentsMargins(0, 0, 0, 0); _hdr2.setSpacing(3)
            title = QLabel(label); title.setStyleSheet("font-size:13px; color:#333333;")
            _hdr2.addWidget(title)
            if label in _TRN_TIPS:
                _hdr2.addWidget(_tip_icon(_TRN_TIPS[label]))
            col.addLayout(_hdr2)
            val = QLabel("—"); val.setStyleSheet("font-size:18px; font-weight:700;")
            col.addWidget(val)
            setattr(self, attr, val)
            trn_stats.addLayout(col)
        trn_stats.addStretch()
        prog_lay.addLayout(trn_stats)

        right_lay.addWidget(prog_grp)

        # Log
        log_grp = QGroupBox("Log")
        log_lay = QVBoxLayout(log_grp)
        log_lay.setContentsMargins(0, 0, 0, 0)
        self.log = QTextEdit()
        self.log.setObjectName("logView")
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(100)
        #self.log.setMaximumHeight(180)
        log_lay.addWidget(self.log)
        right_lay.addWidget(log_grp)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch()
        self.skip_btn  = QPushButton("Skip Current Model")
        self.skip_btn.setEnabled(False)
        self.skip_btn.clicked.connect(self._skip_model)
        self.start_btn = QPushButton("Start Training")
        self.start_btn.setObjectName("primaryBtn")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn  = QPushButton("Stop Training")
        self.stop_btn.setObjectName("dangerBtn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        btn_row.addWidget(self.skip_btn)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        right_lay.addLayout(btn_row)

        splitter.addWidget(right_scroll)
        splitter.setSizes([420, 520])

        # Auto-default save-ckpt path when input file or output dir changes
        self.col_selector.columns_changed.connect(self._update_train_save_ckpt_default)
        self.model_dir_picker._edit.document().contentsChanged.connect(
            self._update_train_save_ckpt_default
        )

    # ── Actions ────────────────────────────────────────────────────────────

    def _reset_api_key_status(self):
        self._api_key_status.setText("")

    def _test_api_key(self):
        key = self.api_key_edit.text().strip()
        if not key:
            QMessageBox.warning(self, "API Key", "Please enter an API key first.")
            return
        if self._api_key_tester is not None and self._api_key_tester.isRunning():
            return  # already testing — ignore duplicate click
        self._api_key_status.setText("Testing…")
        self._api_key_status.setStyleSheet("font-size:13px; font-weight:600; color:#333333;")
        self._api_key_tester = _ApiKeyTester(key, self)
        self._api_key_tester.success.connect(self._on_api_key_ok)
        self._api_key_tester.failure.connect(self._on_api_key_fail)
        self._api_key_tester.start()

    @pyqtSlot(str)
    def _on_api_key_ok(self, msg: str):
        self._api_key_status.setText("✓  Valid")
        self._api_key_status.setStyleSheet("font-size:13px; font-weight:600; color:#16A34A;")
        QMessageBox.information(self, "API Key Test", f"✓  Key is valid.\n\n{msg}")

    @pyqtSlot(str)
    def _on_api_key_fail(self, msg: str):
        self._api_key_status.setText("✗  Invalid")
        self._api_key_status.setStyleSheet("font-size:13px; font-weight:600; color:#DC2626;")
        QMessageBox.critical(self, "API Key Test", f"✗  Key test failed:\n\n{msg}")

    def _on_col_headers_changed(self, cols: list):
        """Repopulate category combo when training file headers change."""
        current = self._cat_col_combo.currentText()
        self._cat_col_combo.clear()
        self._cat_col_combo.addItem("")
        for c in cols:
            self._cat_col_combo.addItem(c)
        if current and self._cat_col_combo.findText(current) >= 0:
            self._cat_col_combo.setCurrentText(current)

    def _update_train_save_ckpt_default(self):
        """Auto-fill the save-embeddings path when the input file or output dir changes."""
        current = self._train_save_ckpt_edit.toPlainText().strip()
        if current and current != self._train_ckpt_auto_path:
            return  # user has a manually-set path — leave it alone
        p = self.col_selector._file_edit.toPlainText().strip()
        if p:
            out_dir = self.model_dir_picker.path() or str(Path(p).parent)
            new_path = str(Path(out_dir) / f"embedded_{Path(p).stem}.naics_embed")
        else:
            new_path = ""
        self._train_ckpt_auto_path = new_path
        self._train_save_ckpt_edit.setPlainText(new_path)

    def _browse_save_train_ckpt(self):
        p, _ = QFileDialog.getSaveFileName(
            self, "Save Embedding Checkpoint",
            self._train_save_ckpt_edit.toPlainText().strip() or APP_DIR,
            "NAICS Embedding Checkpoint (*.naics_embed);;All files (*)",
        )
        if p:
            if not p.endswith(".naics_embed"):
                p += ".naics_embed"
            self._train_save_ckpt_edit.setPlainText(p)

    def _browse_load_train_ckpt(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Open Embedding Checkpoint",
            self._train_load_ckpt_edit.toPlainText().strip() or APP_DIR,
            "NAICS Embedding Checkpoint (*.naics_embed);;All files (*)",
        )
        if p:
            self._train_load_ckpt_edit.setPlainText(p)
            self._load_train_ckpt_info(p)

    def _load_train_ckpt_info(self, path: str):
        try:
            from embed_checkpoint import peek_ckpt
            meta = peek_ckpt(path)
            bundle_type = meta.get("bundle_type", "single")
            categories  = meta.get("categories", [])
            n_rows = meta.get("models", {}).get("__general__", {}).get("n_rows", "?")
            cats_str = (f" | {len(categories)} categories: {', '.join(categories)}"
                        if categories else "")
            text = (f"{bundle_type} | {n_rows:,} general rows{cats_str}"
                    f" | created {meta.get('created_at', '')}"
                    if isinstance(n_rows, int) else
                    f"{bundle_type}{cats_str} | created {meta.get('created_at', '')}")
            self._train_ckpt_info_lbl.setText(text)
            self._train_ckpt_info_lbl.setVisible(True)
        except Exception as e:
            self._train_ckpt_info_lbl.setText(f"Cannot read checkpoint: {e}")
            self._train_ckpt_info_lbl.setVisible(True)

    def _start(self):
        col_vals = self.col_selector.get_values()
        errors = []
        out_dir = self.model_dir_picker.path()
        if not out_dir:
            errors.append("Output directory is required.")
        elif not os.path.isdir(out_dir):
            errors.append(f"Output directory does not exist:\n  {out_dir}")

        load_ckpt  = self._train_load_ckpt_chk.isChecked()
        load_ckpt_path = self._train_load_ckpt_edit.toPlainText().strip() if load_ckpt else ""
        save_ckpt  = self._train_save_ckpt_chk.isChecked()
        save_ckpt_path = self._train_save_ckpt_edit.toPlainText().strip() if save_ckpt else ""

        if load_ckpt:
            if not load_ckpt_path:
                errors.append("Checkpoint file path is required when loading embeddings.")
            elif not os.path.isfile(load_ckpt_path):
                errors.append(f"Checkpoint file not found:\n  {load_ckpt_path}")
        else:
            input_file = col_vals.get("input_file", "")
            if not input_file:
                errors.append("Input file is required.")
            elif not os.path.isfile(input_file):
                errors.append(f"Input file not found:\n  {input_file}")
            if not col_vals["desc_col"]:
                errors.append("Description column must be selected.")
            if not col_vals["label_col"]:
                errors.append("NAICS/EEIO column must be selected.")
            if not self.api_key_edit.text().strip():
                errors.append("OpenAI API key is required.")

        if save_ckpt and save_ckpt_path:
            _ckpt_parent = Path(save_ckpt_path).parent
            if not _ckpt_parent.is_dir():
                errors.append(
                    f"Checkpoint save directory does not exist:\n  {_ckpt_parent}\n"
                    "Please create it or choose a different path."
                )

        # Category column required if checkbox is checked (only when not loading from checkpoint)
        use_category = self.use_cat_chk.isChecked()
        category_col = self._cat_col_combo.currentText().strip() if use_category else ""
        if not load_ckpt and use_category and not category_col:
            errors.append("Category column must be selected when category-based training is enabled.")

        if errors:
            QMessageBox.warning(self, "Cannot start — please fix the following", "\n\n".join(errors))
            return

        config = {
            **col_vals,
            "model_dir":              self.model_dir_picker.path(),
            "model_name":             self.model_name_edit.text().strip() or "MyModel",
            "api_key":                self.api_key_edit.text().strip(),
            "prompt_template":        DEFAULT_PROMPT,
            "batch_size":             self.batch_spin.value(),
            "max_depth":              self.depth_spin.value(),
            "num_boost_round":        self.rounds_spin.value(),
            "use_category":           use_category,
            "category_col":           category_col,
            "save_ckpt_path":         save_ckpt_path,
            "load_ckpt_path":         load_ckpt_path,
        }

        self._reset_ui()
        if use_category:
            _append_log(self.log,
                f"Category-based training enabled  ·  column: '{category_col}'", "info")
        self._thread = _WorkerThread(TrainWorker, config, self)
        self._thread.progress.connect(self._on_progress)
        self._thread.finished.connect(self._on_done)
        self._thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.skip_btn.setEnabled(True)
        self.status_message.emit("Training started…")
        _append_log(self.log, "Training started.", "info")

    def _stop(self):
        if self._thread:
            self._thread.stop()
        self.stop_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)
        _append_log(self.log, "Stop requested — will halt after current batch.", "warning")

    def _skip_model(self):
        if self._thread:
            self._thread.skip_model()
        self.skip_btn.setEnabled(False)
        _append_log(self.log, "Skip requested — saving current model and moving to next.", "warning")

    def _reset_ui(self):
        self.step_indicator.reset()
        self.overall_bar.reset()
        self.embedding_bar.reset()
        self.embedding_model_bar.reset()
        self.embedding_model_bar.update_label("Embedding Model")
        self._emb_lbl_batch.setText("—")
        self._emb_lbl_overall_batch.setText("—")
        self._emb_lbl_tokens.setText("—")
        self._emb_lbl_eta.setText("—")
        self._emb_lbl_eta_overall.setText("—")
        self._emb_lbl_cost.setText("—")
        self.training_bar.reset()
        self.training_model_bar.reset()
        self.training_model_bar.update_label("Training Model")
        self._lbl_model_round.setText("—")
        self._lbl_round.setText("—")
        self._lbl_loss.setText("—")
        self._lbl_acc.setText("—")
        self._lbl_eta.setText("—")
        self._lbl_eta_overall.setText("—")
        self.log.clear()

    # ── Callbacks ──────────────────────────────────────────────────────────

    @pyqtSlot(dict)
    def _on_progress(self, d: dict):
        msg_type = d.get("type", "")

        if msg_type == "log":
            _append_log(self.log, d["message"], d.get("level", "info"))
            return

        if msg_type == "auth_error":
            QMessageBox.critical(
                self, "Invalid API Key",
                "Your OpenAI API key was rejected.\n\n"
                "Please check that the key is correct and has sufficient credits.\n\n"
                f"Detail: {d.get('message', '')}",
            )
            return

        stage = d.get("stage", "")
        pct   = int(d.get("pct", 0))

        if stage == "step":
            step = d.get("step", 1)
            # Piecewise-linear mapping of pipeline step-pct (0–100) to overall %:
            # step1=0–5%, step2=5–10%, embed=10–52%, train=52–95%, eval=95–100%
            _bp = [(0, 0), (20, 5), (40, 10), (60, 52), (80, 95), (100, 100)]
            overall = _bp[-1][1]
            for _i in range(len(_bp) - 1):
                _x0, _y0 = _bp[_i]; _x1, _y1 = _bp[_i + 1]
                if _x0 <= pct <= _x1:
                    overall = int(_y0 + (_y1 - _y0) * (pct - _x0) / (_x1 - _x0))
                    break
            self.overall_bar.set_value(overall, f"{overall}%")
            self.step_indicator.set_active(step)

        elif stage == "embedding":
            model_idx     = d.get("model_idx", 0)
            model_total   = d.get("model_total", 1)
            model_label   = d.get("model_label", "General")
            model_pct     = d.get("model_pct", pct)
            batch_done    = d.get("batch_done", "—")
            total_batches = d.get("total_batches", "—")
            ovr_batch     = d.get("overall_batch_done", "—")
            ovr_total_b   = d.get("overall_total_batches", "—")
            tok_total     = d.get("tokens_total", 0)
            eta_model     = d.get("eta_model", "—")
            eta_overall   = d.get("eta_overall", "—")
            cost_usd      = tok_total / 1_000_000 * 0.13
            self.embedding_bar.set_value(pct, f"{pct:.1f}%")
            self.embedding_model_bar.update_label(
                f"Embedding Model: {model_label}  ({model_idx + 1}/{model_total})"
            )
            self.embedding_model_bar.set_value(model_pct, f"{model_pct:.1f}%")
            self._emb_lbl_batch.setText(f"{batch_done} / {total_batches}")
            self._emb_lbl_overall_batch.setText(f"{ovr_batch} / {ovr_total_b}")
            self._emb_lbl_tokens.setText(f"{tok_total:,}")
            self._emb_lbl_eta.setText(str(eta_model))
            self._emb_lbl_eta_overall.setText(str(eta_overall))
            self._emb_lbl_cost.setText(f"~${cost_usd:.4f}")
            # Embedding = step 3 → overall range 10–52 %
            overall = min(52, int(10 + pct * 0.42))
            self.overall_bar.set_value(overall, f"{overall}%")

        elif stage == "training":
            rnd          = d.get("round", 0)
            total        = d.get("total_rounds", 1)
            model_round  = d.get("model_round", rnd)
            model_rounds = d.get("model_total_rounds", total)
            model_pct    = d.get("model_pct", pct)
            model_idx    = d.get("model_idx", 0)
            model_total  = d.get("model_total", 1)
            model_label  = d.get("model_label", "General")
            loss         = d.get("loss", 0.0)
            acc          = d.get("accuracy", 0.0)
            eta_model    = d.get("eta_model", "—")
            eta_overall  = d.get("eta_overall", "—")
            self.training_bar.set_value(pct, f"{pct:.1f}%")
            self.training_model_bar.update_label(
                f"Training Model: {model_label}  ({model_idx + 1}/{model_total})"
            )
            self.training_model_bar.set_value(model_pct, f"{model_pct:.1f}%")
            if not self.skip_btn.isEnabled():
                self.skip_btn.setEnabled(True)
            self._lbl_model_round.setText(f"{model_round} / {model_rounds}")
            self._lbl_round.setText(f"{rnd} / {total}")
            self._lbl_loss.setText(f"{loss:.4f}")
            self._lbl_acc.setText(f"{acc:.1f}%")
            self._lbl_eta.setText(str(eta_model))
            self._lbl_eta_overall.setText(str(eta_overall))
            # Training = step 4 → overall range 52–95 %
            overall = min(95, int(52.5 + pct * 0.425))
            self.overall_bar.set_value(overall, f"{overall}%")

        elif stage == "done":
            self.overall_bar.set_value(100, "100% — Done")
            self.overall_bar.set_success()
            self.step_indicator.set_all_done()

    @pyqtSlot(object)
    def _on_done(self, result):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)
        if result:
            _append_log(self.log, f"✓ Model saved: {result}", "success")
            self.status_message.emit(f"Training complete: {Path(result).name}")
            self.model_ready.emit(result)
            QMessageBox.information(self, "Training Complete",
                f"Model saved successfully:\n{result}")
        else:
            _append_log(self.log, "Training did not complete.", "warning")
            self.status_message.emit("Training stopped or failed.")

    # ── Config persistence ─────────────────────────────────────────────────

    def save_config(self) -> dict:
        save_key = self.save_key_chk.isChecked()
        return {
            "train_model_dir":        self.model_dir_picker.path(),
            "train_model_name":       self.model_name_edit.text(),
            "train_api_key":          self.api_key_edit.text() if save_key else "",
            "train_save_key":         save_key,
            "train_batch_size":       self.batch_spin.value(),
            "train_max_depth":        self.depth_spin.value(),
            "train_num_rounds":       self.rounds_spin.value(),
            "train_use_category":     self.use_cat_chk.isChecked(),
            "train_category_col":     self._cat_col_combo.currentText(),
            "train_save_ckpt":        self._train_save_ckpt_chk.isChecked(),
            "train_save_ckpt_path":   self._train_save_ckpt_edit.toPlainText().strip(),
            "train_load_ckpt":        self._train_load_ckpt_chk.isChecked(),
            "train_load_ckpt_path":   self._train_load_ckpt_edit.toPlainText().strip(),
            **{f"train_{k}": v for k, v in self.col_selector.get_values().items()},
        }

    def load_config(self, d: dict):
        self.model_dir_picker.set_path(d.get("train_model_dir", ""))
        self.model_name_edit.setText(d.get("train_model_name", "MyModel"))
        save_key = d.get("train_save_key", False)
        self.save_key_chk.blockSignals(True)
        self.save_key_chk.setChecked(save_key)
        self.save_key_chk.blockSignals(False)
        if save_key:
            self.api_key_edit.setText(d.get("train_api_key", ""))
        self.batch_spin.setValue(d.get("train_batch_size", 500))
        self.depth_spin.setValue(d.get("train_max_depth", 100))
        self.rounds_spin.setValue(d.get("train_num_rounds", 50))
        self.use_cat_chk.setChecked(d.get("train_use_category", False))
        self._train_save_ckpt_chk.setChecked(d.get("train_save_ckpt", False))
        self._train_save_ckpt_edit.setPlainText(d.get("train_save_ckpt_path", ""))
        self._train_load_ckpt_chk.setChecked(d.get("train_load_ckpt", False))
        load_p = d.get("train_load_ckpt_path", "")
        self._train_load_ckpt_edit.setPlainText(load_p)
        if load_p and os.path.isfile(load_p):
            self._load_train_ckpt_info(load_p)
        # Restore column selector — strip "train_" prefix to recover original keys
        col_keys = (
            "input_file", "sheet_name",
            "desc_col", "supplier_col", "supplier_absent",
            "label_col", "naics_desc_col", "naics_desc_absent",
        )
        col_d = {k: d[f"train_{k}"] for k in col_keys if f"train_{k}" in d}
        if col_d:
            self.col_selector.set_values(col_d)
        # Restore category column (after col_selector loads headers so the item exists)
        saved_cat_col = d.get("train_category_col", "")
        if saved_cat_col and self._cat_col_combo.findText(saved_cat_col) >= 0:
            self._cat_col_combo.setCurrentText(saved_cat_col)


# ═══════════════════════════════════════════════════════════════════════════
#  Predict Tab
# ═══════════════════════════════════════════════════════════════════════════

class PredictTab(QWidget):
    status_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None           # type: Optional[_WorkerThread]
        self._api_key_tester = None   # type: Optional[_ApiKeyTester]
        self._pred_ckpt_auto_path = ""
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        root.addWidget(splitter)

        # ─── Left: Settings ───────────────────────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_inner = QWidget()
        left_lay   = QVBoxLayout(left_inner)
        left_lay.setSpacing(12)
        left_lay.setContentsMargins(0, 0, 8, 0)
        left_scroll.setWidget(left_inner)

        # Model selection
        model_grp = QGroupBox("Model")
        model_lay = QVBoxLayout(model_grp)
        model_lay.setContentsMargins(0, 0, 0, 0)
        model_lay.setSpacing(4)

        lbl_mf = QLabel("Model file (.naics_model)")
        lbl_mf.setStyleSheet("font-size:13px; font-weight:600; color:#333333; margin-bottom:1px;")
        model_lay.addWidget(lbl_mf)

        model_pick_row = QHBoxLayout()
        model_pick_row.setContentsMargins(0, 0, 0, 0)
        model_pick_row.setSpacing(6)
        self._model_path_edit = _PathEdit("Path to .naics_model file…")
        model_pick_row.addWidget(self._model_path_edit, 1)
        _mb_col = QVBoxLayout()
        _mb_col.setContentsMargins(0, 0, 0, 0)
        _mb_col.setSpacing(0)
        browse_btn = QPushButton("Browse…")
        browse_btn.setObjectName("browseBtn")
        browse_btn.setFixedWidth(96)
        browse_btn.clicked.connect(self._browse_model)
        _mb_col.addWidget(browse_btn)
        _mb_col.addStretch()
        model_pick_row.addLayout(_mb_col)
        model_lay.addLayout(model_pick_row)

        self.model_info_lbl = QLabel("No model loaded")
        self.model_info_lbl.setStyleSheet(
            "color:#5C3A00; font-size:13px; padding:6px 8px; "
            "background:#FFF8E0; border-radius:5px; border:1px solid #FFC72C;"
        )
        self.model_info_lbl.setWordWrap(True)
        model_lay.addWidget(self.model_info_lbl)
        left_lay.addWidget(model_grp)

        # Category routing group (hidden until a multi-model bundle is loaded)
        self._pred_cat_grp = QGroupBox("Category Routing")
        _pred_cat_lay = QVBoxLayout(self._pred_cat_grp)
        _pred_cat_lay.setContentsMargins(0, 0, 0, 0)
        _pred_cat_lay.setSpacing(6)

        self._pred_cat_info_lbl = QLabel("")
        self._pred_cat_info_lbl.setWordWrap(True)
        self._pred_cat_info_lbl.setStyleSheet(
            "font-size:12px; color:#555555; margin-bottom:2px;"
        )
        _pred_cat_lay.addWidget(self._pred_cat_info_lbl)

        _pred_cat_col_hdr = QHBoxLayout()
        _pred_cat_col_hdr.setContentsMargins(0, 0, 0, 0)
        _pred_cat_col_lbl = QLabel("Category column (optional)")
        _pred_cat_col_lbl.setStyleSheet("font-size:13px; font-weight:600; color:#333333;")
        _pred_cat_col_hdr.addWidget(_pred_cat_col_lbl)
        _pred_cat_col_hdr.addWidget(_tip_icon(
            "Select the column in your prediction file that contains category values.\n"
            "Each row will be routed to its matching category model.\n"
            "Rows with an unrecognised or missing category fall back to the general model.\n"
            "Leave blank to use the general model for all rows."
        ))
        _pred_cat_col_hdr.addStretch()
        _pred_cat_lay.addLayout(_pred_cat_col_hdr)
        self._pred_cat_col_combo = _StyledComboBox()
        self._pred_cat_col_combo.addItem("")
        _pred_cat_lay.addWidget(self._pred_cat_col_combo)

        self._pred_cat_grp.setVisible(False)

        # Input data
        data_grp = QGroupBox("Input Dataframe  (choose a file below)")
        data_lay  = QVBoxLayout(data_grp)
        data_lay.setContentsMargins(0, 0, 0, 0)
        self.pred_col_selector = _ColSelector(predict_mode=True)
        data_lay.addWidget(self.pred_col_selector)
        _pred_privacy_note = QLabel(
            "🔒  Only the Description and Supplier column values are sent to OpenAI "
            "for embedding generation. All other columns stay on your machine."
        )
        _pred_privacy_note.setWordWrap(True)
        _pred_privacy_note.setStyleSheet(
            "font-size:12px; color:#333333; background:#FFF8E0;"
            " border-radius:5px; padding:5px 8px; margin-top:4px;"
        )
        data_lay.addWidget(_pred_privacy_note)
        left_lay.addWidget(data_grp)
        left_lay.addWidget(self._pred_cat_grp)

        # Populate prediction category combo when input file headers change
        self.pred_col_selector.columns_changed.connect(self._on_pred_col_headers_changed)

        # API + options
        opt_grp = QGroupBox("API & Options")
        opt_lay = QVBoxLayout(opt_grp)
        opt_lay.setContentsMargins(0, 0, 0, 0)
        opt_lay.setSpacing(4)
        self.pred_api_key = QLineEdit()
        self.pred_api_key.setEchoMode(QLineEdit.Password)
        self.pred_api_key.setPlaceholderText("sk-…")
        self.pred_api_key.textChanged.connect(self._reset_api_key_status)
        self._pred_api_key_status = QLabel("")
        self._pred_api_key_status.setStyleSheet("font-size:13px; font-weight:600;")
        _pred_test_btn = QPushButton("Test")
        _pred_test_btn.setObjectName("browseBtn")
        _pred_test_btn.setFixedWidth(52)
        _pred_test_btn.setCursor(Qt.PointingHandCursor)
        _pred_test_btn.clicked.connect(self._test_api_key)
        _pred_key_row = QHBoxLayout()
        _pred_key_row.setContentsMargins(0, 0, 0, 0)
        _pred_key_row.setSpacing(6)
        _pred_key_row.addWidget(self.pred_api_key)
        _pred_key_row.addWidget(self._pred_api_key_status)
        _pred_key_row.addWidget(_pred_test_btn)
        _pred_key_lbl = QLabel("OpenAI API Key")
        _pred_key_lbl.setStyleSheet("font-size:13px; font-weight:600; color:#333333; margin-bottom:1px;")
        opt_lay.addWidget(_pred_key_lbl)
        opt_lay.addLayout(_pred_key_row)
        opt_lay.addSpacing(2)

        self.pred_save_key_chk = QCheckBox("Save API key to config file")
        self.pred_save_key_chk.setStyleSheet("font-size:13px; color:#333333; margin-top:2px;")
        self.pred_save_key_chk.toggled.connect(lambda checked: _warn_save_key(self) if checked else None)
        _pred_save_key_warn = QLabel("⚠ Do this on your own computer only. Do not share the config file with others.")
        _pred_save_key_warn.setStyleSheet("font-size:12px; color:#DC2626;")
        opt_lay.addWidget(self.pred_save_key_chk)
        opt_lay.addWidget(_pred_save_key_warn)
        _pred_cost_warn = QLabel(
            "⚠ You are responsible for all charges on your OpenAI account. "
            "Do not load more credits than you need, never enable auto-recharge, "
            "and monitor usage at platform.openai.com/usage while the app runs. "
            "Always check openai.com/api/pricing/ for current rates before starting a run."
        )
        _pred_cost_warn.setWordWrap(True)
        _pred_cost_warn.setStyleSheet("font-size:12px; color:#B45309;")
        opt_lay.addWidget(_pred_cost_warn)
        _pred_config_note = QLabel(f"ℹ Settings are auto-saved to config.json in the app folder:\n{CONFIG_PATH}")
        _pred_config_note.setWordWrap(True)
        _pred_config_note.setStyleSheet("font-size:11px; color:#6B7280; margin-top:2px;")
        opt_lay.addWidget(_pred_config_note)

        # Top-K and Batch side by side
        spin_row = QHBoxLayout()
        spin_row.setSpacing(16)
        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(1, 10)
        self.topk_spin.setValue(3)
        self.pred_batch_spin = QSpinBox()
        self.pred_batch_spin.setRange(10, 2000)
        self.pred_batch_spin.setValue(500)
        _PRED_TIPS = {
            "Top-K predictions": (
                "How many NAICS category predictions to return for each item.\n"
                "The model picks the top K most likely categories and shows their confidence scores.\n"
                "3 is a good default — gives the best guess plus two alternatives."
            ),
            "Batch size": (
                "How many items are sent to the AI in one go.\n"
                "Larger batches are faster but use more memory. 200 is a safe default."
            ),
        }
        for lbl_text, spin in [("Top-K predictions", self.topk_spin), ("Batch size", self.pred_batch_spin)]:
            col = QVBoxLayout()
            col.setSpacing(3)
            hdr = QHBoxLayout()
            hdr.setContentsMargins(0, 0, 0, 0)
            hdr.setSpacing(4)
            lbl = QLabel(lbl_text)
            lbl.setStyleSheet("font-size:13px; font-weight:600; color:#333333;")
            hdr.addWidget(lbl)
            hdr.addWidget(_tip_icon(_PRED_TIPS[lbl_text]))
            hdr.addStretch()
            spin.setMinimumWidth(80)
            col.addLayout(hdr)
            col.addWidget(spin)
            spin_row.addLayout(col)
        spin_row.addStretch()
        opt_lay.addSpacing(2)
        opt_lay.addLayout(spin_row)
        left_lay.addWidget(opt_grp)

        # Embedding Checkpoint
        pred_ckpt_grp = QGroupBox("Embedding Checkpoint")
        pred_ckpt_lay = QVBoxLayout(pred_ckpt_grp)
        pred_ckpt_lay.setContentsMargins(0, 0, 0, 0)
        pred_ckpt_lay.setSpacing(6)

        _pckpt_save_hdr = QLabel("Save Embeddings")
        _pckpt_save_hdr.setStyleSheet("font-size:13px; font-weight:600; color:#333333;")
        pred_ckpt_lay.addWidget(_pckpt_save_hdr)
        self._pred_save_ckpt_chk = QCheckBox("Save embeddings to checkpoint file after embedding step")
        self._pred_save_ckpt_chk.setStyleSheet("font-size:13px; color:#333333;")
        pred_ckpt_lay.addWidget(self._pred_save_ckpt_chk)
        _psave_row = QHBoxLayout(); _psave_row.setContentsMargins(0, 0, 0, 0); _psave_row.setSpacing(6)
        self._pred_save_ckpt_edit = _PathEdit("Path to save .naics_embed file…")
        _psave_row.addWidget(self._pred_save_ckpt_edit, 1)
        _psave_btn = QPushButton("Browse…"); _psave_btn.setObjectName("browseBtn"); _psave_btn.setFixedWidth(96)
        _psave_btn.clicked.connect(self._browse_save_pred_ckpt)
        _psb = QVBoxLayout(); _psb.setContentsMargins(0, 0, 0, 0); _psb.setSpacing(0)
        _psb.addWidget(_psave_btn); _psb.addStretch()
        _psave_row.addLayout(_psb)
        pred_ckpt_lay.addLayout(_psave_row)

        _pckpt_div = QFrame(); _pckpt_div.setFrameShape(QFrame.HLine)
        _pckpt_div.setStyleSheet("color:#E2E8F0; margin-top:4px; margin-bottom:4px;")
        pred_ckpt_lay.addWidget(_pckpt_div)

        _pckpt_load_hdr = QLabel("Load Embeddings")
        _pckpt_load_hdr.setStyleSheet("font-size:13px; font-weight:600; color:#333333;")
        pred_ckpt_lay.addWidget(_pckpt_load_hdr)
        self._pred_load_ckpt_chk = QCheckBox("Load from checkpoint — skips embedding step (API key not required)")
        self._pred_load_ckpt_chk.setStyleSheet("font-size:13px; color:#333333;")
        pred_ckpt_lay.addWidget(self._pred_load_ckpt_chk)
        _pload_row = QHBoxLayout(); _pload_row.setContentsMargins(0, 0, 0, 0); _pload_row.setSpacing(6)
        self._pred_load_ckpt_edit = _PathEdit("Path to .naics_embed file…")
        _pload_row.addWidget(self._pred_load_ckpt_edit, 1)
        _pload_btn = QPushButton("Browse…"); _pload_btn.setObjectName("browseBtn"); _pload_btn.setFixedWidth(96)
        _pload_btn.clicked.connect(self._browse_load_pred_ckpt)
        _plb = QVBoxLayout(); _plb.setContentsMargins(0, 0, 0, 0); _plb.setSpacing(0)
        _plb.addWidget(_pload_btn); _plb.addStretch()
        _pload_row.addLayout(_plb)
        pred_ckpt_lay.addLayout(_pload_row)

        self._pred_ckpt_info_lbl = QLabel("")
        self._pred_ckpt_info_lbl.setWordWrap(True)
        self._pred_ckpt_info_lbl.setStyleSheet(
            "font-size:12px; color:#555555; padding:4px 6px;"
            " background:#F8F9FA; border-radius:4px; border:1px solid #E2E8F0;"
        )
        self._pred_ckpt_info_lbl.setVisible(False)
        pred_ckpt_lay.addWidget(self._pred_ckpt_info_lbl)
        left_lay.addWidget(pred_ckpt_grp)

        # Output
        out_grp = QGroupBox("Output")
        out_lay = QVBoxLayout(out_grp)
        out_lay.setContentsMargins(0, 0, 0, 0)
        out_lay.setSpacing(6)
        self.out_dir_picker = _FilePicker("", "Same folder as input file", pick_dir=True)
        _fl(out_lay, "Output folder", self.out_dir_picker)
        self.out_name_edit = QLineEdit()
        self.out_name_edit.setPlaceholderText("{input_name}_with_LLM_NAICS_Prediction_MM-DD-YYYY_HH-MM-SS.csv")
        _fl(out_lay, "Output filename (leave blank for auto)", self.out_name_edit)
        left_lay.addWidget(out_grp)

        # Update filename placeholder whenever the input file changes
        self.pred_col_selector._file_edit.document().contentsChanged.connect(
            self._update_out_name_hint
        )

        # Auto-default save-ckpt path when input file or output dir changes
        self.pred_col_selector.columns_changed.connect(self._update_pred_save_ckpt_default)
        self.out_dir_picker._edit.document().contentsChanged.connect(
            self._update_pred_save_ckpt_default
        )

        left_lay.addStretch()
        splitter.addWidget(left_scroll)

        # ─── Right: Progress & Log ────────────────────────────────────────
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_inner = QWidget()
        right_lay   = QVBoxLayout(right_inner)
        right_lay.setSpacing(12)
        right_lay.setContentsMargins(8, 0, 0, 0)
        right_scroll.setWidget(right_inner)

        # Prompt preview (read-only, from model)
        prompt_grp = QGroupBox("Prompt Template (from model — read only)")
        prompt_lay = QVBoxLayout(prompt_grp)
        prompt_lay.setContentsMargins(0, 0, 0, 0)
        self.prompt_preview = QPlainTextEdit()
        self.prompt_preview.setReadOnly(True)
        self.prompt_preview.setMinimumHeight(52)
        self.prompt_preview.setMaximumHeight(180)
        self.prompt_preview.setPlaceholderText("Load a model to see its prompt template…")
        self.prompt_preview.setStyleSheet(
            "background:#FFF8E0; color:#5C3A00; border:1px solid #FFC72C; border-radius:6px;"
        )
        prompt_lay.addWidget(self.prompt_preview)
        right_lay.addWidget(prompt_grp)

        # Progress
        prog_grp = QGroupBox("Progress")
        prog_lay = QVBoxLayout(prog_grp)
        prog_lay.setContentsMargins(0, 0, 0, 0)
        prog_lay.setSpacing(10)
        _grn_chunk = (
            "QProgressBar::chunk { background: qlineargradient("
            "x1:0,y1:0,x2:1,y2:0, stop:0 #990000, stop:1 #CC3333); border-radius:6px; }"
        )
        self.pred_embedding_bar = _LabeledBar("Embeddings")
        self.pred_embedding_bar.bar.setStyleSheet(_grn_chunk)
        self.pred_overall_bar   = _LabeledBar("Overall")
        self.pred_overall_bar.bar.setStyleSheet(_grn_chunk)
        prog_lay.addWidget(self.pred_embedding_bar)

        # Embedding metrics row
        _pred_emb_row = QHBoxLayout()
        _pred_emb_row.setSpacing(20)
        for _attr, _label in [
            ("_pred_emb_lbl_batch",  "Batches"),
            ("_pred_emb_lbl_tokens", "Tokens"),
            ("_pred_emb_lbl_eta",    "ETA"),
            ("_pred_emb_lbl_cost",   "Est. Cost"),
        ]:
            _pc = QVBoxLayout()
            _pc.setSpacing(2)
            _pt = QLabel(_label)
            _pt.setStyleSheet("font-size:13px; color:#333333;")
            _pv = QLabel("—")
            _pv.setStyleSheet("font-size:18px; font-weight:700;")
            _pc.addWidget(_pt)
            _pc.addWidget(_pv)
            setattr(self, _attr, _pv)
            _pred_emb_row.addLayout(_pc)
        _pred_emb_row.addStretch()
        prog_lay.addLayout(_pred_emb_row)

        _pred_cost_note = QLabel(
            '* Estimated cost · $0.13 / 1M tokens (text-embedding-3-large, as of Feb 20, 2026) · '
            '<a href="https://platform.openai.com/docs/pricing" style="color:#990000;">'
            'Check current rates</a>'
            ' · For reference only — actual cost varies depending on your data.'
        )
        _pred_cost_note.setOpenExternalLinks(True)
        _pred_cost_note.setStyleSheet("font-size:12px; color:#555555; margin-top:1px;")
        _pred_cost_note.setWordWrap(True)
        prog_lay.addWidget(_pred_cost_note)

        prog_lay.addWidget(self.pred_overall_bar)
        right_lay.addWidget(prog_grp)

        # Log
        log_grp = QGroupBox("Log")
        log_lay = QVBoxLayout(log_grp)
        log_lay.setContentsMargins(0, 0, 0, 0)
        self.log = QTextEdit()
        self.log.setObjectName("logView")
        self.log.setReadOnly(True)
        log_lay.addWidget(self.log)
        right_lay.addWidget(log_grp, 1)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.pred_stop_btn = QPushButton("Stop")
        self.pred_stop_btn.setObjectName("dangerBtn")
        self.pred_stop_btn.setEnabled(False)
        self.pred_stop_btn.clicked.connect(self._stop)
        self.pred_run_btn = QPushButton("Run Prediction")
        self.pred_run_btn.setObjectName("primaryBtn")
        self.pred_run_btn.setStyleSheet(
            "QPushButton { background:#FFC72C; border:none; color:#000000; font-weight:600;"
            " font-size:15px; padding:9px 28px; border-radius:8px; min-height:36px; }"
            "QPushButton:hover { background:#E6B020; }"
            "QPushButton:pressed { background:#CC9A10; }"
            "QPushButton:disabled { background:#E2E8F0; color:#555555; }"
        )
        self.pred_run_btn.clicked.connect(self._run)
        btn_row.addWidget(self.pred_stop_btn)
        btn_row.addWidget(self.pred_run_btn)
        right_lay.addLayout(btn_row)

        splitter.addWidget(right_scroll)
        splitter.setSizes([420, 520])

    # ── Actions ────────────────────────────────────────────────────────────

    def _update_out_name_hint(self):
        p = self.pred_col_selector._file_edit.toPlainText().strip()
        if p:
            stem = Path(p).stem
            self.out_name_edit.setPlaceholderText(f"{stem}_with_LLM_NAICS_Prediction_MM-DD-YYYY_HH-MM-SS.csv")
        else:
            self.out_name_edit.setPlaceholderText("{input_name}_with_LLM_NAICS_Prediction_MM-DD-YYYY_HH-MM-SS.csv")

    def _update_pred_save_ckpt_default(self):
        """Auto-fill the save-embeddings path when the input file or output dir changes."""
        current = self._pred_save_ckpt_edit.toPlainText().strip()
        if current and current != self._pred_ckpt_auto_path:
            return  # user has a manually-set path — leave it alone
        p = self.pred_col_selector._file_edit.toPlainText().strip()
        if p:
            out_dir = self.out_dir_picker.path() or str(Path(p).parent)
            new_path = str(Path(out_dir) / f"embedded_{Path(p).stem}.naics_embed")
        else:
            new_path = ""
        self._pred_ckpt_auto_path = new_path
        self._pred_save_ckpt_edit.setPlainText(new_path)

    def set_model_path(self, path: str):
        """Called by MainWindow after a successful training run."""
        self._model_path_edit.setPlainText(path)
        if os.path.isfile(path):
            self._load_model_info(path)

    def _reset_api_key_status(self):
        self._pred_api_key_status.setText("")

    def _test_api_key(self):
        key = self.pred_api_key.text().strip()
        if not key:
            QMessageBox.warning(self, "API Key", "Please enter an API key first.")
            return
        if self._api_key_tester is not None and self._api_key_tester.isRunning():
            return  # already testing — ignore duplicate click
        self._pred_api_key_status.setText("Testing…")
        self._pred_api_key_status.setStyleSheet("font-size:13px; font-weight:600; color:#333333;")
        self._api_key_tester = _ApiKeyTester(key, self)
        self._api_key_tester.success.connect(self._on_api_key_ok)
        self._api_key_tester.failure.connect(self._on_api_key_fail)
        self._api_key_tester.start()

    @pyqtSlot(str)
    def _on_api_key_ok(self, msg: str):
        self._pred_api_key_status.setText("✓  Valid")
        self._pred_api_key_status.setStyleSheet("font-size:13px; font-weight:600; color:#16A34A;")
        QMessageBox.information(self, "API Key Test", f"✓  Key is valid.\n\n{msg}")

    @pyqtSlot(str)
    def _on_api_key_fail(self, msg: str):
        self._pred_api_key_status.setText("✗  Invalid")
        self._pred_api_key_status.setStyleSheet("font-size:13px; font-weight:600; color:#DC2626;")
        QMessageBox.critical(self, "API Key Test", f"✗  Key test failed:\n\n{msg}")

    def _browse_model(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Open model", self._model_path_edit.toPlainText().strip() or APP_DIR,
            "NAICS Model (*.naics_model);;All files (*)",
        )
        if p:
            self._model_path_edit.setPlainText(p)
            self._load_model_info(p)

    def _load_model_info(self, path: str):
        try:
            info = ModelBundle.peek(path)
            tc   = info.get("training_config", {})
            acc  = tc.get("training_accuracy", None)
            acc_str = f"{acc*100:.1f}%" if acc is not None else "N/A"
            rows = tc.get("num_training_rows", "N/A")
            bundle_type = info.get("bundle_type", "single")
            categories  = info.get("categories", [])

            if bundle_type == "multi":
                self.model_info_lbl.setText(
                    f"<b>{info['model_name']}</b>  ·  "
                    f"<b>multi-model</b>: general ({info['num_classes']} classes) "
                    f"+ {len(categories)} category model(s)  ·  "
                    f"created {info['created_at']}  ·  "
                    f"train rows: {rows}  ·  general acc: {acc_str}"
                )
                # Build prompt preview: general + each category
                cat_models = info.get("category_models", {})
                lines = [f"[General]  {info['prompt_template']}"]
                for cat in categories:
                    cm = cat_models.get(cat, {})
                    lines.append(f"[{cat}]  {cm.get('prompt_template', '')}")
                self.prompt_preview.setPlainText("\n".join(lines))
                # Show category routing section and populate info label
                self._pred_cat_info_lbl.setText(
                    f"This model has {len(categories)} category sub-model(s): "
                    + ", ".join(f"<b>{c}</b>" for c in categories)
                    + ".<br>Select your data's category column below to route each row "
                    "to its matching model.  Unrecognised categories fall back to the "
                    "general model.  Leave blank to use the general model for all rows."
                )
                self._pred_cat_info_lbl.setTextFormat(Qt.RichText)
                self._pred_cat_grp.setVisible(True)
            else:
                self.model_info_lbl.setText(
                    f"<b>{info['model_name']}</b>  ·  "
                    f"{info['num_classes']} classes  ·  "
                    f"created {info['created_at']}  ·  "
                    f"train rows: {rows}  ·  train acc: {acc_str}"
                )
                self.prompt_preview.setPlainText(info["prompt_template"])
                self._pred_cat_grp.setVisible(False)
        except Exception as e:
            self.model_info_lbl.setText(f"<span style='color:#DC2626'>Cannot read model: {e}</span>")
            self._pred_cat_grp.setVisible(False)

    def _browse_save_pred_ckpt(self):
        p, _ = QFileDialog.getSaveFileName(
            self, "Save Prediction Checkpoint",
            self._pred_save_ckpt_edit.toPlainText().strip() or APP_DIR,
            "NAICS Embedding Checkpoint (*.naics_embed);;All files (*)",
        )
        if p:
            if not p.endswith(".naics_embed"):
                p += ".naics_embed"
            self._pred_save_ckpt_edit.setPlainText(p)

    def _browse_load_pred_ckpt(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Open Prediction Checkpoint",
            self._pred_load_ckpt_edit.toPlainText().strip() or APP_DIR,
            "NAICS Embedding Checkpoint (*.naics_embed);;All files (*)",
        )
        if p:
            self._pred_load_ckpt_edit.setPlainText(p)
            self._load_pred_ckpt_info(p)

    def _load_pred_ckpt_info(self, path: str):
        try:
            from embed_checkpoint import peek_ckpt
            meta = peek_ckpt(path)
            n_rows = meta.get("n_rows", "?")
            n_dims = meta.get("n_dims", "?")
            cfg_s  = meta.get("config", {})
            desc   = cfg_s.get("desc_col", "")
            text = (f"{n_rows:,} rows | {n_dims} dims"
                    if isinstance(n_rows, int) else f"rows: {n_rows}")
            if desc:
                text += f" | desc_col: {desc}"
            text += f" | created {meta.get('created_at', '')}"
            self._pred_ckpt_info_lbl.setText(text)
            self._pred_ckpt_info_lbl.setVisible(True)
        except Exception as e:
            self._pred_ckpt_info_lbl.setText(f"Cannot read checkpoint: {e}")
            self._pred_ckpt_info_lbl.setVisible(True)

    def _run(self):
        col_vals   = self.pred_col_selector.get_values()
        model_path = self._model_path_edit.toPlainText().strip()
        errors     = []
        if not model_path:
            errors.append("Model file is required.")
        elif not os.path.isfile(model_path):
            errors.append(f"Model file not found:\n  {model_path}")

        load_ckpt  = self._pred_load_ckpt_chk.isChecked()
        load_ckpt_path = self._pred_load_ckpt_edit.toPlainText().strip() if load_ckpt else ""
        save_ckpt  = self._pred_save_ckpt_chk.isChecked()
        save_ckpt_path = self._pred_save_ckpt_edit.toPlainText().strip() if save_ckpt else ""

        if load_ckpt:
            if not load_ckpt_path:
                errors.append("Checkpoint file path is required when loading embeddings.")
            elif not os.path.isfile(load_ckpt_path):
                errors.append(f"Checkpoint file not found:\n  {load_ckpt_path}")
        else:
            input_file = col_vals.get("input_file", "")
            if not input_file:
                errors.append("Input file is required.")
            elif not os.path.isfile(input_file):
                errors.append(f"Input file not found:\n  {input_file}")
            if not col_vals["desc_col"]:
                errors.append("Description column must be selected.")
            if not self.pred_api_key.text().strip():
                errors.append("OpenAI API key is required.")

        if save_ckpt and save_ckpt_path:
            _ckpt_parent = Path(save_ckpt_path).parent
            if not _ckpt_parent.is_dir():
                errors.append(
                    f"Checkpoint save directory does not exist:\n  {_ckpt_parent}\n"
                    "Please create it or choose a different path."
                )

        if errors:
            QMessageBox.warning(self, "Cannot start — please fix the following", "\n\n".join(errors))
            return

        pred_category_col = (
            self._pred_cat_col_combo.currentText().strip()
            if self._pred_cat_grp.isVisible() else ""
        )

        # Build output path — use checkpoint path as base when loading
        _base_for_output = Path(load_ckpt_path if load_ckpt else col_vals.get("input_file", ""))
        out_dir  = self.out_dir_picker.path() or str(_base_for_output.parent or APP_DIR)
        out_name = self.out_name_edit.text().strip()
        if not out_name:
            _ts = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            out_name = f"{_base_for_output.stem or 'prediction'}_with_LLM_NAICS_Prediction_{_ts}.csv"
        if not out_name.lower().endswith(".csv"):
            out_name += ".csv"
        out_path = os.path.join(out_dir, out_name)

        config = {
            **col_vals,
            "model_path":      self._model_path_edit.toPlainText().strip(),
            "api_key":         self.pred_api_key.text().strip(),
            "topk":            self.topk_spin.value(),
            "batch_size":      self.pred_batch_spin.value(),
            "output_file":     out_path,
            "category_col":    pred_category_col,
            "save_ckpt_path":  save_ckpt_path,
            "load_ckpt_path":  load_ckpt_path,
        }

        self.log.clear()
        self.pred_embedding_bar.reset()
        self._pred_emb_lbl_batch.setText("—")
        self._pred_emb_lbl_tokens.setText("—")
        self._pred_emb_lbl_eta.setText("—")
        self._pred_emb_lbl_cost.setText("—")
        self.pred_overall_bar.reset()

        if pred_category_col:
            _append_log(self.log,
                f"Category routing enabled  ·  column: '{pred_category_col}'", "info")

        self._thread = _WorkerThread(PredictWorker, config, self)
        self._thread.progress.connect(self._on_progress)
        self._thread.finished.connect(self._on_done)
        self._thread.start()
        self.pred_run_btn.setEnabled(False)
        self.pred_stop_btn.setEnabled(True)
        self.status_message.emit("Prediction running…")
        _append_log(self.log, "Prediction started.", "info")

    def _on_pred_col_headers_changed(self, cols: list):
        """Repopulate the prediction category combo when input file headers change."""
        current = self._pred_cat_col_combo.currentText()
        self._pred_cat_col_combo.clear()
        self._pred_cat_col_combo.addItem("")
        for c in cols:
            self._pred_cat_col_combo.addItem(c)
        if current and self._pred_cat_col_combo.findText(current) >= 0:
            self._pred_cat_col_combo.setCurrentText(current)

    def _stop(self):
        if self._thread:
            self._thread.stop()
        self.pred_stop_btn.setEnabled(False)
        _append_log(self.log, "Stop requested.", "warning")

    @pyqtSlot(dict)
    def _on_progress(self, d: dict):
        msg_type = d.get("type", "")

        if msg_type == "log":
            _append_log(self.log, d["message"], d.get("level", "info"))
            return

        if msg_type == "auth_error":
            QMessageBox.critical(
                self, "Invalid API Key",
                "Your OpenAI API key was rejected.\n\n"
                "Please check that the key is correct and has sufficient credits.\n\n"
                f"Detail: {d.get('message', '')}",
            )
            return

        stage = d.get("stage", "")
        pct   = int(d.get("pct", 0))

        if stage == "embedding":
            done          = d.get("done", 0)
            total         = d.get("total", 1)
            tok           = d.get("tokens_total", 0)
            eta           = d.get("eta_model", "—")
            batch_done    = d.get("batch_done", "—")
            total_batches = d.get("total_batches", "—")
            cost_usd      = tok / 1_000_000 * 0.13
            self.pred_embedding_bar.set_value(pct, f"{pct:.1f}%")
            self._pred_emb_lbl_batch.setText(f"{batch_done}/{total_batches}")
            self._pred_emb_lbl_tokens.setText(f"{tok:,}")
            self._pred_emb_lbl_eta.setText(str(eta))
            self._pred_emb_lbl_cost.setText(f"~${cost_usd:.4f}")
            # Embedding = 0–80 % of overall
            self.pred_overall_bar.set_value(int(pct * 0.8), f"{int(pct * 0.8)}%")

        elif stage == "predicting":
            overall = 80 + int(pct * 0.2)
            self.pred_overall_bar.set_value(overall, f"{overall}%")

        elif stage == "done":
            self.pred_overall_bar.set_value(100, "100% — Done")
            self.pred_overall_bar.set_success()

    @pyqtSlot(object)
    def _on_done(self, result):
        self.pred_run_btn.setEnabled(True)
        self.pred_stop_btn.setEnabled(False)
        if result:
            _append_log(self.log, f"✓ Output: {result}", "success")
            self.status_message.emit(f"Prediction complete: {Path(result).name}")
            QMessageBox.information(self, "Prediction Complete",
                f"Output saved:\n{result}")
        else:
            _append_log(self.log, "Prediction did not complete.", "warning")
            self.status_message.emit("Prediction stopped or failed.")

    # ── Config ─────────────────────────────────────────────────────────────

    def save_config(self) -> dict:
        save_key = self.pred_save_key_chk.isChecked()
        return {
            "pred_model_path":     self._model_path_edit.toPlainText().strip(),
            "pred_api_key":        self.pred_api_key.text() if save_key else "",
            "pred_save_key":       save_key,
            "pred_topk":           self.topk_spin.value(),
            "pred_batch_size":     self.pred_batch_spin.value(),
            "pred_out_dir":        self.out_dir_picker.path(),
            "pred_out_name":       self.out_name_edit.text().strip(),
            "pred_category_col":   self._pred_cat_col_combo.currentText(),
            "pred_save_ckpt":      self._pred_save_ckpt_chk.isChecked(),
            "pred_save_ckpt_path": self._pred_save_ckpt_edit.toPlainText().strip(),
            "pred_load_ckpt":      self._pred_load_ckpt_chk.isChecked(),
            "pred_load_ckpt_path": self._pred_load_ckpt_edit.toPlainText().strip(),
            **{f"pred_{k}": v for k, v in self.pred_col_selector.get_values().items()},
        }

    def load_config(self, d: dict):
        path = d.get("pred_model_path", "")
        self._model_path_edit.setPlainText(path or "")
        if path and os.path.isfile(path):
            self._load_model_info(path)
        save_key = d.get("pred_save_key", False)
        self.pred_save_key_chk.blockSignals(True)
        self.pred_save_key_chk.setChecked(save_key)
        self.pred_save_key_chk.blockSignals(False)
        if save_key:
            self.pred_api_key.setText(d.get("pred_api_key", ""))
        self.topk_spin.setValue(d.get("pred_topk", 3))
        self.pred_batch_spin.setValue(d.get("pred_batch_size", 200))
        self.out_dir_picker.set_path(d.get("pred_out_dir", ""))
        self.out_name_edit.setText(d.get("pred_out_name", ""))
        self._pred_save_ckpt_chk.setChecked(d.get("pred_save_ckpt", False))
        self._pred_save_ckpt_edit.setPlainText(d.get("pred_save_ckpt_path", ""))
        self._pred_load_ckpt_chk.setChecked(d.get("pred_load_ckpt", False))
        pred_load_p = d.get("pred_load_ckpt_path", "")
        self._pred_load_ckpt_edit.setPlainText(pred_load_p)
        if pred_load_p and os.path.isfile(pred_load_p):
            self._load_pred_ckpt_info(pred_load_p)
        # Restore prediction input file and column selections
        pred_col_keys = (
            "input_file", "sheet_name",
            "desc_col", "supplier_col", "supplier_absent",
        )
        pred_col_d = {k: d[f"pred_{k}"] for k in pred_col_keys if f"pred_{k}" in d}
        if pred_col_d:
            self.pred_col_selector.set_values(pred_col_d)
        # Restore category column after columns may be loaded via pred_col_selector
        saved_cat = d.get("pred_category_col", "")
        if saved_cat and self._pred_cat_col_combo.findText(saved_cat) >= 0:
            self._pred_cat_col_combo.setCurrentText(saved_cat)


# ═══════════════════════════════════════════════════════════════════════════
#  Main Window
# ═══════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scope 3 Category 1 Emissions - NAICS Classifier")
        self.setMinimumSize(1040, 720)
        self._build_ui()
        self._load_config()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header bar (colour switches with active tab)
        self.header = QWidget()
        header = self.header
        header.setFixedHeight(56)
        header.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #6B0000, stop:1 #990000);"
        )
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(20, 0, 20, 0)

        # Logo (top-left)
        logo_lbl = QLabel()
        logo_lbl.setStyleSheet("background: transparent;")
        _logo_pix = QPixmap(os.path.join(ASSETS_DIR, "AsgmtEarth_Circle_small.png"))
        if not _logo_pix.isNull():
            logo_lbl.setPixmap(
                _logo_pix.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        logo_lbl.setFixedSize(44, 44)
        h_lay.addWidget(logo_lbl)
        h_lay.addSpacing(8)

        title_lbl = QLabel("Scope 3 Category 1 — NAICS Classifier")
        title_lbl.setStyleSheet(
            "color:white; font-size:18px; font-weight:700; background:transparent;"
        )
        subtitle = QLabel("OpenAI Embeddings + XGBoost")
        subtitle.setStyleSheet(
            "color:rgba(255,255,255,0.7); font-size:14px; background:transparent; margin-left:12px;"
        )
        h_lay.addWidget(title_lbl)
        h_lay.addWidget(subtitle)
        h_lay.addStretch()

        # Segmented toggle pill (Train | Predict)
        seg_container = QWidget()
        seg_container.setObjectName("segPill")
        seg_container.setFixedHeight(40)
        # Use #segPill so the rule does NOT propagate to child QPushButtons
        seg_container.setStyleSheet(
            "#segPill { background: rgba(255,255,255,0.15); border-radius: 9px;"
            " border: 1px solid rgba(255,255,255,0.30); }"
        )
        seg_lay = QHBoxLayout(seg_container)
        seg_lay.setContentsMargins(3, 3, 3, 3)
        seg_lay.setSpacing(2)
        self._train_btn = QPushButton("Train")
        self._pred_btn  = QPushButton("Predict")
        for _b in (self._train_btn, self._pred_btn):
            _b.setCursor(Qt.PointingHandCursor)
            _b.setFixedHeight(34)   # 40 - 3 - 3 = 34
        seg_lay.addWidget(self._train_btn)
        seg_lay.addWidget(self._pred_btn)
        h_lay.addWidget(seg_container)

        h_lay.addSpacing(12)

        reset_btn = QPushButton("↺  Reset Config")
        reset_btn.setCursor(Qt.PointingHandCursor)
        reset_btn.setStyleSheet(
            "QPushButton { background: #FFC72C; border: none; color: #000000;"
            " border-radius: 6px; padding: 2px 12px; font-size: 14px; font-weight: 600;"
            " min-height: 0; max-height: 9999px; }"
            "QPushButton:hover { background: #E6B020; }"
            "QPushButton:pressed { background: #CC9A10; }"
        )
        reset_btn.clicked.connect(self._reset_config)
        h_lay.addWidget(reset_btn)

        root.addWidget(header)

        # Stacked widget (replaces QTabWidget — no visible tab bar)
        self.train_tab   = TrainTab(self)
        self.predict_tab = PredictTab(self)

        self._stack = QStackedWidget()
        self._stack.addWidget(self.train_tab)
        self._stack.addWidget(self.predict_tab)
        root.addWidget(self._stack, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        self.train_tab.status_message.connect(self.status_bar.showMessage)
        self.predict_tab.status_message.connect(self.status_bar.showMessage)
        self.train_tab.model_ready.connect(self.predict_tab.set_model_path)

        self._train_btn.clicked.connect(lambda: self._switch_tab(0))
        self._pred_btn.clicked.connect(lambda: self._switch_tab(1))
        self._switch_tab(0)  # set initial active state

        # Sync API keys and save-key checkbox between tabs
        self._syncing_key = False
        self.train_tab.api_key_edit.textChanged.connect(self._sync_key_train_to_pred)
        self.predict_tab.pred_api_key.textChanged.connect(self._sync_key_pred_to_train)
        self.train_tab.save_key_chk.toggled.connect(self._sync_save_chk_train_to_pred)
        self.predict_tab.pred_save_key_chk.toggled.connect(self._sync_save_chk_pred_to_train)

        # Auto-save: debounce 600 ms after any change
        self._loading = False
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._save_config)
        S  = self._schedule_save
        tt = self.train_tab
        pt = self.predict_tab
        for _sig in [
            tt.model_dir_picker._edit.document().contentsChanged,
            tt.model_name_edit.textChanged,
            tt.col_selector._file_edit.document().contentsChanged,
            tt.col_selector.sheet_edit.textChanged,
            tt.col_selector.desc_combo.currentIndexChanged,
            tt.col_selector.supplier_combo.currentIndexChanged,
            tt.col_selector.no_supplier_chk.toggled,
            tt.col_selector.label_combo.currentIndexChanged,
            tt.col_selector.naics_desc_combo.currentIndexChanged,
            tt.col_selector.no_naics_desc_chk.toggled,
            tt.api_key_edit.textChanged,
            tt.save_key_chk.toggled,
            tt.batch_spin.valueChanged,
            tt.depth_spin.valueChanged,
            tt.rounds_spin.valueChanged,
            tt.use_cat_chk.toggled,
            tt._cat_col_combo.currentIndexChanged,
            tt._train_save_ckpt_chk.toggled,
            tt._train_save_ckpt_edit.document().contentsChanged,
            tt._train_load_ckpt_chk.toggled,
            tt._train_load_ckpt_edit.document().contentsChanged,
            pt._model_path_edit.document().contentsChanged,
            pt.pred_col_selector._file_edit.document().contentsChanged,
            pt.pred_col_selector.sheet_edit.textChanged,
            pt.pred_col_selector.desc_combo.currentIndexChanged,
            pt.pred_col_selector.supplier_combo.currentIndexChanged,
            pt.pred_col_selector.no_supplier_chk.toggled,
            pt.pred_api_key.textChanged,
            pt.pred_save_key_chk.toggled,
            pt.topk_spin.valueChanged,
            pt.pred_batch_spin.valueChanged,
            pt.out_dir_picker._edit.document().contentsChanged,
            pt.out_name_edit.textChanged,
            pt._pred_cat_col_combo.currentIndexChanged,
            pt._pred_save_ckpt_chk.toggled,
            pt._pred_save_ckpt_edit.document().contentsChanged,
            pt._pred_load_ckpt_chk.toggled,
            pt._pred_load_ckpt_edit.document().contentsChanged,
        ]:
            _sig.connect(S)

    def _sync_key_train_to_pred(self, text: str):
        if self._syncing_key:
            return
        self._syncing_key = True
        self.predict_tab.pred_api_key.setText(text)
        self._syncing_key = False

    def _sync_key_pred_to_train(self, text: str):
        if self._syncing_key:
            return
        self._syncing_key = True
        self.train_tab.api_key_edit.setText(text)
        self._syncing_key = False

    def _sync_save_chk_train_to_pred(self, checked: bool):
        if self.predict_tab.pred_save_key_chk.isChecked() != checked:
            self.predict_tab.pred_save_key_chk.setChecked(checked)

    def _sync_save_chk_pred_to_train(self, checked: bool):
        if self.train_tab.save_key_chk.isChecked() != checked:
            self.train_tab.save_key_chk.setChecked(checked)

    def _switch_tab(self, idx: int):
        self._stack.setCurrentIndex(idx)
        self._on_tab_changed(idx)

    def _on_tab_changed(self, idx: int):
        _SEL_BASE = (
            "QPushButton { background: #FFC72C; border: none; border-radius: 7px;"
            " padding: 4px 20px; font-weight: 700; font-size: 15px; color: #000000;"
            " min-height: 0; max-height: 9999px; }"
        )
        _UNSEL = (
            "QPushButton { background: transparent; color: rgba(255,255,255,0.82); border: none;"
            " border-radius: 7px; padding: 4px 20px; font-weight: 500; font-size: 15px;"
            " min-height: 0; max-height: 9999px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.15); }"
        )
        self.header.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #6B0000, stop:1 #990000);"
        )
        if idx == 1:
            self._train_btn.setStyleSheet(_UNSEL)
            self._pred_btn.setStyleSheet(_SEL_BASE)
        else:
            self._train_btn.setStyleSheet(_SEL_BASE)
            self._pred_btn.setStyleSheet(_UNSEL)

    # ── Config ─────────────────────────────────────────────────────────────

    def _schedule_save(self, *_):
        """Debounced auto-save — fires 600 ms after the last change."""
        if not self._loading:
            self._save_timer.start(600)

    def _load_config(self):
        self._loading = True
        try:
            if not os.path.isfile(CONFIG_PATH):
                return
            try:
                with open(CONFIG_PATH, encoding="utf-8") as f:
                    d = json.load(f)
                self.train_tab.load_config(d)
                self.predict_tab.load_config(d)
                if d.get("train_save_key", False) or d.get("pred_save_key", False):
                    _warn_save_key(self)
            except Exception as e:
                self.status_bar.showMessage(f"Could not load config: {e}")
        finally:
            self._loading = False

    def _save_config(self):
        cfg = {}
        cfg.update(self.train_tab.save_config())
        cfg.update(self.predict_tab.save_config())
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        except Exception as e:
            self.status_bar.showMessage(f"Could not save config: {e}")

    def _reset_config(self):
        reply = QMessageBox.question(
            self, "Reset Configuration",
            "Clear all saved settings and reset to defaults?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._loading = True
        try:
            if os.path.isfile(CONFIG_PATH):
                os.remove(CONFIG_PATH)
            self.train_tab.load_config({})
            self.predict_tab.load_config({})
            # Explicitly clear API key fields (not cleared by load_config when save_key=False)
            self.train_tab.api_key_edit.clear()
            self.predict_tab.pred_api_key.clear()
            # Reset model output dir to app directory
            self.train_tab.model_dir_picker.set_path(APP_DIR)
        finally:
            self._loading = False
        self.status_bar.showMessage("Configuration reset to defaults.")

    def closeEvent(self, event):
        self._save_config()
        super().closeEvent(event)


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    # macOS: prevent App Nap during long tasks
    try:
        from Foundation import NSBundle, NSProcessInfo
        info = NSProcessInfo.processInfo()
        info.beginActivityWithOptions_reason_(
            0x00FFFFFF, "NAICS Classifier training"
        )
    except ImportError:
        pass

    # Write checkmark SVG so QSS can reference it for checkbox indicators
    _chk_svg = os.path.join(ASSETS_DIR, "_check.svg")
    try:
        with open(_chk_svg, "w", encoding="utf-8") as _f:
            _f.write(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
                '<polyline points="3,8.5 6.5,12 13,4.5" stroke="white"'
                ' stroke-width="2.2" fill="none"'
                ' stroke-linecap="round" stroke-linejoin="round"/></svg>'
            )
    except OSError:
        _chk_svg = ""

    app = QApplication(sys.argv)
    app.setApplicationName("Scope 3 Category 1 Emissions - NAICS Classifier")
    app.setStyleSheet(make_stylesheet(_chk_svg))

    # HiDPI: these attributes were removed in Qt6 (automatic); kept for Qt5 compat
    try:
        app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except AttributeError:
        pass

    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
