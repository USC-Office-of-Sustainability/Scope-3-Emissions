"""
styles.py - Modern QSS stylesheet for NAICS Classifier.
"""

# Colour palette
C_BG        = "#FFFFFF"
C_BG_SURFACE = "#F8FAFC"
C_BG_MUTED   = "#F1F5F9"
C_BORDER     = "#E2E8F0"
C_BORDER_FOCUS = "#990000"

C_TEXT       = "#000000"
C_TEXT_MUTED = "#333333"
C_TEXT_LIGHT = "#555555"

C_PRIMARY    = "#990000"     # USC Red
C_PRIMARY_HOVER = "#7A0000"
C_PRIMARY_PRESS = "#5C0000"
C_PRIMARY_BG = "#FFF0F0"

C_SUCCESS    = "#16A34A"
C_SUCCESS_BG = "#F0FDF4"
C_WARNING    = "#D97706"
C_WARNING_BG = "#FFFBEB"
C_ERROR      = "#DC2626"
C_ERROR_BG   = "#FEF2F2"

C_SECONDARY      = "#64748B"
C_SECONDARY_HOVER = "#475569"

C_PROGRESS_TRACK = "#E2E8F0"
C_PROGRESS_FILL  = "#990000"

def make_stylesheet(check_icon: str = "") -> str:
    """Return the full QSS stylesheet. Pass check_icon as absolute SVG path for checkbox ticks."""
    _chk = f'image: url("{check_icon.replace(chr(92), "/")}");' if check_icon else ""
    return f"""
/* ── Base ─────────────────────────────────────────────────────── */
QWidget {{
    font-family: -apple-system, "SF Pro Text", "Segoe UI", sans-serif;
    font-size: 15px;
    color: {C_TEXT};
    background-color: {C_BG};
}}

QMainWindow {{
    background-color: {C_BG};
}}

/* ── Tab Widget ───────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    background: {C_BG};
    top: -1px;
}}

QTabBar::tab {{
    background: {C_BG_MUTED};
    color: {C_TEXT_MUTED};
    border: 1px solid {C_BORDER};
    border-bottom: none;
    padding: 9px 22px;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
    margin-right: 3px;
    font-weight: 500;
    font-size: 15px;
}}
QTabBar::tab:selected {{
    background: {C_BG};
    color: {C_PRIMARY};
    border-bottom: 2px solid {C_PRIMARY};
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{
    background: {C_PRIMARY_BG};
    color: {C_PRIMARY};
}}

/* ── Group Box ────────────────────────────────────────────────── */
QGroupBox {{
    background: {C_BG_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    margin-top: 18px;
    padding: 12px 14px 10px 14px;
    font-weight: 600;
    font-size: 14px;
    color: {C_TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    left: 12px;
    top: 2px;
    background: {C_BG};
    border-radius: 4px;
}}

/* ── Labels ───────────────────────────────────────────────────── */
QLabel {{
    color: {C_TEXT};
    background: transparent;
}}
QLabel#sectionTitle {{
    font-weight: 700;
    font-size: 17px;
    color: {C_TEXT};
}}
QLabel#hint {{
    color: {C_TEXT_LIGHT};
    font-size: 13px;
}}
QLabel#statusGood {{
    color: {C_SUCCESS};
    font-weight: 500;
}}
QLabel#statusBad {{
    color: {C_ERROR};
    font-weight: 500;
}}

/* ── Inputs ───────────────────────────────────────────────────── */
QLineEdit, QSpinBox, QTextEdit, QPlainTextEdit {{
    background: {C_BG};
    border: 1.5px solid {C_BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    color: {C_TEXT};
    selection-background-color: {C_PRIMARY_BG};
}}
QLineEdit:focus, QSpinBox:focus, QTextEdit:focus {{
    border-color: {C_BORDER_FOCUS};
    background: {C_BG};
}}
QLineEdit[echoMode="2"] {{
    font-family: monospace;
}}

QSpinBox::up-button, QSpinBox::down-button {{
    border: none;
    background: {C_BG_MUTED};
    width: 18px;
    border-radius: 3px;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
    background: {C_BORDER};
}}

/* ── ComboBox ─────────────────────────────────────────────────── */
QComboBox {{
    combobox-popup: 0;
    background: {C_BG};
    border: 1.5px solid {C_BORDER};
    border-radius: 6px;
    padding: 5px 32px 5px 10px;
    color: {C_TEXT};
    min-height: 22px;
}}
QComboBox:focus {{
    border-color: {C_BORDER_FOCUS};
}}
QComboBox::drop-down {{
    subcontrol-origin: border;
    subcontrol-position: right center;
    width: 28px;
    border: none;
    border-left: none;
    background: transparent;
}}
QComboBox::down-arrow {{
    width: 0; height: 0;
    border-left:  5px solid transparent;
    border-right: 5px solid transparent;
    border-top:   5px solid {C_TEXT_MUTED};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    border: 1.5px solid {C_BORDER};
    border-radius: 6px;
    background: {C_BG};
    outline: none;
    padding: 3px;
    selection-background-color: {C_PRIMARY};
    selection-color: white;
}}
QComboBox QAbstractItemView::item {{
    padding: 5px 10px;
    min-height: 22px;
    color: {C_TEXT};
    background: transparent;
    border-radius: 4px;
}}
QComboBox QAbstractItemView::item:hover {{
    background: {C_PRIMARY_BG};
    color: {C_PRIMARY};
}}
QComboBox QAbstractItemView::item:selected {{
    background: {C_PRIMARY};
    color: white;
}}
QComboBox QAbstractItemView QScrollBar:vertical {{
    width: 5px;
    background: transparent;
    border: none;
    margin: 2px 1px;
}}
QComboBox QAbstractItemView QScrollBar::handle:vertical {{
    background: {C_BORDER};
    border-radius: 2px;
    min-height: 16px;
}}
QComboBox QAbstractItemView QScrollBar::add-line:vertical,
QComboBox QAbstractItemView QScrollBar::sub-line:vertical {{
    height: 0;
}}

/* ── Buttons ──────────────────────────────────────────────────── */
QPushButton {{
    background: {C_BG_MUTED};
    border: 1.5px solid {C_BORDER};
    border-radius: 7px;
    padding: 7px 16px;
    color: {C_TEXT};
    font-weight: 500;
    min-height: 28px;
}}
QPushButton:hover {{
    background: {C_BORDER};
    border-color: {C_BORDER_FOCUS};
}}
QPushButton:pressed {{
    background: {C_BORDER};
}}

QPushButton#primaryBtn {{
    background: #FFC72C;
    border: none;
    color: #000000;
    font-weight: 600;
    font-size: 15px;
    padding: 9px 28px;
    border-radius: 8px;
    min-height: 36px;
}}
QPushButton#primaryBtn:hover {{
    background: #E6B020;
}}
QPushButton#primaryBtn:pressed {{
    background: #CC9A10;
}}
QPushButton#primaryBtn:disabled {{
    background: {C_BORDER};
    color: {C_TEXT_LIGHT};
}}

QPushButton#dangerBtn {{
    background: {C_ERROR_BG};
    border: 1.5px solid {C_ERROR};
    color: {C_ERROR};
    font-weight: 600;
    padding: 9px 22px;
    border-radius: 8px;
    min-height: 36px;
}}
QPushButton#dangerBtn:hover {{
    background: {C_ERROR};
    color: white;
}}
QPushButton#dangerBtn:disabled {{
    background: {C_BG_MUTED};
    border-color: {C_BORDER};
    color: {C_TEXT_LIGHT};
}}

QPushButton#browseBtn {{
    padding: 5px 12px;
    min-height: 24px;
    font-size: 14px;
    color: {C_PRIMARY};
    border-color: {C_PRIMARY};
    background: {C_PRIMARY_BG};
}}
QPushButton#browseBtn:hover {{
    background: {C_PRIMARY};
    color: white;
}}

/* ── Progress Bar ─────────────────────────────────────────────── */
QProgressBar {{
    background: {C_PROGRESS_TRACK};
    border: none;
    border-radius: 6px;
    height: 12px;
    text-align: center;
    font-size: 13px;
    font-weight: 600;
    color: {C_TEXT};
}}
QProgressBar::chunk {{
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 {C_PRIMARY}, stop:1 #CC3333
    );
    border-radius: 6px;
}}
QProgressBar#successBar::chunk {{
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 {C_SUCCESS}, stop:1 #4ADE80
    );
}}

/* ── Log / Text area ──────────────────────────────────────────── */
QTextEdit#logView {{
    background: {C_BG_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    color: {C_TEXT};
    font-family: "JetBrains Mono", "Fira Code", "Menlo", "Consolas", monospace;
    font-size: 14px;
    padding: 8px;
}}

/* ── Scroll bars ──────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {C_BG_MUTED};
    width: 8px;
    border-radius: 4px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {C_BORDER};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {C_TEXT_MUTED};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    height: 8px;
    background: {C_BG_MUTED};
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {C_BORDER};
    border-radius: 4px;
}}

/* ── Separator ────────────────────────────────────────────────── */
QFrame[frameShape="4"], QFrame[frameShape="HLine"] {{
    color: {C_BORDER};
    max-height: 1px;
    background: {C_BORDER};
    border: none;
}}

/* ── Checkbox ─────────────────────────────────────────────────── */
QCheckBox {{
    spacing: 7px;
    color: {C_TEXT};
    background: transparent;
}}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1.5px solid {C_BORDER};
    border-radius: 4px;
    background: {C_BG};
}}
QCheckBox::indicator:checked {{
    background: {C_PRIMARY};
    border-color: {C_PRIMARY};
    {_chk}
}}
QCheckBox::indicator:checked:hover {{
    background: {C_PRIMARY_HOVER};
    border-color: {C_PRIMARY_HOVER};
    {_chk}
}}
QCheckBox::indicator:hover {{
    border-color: {C_BORDER_FOCUS};
}}

/* ── ComboBox disabled ────────────────────────────────────────── */
QComboBox:disabled {{
    background: {C_BG_MUTED};
    color: {C_TEXT_LIGHT};
    border-color: {C_BORDER};
}}

/* ── Status bar ───────────────────────────────────────────────── */
QStatusBar {{
    background: {C_BG_MUTED};
    border-top: 1px solid {C_BORDER};
    color: {C_TEXT_MUTED};
    font-size: 14px;
    padding: 2px 8px;
}}

/* ── Tool tip ─────────────────────────────────────────────────── */
QToolTip {{
    background: {C_TEXT};
    color: white;
    border: none;
    border-radius: 5px;
    padding: 4px 8px;
    font-size: 14px;
}}
"""


# Log-level HTML colours (tuned for light background)
LOG_COLORS = {
    "info":    "#1E3A5F",   # dark blue
    "success": "#14532D",   # dark green — distinct from info, accessible
    "warning": "#B45309",   # amber-700
    "error":   "#990000",   # USC Red
}
