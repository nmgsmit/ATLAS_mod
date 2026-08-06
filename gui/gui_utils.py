from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QSpinBox, QProgressBar)


def create_parameter_box(min_val: float, max_val: float, text: str, step: float=1, callback=None):
    layout = QHBoxLayout()

    dial = QSpinBox()
    dial.setMaximumHeight(28)
    dial.setMaximumWidth(150)
    dial.setMinimum(min_val)
    dial.setMaximum(max_val)
    dial.setAlignment(Qt.AlignmentFlag.AlignRight)
    dial.setSingleStep(step)
    dial.valueChanged.connect(callback)

    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignRight)

    layout.addWidget(label)
    layout.addWidget(dial)

    return dial, layout


def create_gauge(text: str):
    layout = QHBoxLayout()

    gauge = QProgressBar()
    gauge.setMaximumHeight(28)
    gauge.setMaximumWidth(200)
    gauge.setAlignment(Qt.AlignmentFlag.AlignCenter)

    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignRight)

    layout.addWidget(label)
    layout.addWidget(gauge)

    return gauge, layout


def fit_button_text(button, padding: int = 28):
    """Make a button at least as wide as its own label.

    Qt sizes a button from its text, but the stylesheet's horizontal padding and the
    checked-state border are added on top of that, so a long label ends up elided.
    """
    width = button.fontMetrics().horizontalAdvance(button.text()) + padding
    if button.minimumWidth() < width:
        button.setMinimumWidth(width)


def apply_to_all_children_widget(layout, func):
    # deliberately non-recursive
    for i in range(layout.count()):
        w = layout.itemAt(i).widget()
        if w is not None:       # nested layouts have no widget of their own
            func(w)


def workspace_name_for_video(video_path: str, raw_videos_root: str = None) -> str:
    """Build a workspace folder name that preserves raw-video subfolders.

    If the video lives under a raw_videos/raw-videos root, include the relative path under
    that root and flatten path separators so different folders do not collide.
    """
    video = Path(video_path)

    rel_parts = None
    if raw_videos_root:
        try:
            rel_parts = video.resolve().relative_to(Path(raw_videos_root).resolve()).parts
        except Exception:
            rel_parts = None

    if rel_parts is None:
        parts = video.parts
        for marker in ('raw_videos', 'raw-videos'):
            if marker in parts:
                rel_parts = parts[parts.index(marker) + 1:]
                break

    if not rel_parts:
        rel_parts = (video.name,)

    return '__'.join(rel_parts)


def workspace_path_for_video(video_path: str, workspace_root: str,
                             raw_videos_root: str = None) -> str:
    return str(Path(workspace_root) / workspace_name_for_video(video_path, raw_videos_root))


def legacy_workspace_path_for_video(video_path: str, workspace_root: str) -> str:
    return str(Path(workspace_root) / Path(video_path).name)