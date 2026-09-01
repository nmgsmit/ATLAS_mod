import os
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where source videos are looked for, in the order they are offered. Relative entries are
# tried against the working directory and then the project root, and one that does not
# exist is simply skipped -- so listing several costs nothing. '../data' is a sibling of
# the project rather than a folder inside it, which is why more than one root is needed
# at all: the recordings live next to the checkout, not in it.
DEFAULT_VIDEO_ROOTS = ('raw_videos', 'raw-videos', '../data')


def _existing_dirs(candidates) -> list:
    """The candidates that are real directories, resolved and deduplicated, in order.
    A relative path is tried against the working directory and then the project root."""
    out, seen = [], set()
    for candidate in candidates:
        if not candidate:
            continue
        p = Path(candidate)
        bases = [Path('.')] if p.is_absolute() else [Path.cwd(), PROJECT_ROOT]
        for base in bases:
            full = (base / p).resolve()
            if full.is_dir() and str(full) not in seen:
                seen.add(str(full))
                out.append(str(full))
                break
    return out


def video_roots(primary=None, extra=None) -> list:
    """Existing folders to look for source videos in: `primary` first, then `extra`.

    What is configured is honoured exactly, so `extra=[]` really does mean "only the
    primary root" -- otherwise the setting could add roots but never remove one. The
    built-in defaults are a fallback for when nothing configured exists at all (an old
    config naming a folder that was since renamed), not an always-on addition.
    """
    def flatten(*groups):
        out = []
        for g in groups:
            if g:
                out += [g] if isinstance(g, (str, Path)) else list(g)
        return out

    return (_existing_dirs(flatten(primary, extra))
            or _existing_dirs(DEFAULT_VIDEO_ROOTS))


def root_for_video(video_path: str, roots) -> str:
    """Which of `roots` contains this video -- the most specific one, so the answer does
    not depend on the order they happen to be configured in. None if none of them does."""
    if not roots:
        return None
    if isinstance(roots, (str, Path)):
        roots = [roots]
    try:
        video = Path(video_path).resolve()
    except OSError:
        return None
    best = None
    for root in roots:
        try:
            rp = Path(root).resolve()
            video.relative_to(rp)
        except (ValueError, OSError):
            continue
        if best is None or len(rp.parts) > len(Path(best).resolve().parts):
            best = str(rp)
    return best


def workspace_name_for_video(video_path: str, raw_videos_root=None) -> str:
    """Build a workspace folder name that mirrors the video's subfolders.

    raw_videos_root may be one root or several (see video_roots). The video's path
    relative to the root that contains it becomes the name, keeping the separators, so
    the workspace tree has the same shape as the data tree (and the many clip_001.mp4 in
    different case folders cannot collide).
    """
    video = Path(video_path)

    rel_parts = None
    root = root_for_video(video_path, raw_videos_root) if raw_videos_root else None
    if root:
        try:
            rel_parts = video.resolve().relative_to(Path(root).resolve()).parts
        except (ValueError, OSError):
            rel_parts = None

    if rel_parts is None:
        parts = video.parts
        for marker in ('raw_videos', 'raw-videos', 'data'):
            if marker in parts:
                rel_parts = parts[parts.index(marker) + 1:]
                break

    if not rel_parts:
        rel_parts = (video.name,)

    return str(Path(*rel_parts))


def workspace_path_for_video(video_path: str, workspace_root: str,
                             raw_videos_root=None) -> str:
    return str(Path(workspace_root) / workspace_name_for_video(video_path, raw_videos_root))


def legacy_workspace_path_for_video(video_path: str, workspace_root: str,
                                    raw_videos_root=None) -> str:
    """Where an older version of the GUI would have put this video's workspace: flat
    (just the file name), or the later flattened-subfolder name. Whichever of the two
    exists is returned, so workspaces made before the nested layout still open."""
    root = Path(workspace_root)
    flattened = workspace_name_for_video(video_path, raw_videos_root).replace(os.sep, '__')
    candidates = [root / Path(video_path).name, root / flattened]
    for c in candidates:
        if c.is_dir():
            return str(c)
    return str(candidates[0])


if __name__ == '__main__':
    # python -m gui.gui_utils -- checks the data tree is mirrored into the workspace
    assert workspace_name_for_video('/data/caseA/day1/clip.mp4', ['/data']) == \
        os.path.join('caseA', 'day1', 'clip.mp4')
    assert workspace_name_for_video('/x/data/caseB/clip.mp4') == os.path.join('caseB', 'clip.mp4')
    assert workspace_name_for_video('/elsewhere/clip.mp4') == 'clip.mp4'
    print('ok')
