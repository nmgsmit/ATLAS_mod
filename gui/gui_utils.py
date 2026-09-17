import os
import re
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

# containers OpenCV can decode; anything else in a video folder is ignored
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.mpg', '.mpeg', '.wmv', '.m4v', '.webm'}


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


def video_for_workspace(workspace: str, workspace_root: str, roots) -> str:
    """The source video a workspace was made from, or None if it cannot be found.

    The workspace tree mirrors the video tree (workspace_name_for_video), so the
    workspace path relative to workspace_root is the video path relative to its root --
    that mapping just has to be walked backwards. Older workspaces were named flat (the
    bare file name) or with '__' for the separators, so those two spellings are tried as
    well before giving up.
    """
    if not workspace:
        return None
    try:
        rel = Path(workspace).resolve().relative_to(Path(workspace_root).resolve())
    except (ValueError, OSError):
        rel = Path(Path(workspace).name)

    candidates = [rel]
    if '__' in rel.name:                       # legacy flattened-subfolder name
        candidates.append(Path(*rel.name.split('__')))

    for root in (roots or []):
        for cand in candidates:
            full = Path(root) / cand
            if full.is_file():
                return str(full)

    # legacy flat workspace: only the file name survived, so look it up in the tree
    for root in (roots or []):
        for found in Path(root).rglob(rel.name):
            if found.is_file():
                return str(found)
    return None


def _natural_key(name: str):
    """Sort key splitting digit runs out as numbers, so clip2 sorts before clip10."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', name)]


def sibling_videos(video_path: str) -> list:
    """Every video sitting in the same folder as `video_path`, in natural name order.

    Natural (not plain lexicographic) so numbered clips run 2, 3, ... 10 rather than
    10, 2, 3 -- the order the file browser shows and the order they were recorded in.
    Case is ignored so the run does not split on a stray capital.
    """
    folder = Path(video_path).parent
    if not folder.is_dir():
        return []
    vids = [str(folder / f.name) for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS]
    return sorted(vids, key=lambda p: _natural_key(Path(p).name))


def neighbour_video(video_path: str, step: int = 1) -> str:
    """The next (step=1) or previous (step=-1) video in the same folder.

    None at either end of the folder, so the caller can say so rather than silently
    wrapping around to a clip that was already annotated.
    """
    vids = sibling_videos(video_path)
    try:
        i = vids.index(str(Path(video_path)))
    except ValueError:
        try:
            resolved = str(Path(video_path).resolve())
            i = [str(Path(v).resolve()) for v in vids].index(resolved)
        except (ValueError, OSError):
            return None
    j = i + step
    if 0 <= j < len(vids):
        return vids[j]
    return None
