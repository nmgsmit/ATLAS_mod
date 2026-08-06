"""Loader dialog: pick a raw video or an existing workspace without leaving the GUI.

The GUI opens this modal dialog (the "Open..." button / the `O` shortcut). On accept it
exposes `selection`, a dict the controller feeds to MainController.load_workspace():
    {'video': <path>}      -- a raw source video (frames are extracted on first open)
    {'workspace': <path>}  -- an existing workspace folder (read back as-is)

A raw video is tagged "in workspace" when a workspace folder named after it already holds
extracted frames, so you can tell at a glance what still needs importing. Opening such a
video reuses that workspace (no re-decode, existing masks/arches are kept) -- the same thing
as picking it from the Workspaces tab.
"""
import os
from os import path
from pathlib import Path

import json

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
                               QTreeWidget, QTreeWidgetItem, QListWidget, QListWidgetItem,
                               QLineEdit, QLabel, QPushButton, QDialogButtonBox)
from PySide6.QtGui import QBrush, QColor
from PySide6.QtCore import Qt

from gui.gui_utils import (legacy_workspace_path_for_video, workspace_path_for_video)
from gui.scale_objects import CLASSES as SCALE_CLASSES

# containers OpenCV can decode; anything else in raw_videos is ignored by the browser
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.mpg', '.mpeg', '.wmv', '.m4v', '.webm'}

_IMPORTED_BRUSH = QBrush(QColor(80, 190, 120))   # green  -- already has extracted frames
_NEW_BRUSH = QBrush(QColor(150, 150, 150))       # grey   -- not imported yet

_ROLE_PATH = Qt.ItemDataRole.UserRole
_ROLE_KIND = Qt.ItemDataRole.UserRole + 1        # 'video' or 'workspace'


def _resolve_existing_path(path_value: str, fallback_names=()) -> str:
    """Resolve a path from the current working directory or the project root.

    This also accepts both raw_videos and raw-videos naming, which is the common cause
    of the Open dialog not showing files.
    """
    if not path_value:
        path_value = ''

    project_root = Path(__file__).resolve().parent.parent
    candidates = []
    if path_value:
        candidates.append(path_value)
    for name in fallback_names:
        if name:
            candidates.append(name)

    for candidate in candidates:
        p = Path(candidate)
        if not p.is_absolute():
            for base in (Path.cwd(), project_root):
                full = (base / p).resolve()
                if full.exists():
                    return str(full)
        else:
            if p.exists():
                return str(p.resolve())

    for alias in (*fallback_names, 'raw_videos', 'raw-videos'):
        if not alias:
            continue
        full = (project_root / alias).resolve()
        if full.exists():
            return str(full)

    if path_value:
        return str((Path.cwd() / Path(path_value)).resolve())
    return str(project_root)


def _has_frames(workspace_dir: str) -> int:
    """Number of extracted frames in a workspace, or 0 if it has none / does not exist."""
    images = path.join(workspace_dir, 'images')
    if not path.isdir(images):
        return 0
    return sum(1 for f in os.listdir(images)
               if f.lower().endswith(('.jpg', '.jpeg', '.png')))


def _has_files(folder: str, suffixes=()) -> int:
    if not path.isdir(folder):
        return 0
    return sum(1 for f in os.listdir(folder)
               if not suffixes or f.lower().endswith(suffixes))


def _has_json_entries(file_path: str) -> int:
    if not path.isfile(file_path):
        return 0
    try:
        with open(file_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
    except Exception:
        return 0
    if isinstance(data, dict):
        return sum(1 for v in data.values() if v)
    if isinstance(data, list):
        return len(data)
    return int(bool(data))


def _scale_reference_counts(file_path: str):
    counts = {cid: 0 for cid in SCALE_CLASSES}
    if not path.isfile(file_path):
        return counts
    try:
        with open(file_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
    except Exception:
        return counts
    for objs in data.get('frames', {}).values():
        seen = set()
        for obj in objs:
            try:
                cid = int(obj.get('class_id'))
            except Exception:
                continue
            if cid in counts:
                seen.add(cid)
        for cid in seen:
            counts[cid] += 1
    return counts


_SCALE_SHORT_NAMES = {1: 'Ruler', 2: 'Catheter', 3: 'Robot'}


def _scale_reference_overview(counts: dict) -> str:
    parts = []
    for cid in sorted(SCALE_CLASSES):
        short = _SCALE_SHORT_NAMES.get(cid, SCALE_CLASSES[cid]['name'])
        parts.append(f"{short}{'✓' if counts.get(cid, 0) else '×'}")
    return 'Scale[' + ' '.join(parts) + ']'


def _workspace_info_summary(workspace: str) -> str:
    masks_dir = path.join(workspace, 'masks')
    masks_present = bool(_has_files(masks_dir, ('.png', '.jpg', '.jpeg')))
    arch_present = bool(_has_json_entries(path.join(workspace, 'arches.json')))
    scale_counts = _scale_reference_counts(path.join(workspace, 'scale_objects.json'))
    scale_present = any(scale_counts.values())

    parts = [f"Mask{'✓' if masks_present else '×'}",
             f"Arch{'✓' if arch_present else '×'}",
             _scale_reference_overview(scale_counts) if scale_present else 'Scale×']
    return ' '.join(parts)


def _describe_status(count: int, present_label: str, missing_label: str) -> str:
    return f'{present_label} ({count})' if count else missing_label


class LoaderDialog(QDialog):

    def __init__(self, parent, raw_videos_root: str, workspace_root: str,
                 current_workspace: str = None):
        super().__init__(parent)
        self.raw_videos_root = _resolve_existing_path(raw_videos_root, ('raw_videos', 'raw-videos'))
        self.workspace_root = _resolve_existing_path(workspace_root, ('workspace',))
        self.current_workspace = (path.normpath(current_workspace)
                                  if current_workspace else None)
        self.selection = None    # set on accept

        self.setWindowTitle('Open video / workspace')
        self.resize(760, 520)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_videos_tab(), 'Raw videos')
        self.tabs.addTab(self._build_workspaces_tab(), 'Workspaces')

        self.hint = QLabel()
        self.hint.setWordWrap(True)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Open |
                                        QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Open).setText('Open')
        self.buttons.accepted.connect(self._on_open)
        self.buttons.rejected.connect(self.reject)

        refresh = QPushButton('Refresh')
        refresh.clicked.connect(self.reload)
        bottom = QHBoxLayout()
        bottom.addWidget(refresh)
        bottom.addStretch(1)
        bottom.addWidget(self.buttons)

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        layout.addWidget(self.hint)
        layout.addLayout(bottom)
        self.setLayout(layout)

        self.reload()
        self.tabs.currentChanged.connect(self._update_hint)
        self._update_hint()

    # ---- tab construction -------------------------------------------------

    def _build_videos_tab(self) -> QWidget:
        w = QWidget()
        self.filter_box = QLineEdit()
        self.filter_box.setPlaceholderText('Filter videos by name...')
        self.filter_box.textChanged.connect(self._apply_filter)

        self.video_tree = QTreeWidget()
        self.video_tree.setHeaderLabels(['Video', 'Status', 'Info'])
        self.video_tree.setColumnWidth(0, 430)
        self.video_tree.setColumnWidth(1, 170)
        self.video_tree.setColumnWidth(2, 180)
        self.video_tree.itemSelectionChanged.connect(self._update_hint)
        self.video_tree.itemDoubleClicked.connect(lambda *_: self._on_open())

        lay = QVBoxLayout()
        lay.addWidget(self.filter_box)
        lay.addWidget(self.video_tree)
        w.setLayout(lay)
        return w

    def _build_workspaces_tab(self) -> QWidget:
        w = QWidget()
        self.workspace_list = QListWidget()
        self.workspace_list.itemSelectionChanged.connect(self._update_hint)
        self.workspace_list.itemDoubleClicked.connect(lambda *_: self._on_open())

        lay = QVBoxLayout()
        lay.addWidget(QLabel('Existing workspaces (folders with extracted frames):'))
        lay.addWidget(self.workspace_list)
        w.setLayout(lay)
        return w

    # ---- populate ---------------------------------------------------------

    def reload(self):
        self._populate_videos()
        self._populate_workspaces()
        self._apply_filter(self.filter_box.text())
        self._update_hint()

    def _populate_videos(self):
        self.video_tree.clear()
        if not path.isdir(self.raw_videos_root):
            item = QTreeWidgetItem([f'(folder not found: {self.raw_videos_root})', ''])
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.video_tree.addTopLevelItem(item)
            return

        # group videos by their sub-folder under raw_videos (e.g. Nick / Vivian)
        groups = {}
        for root, _dirs, files in os.walk(self.raw_videos_root):
            for f in sorted(files):
                if path.splitext(f)[1].lower() not in VIDEO_EXTS:
                    continue
                full = path.join(root, f)
                rel = path.relpath(root, self.raw_videos_root)
                group = '.' if rel == os.curdir else rel
                groups.setdefault(group, []).append((f, full))

        for group in sorted(groups):
            label = 'raw_videos' if group == '.' else group
            parent = QTreeWidgetItem([label, ''])
            parent.setFirstColumnSpanned(True)
            self.video_tree.addTopLevelItem(parent)
            for name, full in groups[group]:
                ws = workspace_path_for_video(full, self.workspace_root, self.raw_videos_root)
                n_frames = _has_frames(ws)
                legacy_ws = legacy_workspace_path_for_video(full, self.workspace_root)
                active_ws = ws if n_frames else legacy_ws
                if not n_frames:
                    n_frames = _has_frames(legacy_ws)
                if n_frames:
                    status, brush = f'in workspace ({n_frames} frames)', _IMPORTED_BRUSH
                else:
                    status, brush = 'not imported', _NEW_BRUSH
                info = _workspace_info_summary(active_ws) if n_frames else 'Mask× Arch× Scale×'
                child = QTreeWidgetItem([name, status, info])
                child.setForeground(1, brush)
                child.setData(0, _ROLE_PATH, full)
                child.setData(0, _ROLE_KIND, 'video')
                parent.addChild(child)
            parent.setExpanded(True)

    def _populate_workspaces(self):
        self.workspace_list.clear()
        if not path.isdir(self.workspace_root):
            return
        for name in sorted(os.listdir(self.workspace_root)):
            ws = path.join(self.workspace_root, name)
            if not path.isdir(ws):
                continue
            n_frames = _has_frames(ws)
            if not n_frames:
                continue
            label = f'{name}   ({n_frames} frames)'
            if self.current_workspace and path.normpath(ws) == self.current_workspace:
                label += '   [current]'
            item = QListWidgetItem(label)
            item.setData(_ROLE_PATH, ws)
            item.setData(_ROLE_KIND, 'workspace')
            self.workspace_list.addItem(item)

    # ---- filtering / hints ------------------------------------------------

    def _apply_filter(self, text: str):
        text = (text or '').strip().lower()
        for i in range(self.video_tree.topLevelItemCount()):
            parent = self.video_tree.topLevelItem(i)
            any_visible = False
            for j in range(parent.childCount()):
                child = parent.child(j)
                match = text in child.text(0).lower()
                child.setHidden(not match)
                any_visible = any_visible or match
            parent.setHidden(not any_visible)

    def _selected(self):
        """Return (path, kind) for the current tab's selection, or (None, None)."""
        if self.tabs.currentIndex() == 0:
            items = self.video_tree.selectedItems()
            if items and items[0].data(0, _ROLE_KIND):
                return items[0].data(0, _ROLE_PATH), items[0].data(0, _ROLE_KIND)
        else:
            item = self.workspace_list.currentItem()
            if item is not None:
                return item.data(_ROLE_PATH), item.data(_ROLE_KIND)
        return None, None

    def _selected_any(self):
        """Return whichever item is selected in either tab, if any."""
        items = self.video_tree.selectedItems()
        if items and items[0].data(0, _ROLE_KIND):
            return items[0].data(0, _ROLE_PATH), items[0].data(0, _ROLE_KIND)
        item = self.workspace_list.currentItem()
        if item is not None:
            return item.data(_ROLE_PATH), item.data(_ROLE_KIND)
        return None, None

    def _update_hint(self, *_):
        sel_path, kind = self._selected()
        if kind is None:
            self.hint.setText('Select a raw video or an existing workspace, then Open.')
        elif kind == 'video':
            ws = workspace_path_for_video(sel_path, self.workspace_root, self.raw_videos_root)
            legacy_ws = legacy_workspace_path_for_video(sel_path, self.workspace_root)
            if _has_frames(ws) or _has_frames(legacy_ws):
                self.hint.setText('This video already has a workspace. Opening it continues '
                                  'that annotation (existing frames, masks and arches are kept).')
            else:
                self.hint.setText('New video -- frames will be extracted into a new workspace '
                                  'on open. This can take a moment for long clips.')
        else:
            self.hint.setText('Opens this workspace as-is.')

    # ---- accept -----------------------------------------------------------

    def _on_open(self):
        sel_path, kind = self._selected()
        if kind is None:
            return
        if kind == 'video':
            ws = workspace_path_for_video(sel_path, self.workspace_root, self.raw_videos_root)
            legacy_ws = legacy_workspace_path_for_video(sel_path, self.workspace_root)
            if _has_frames(ws):
                self.selection = {'workspace': ws}
                self.accept()
                return
            if _has_frames(legacy_ws):
                self.selection = {'workspace': legacy_ws}
                self.accept()
                return
        self.selection = {kind: sel_path}
        self.accept()
