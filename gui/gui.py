import functools
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from PySide6.QtWidgets import (QWidget, QComboBox, QCheckBox, QHBoxLayout, QLabel, QPushButton,
                               QTextEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QVBoxLayout,
                               QSizePolicy, QButtonGroup, QSlider, QRadioButton, QApplication,
                               QFileDialog, QMessageBox, QGridLayout, QGroupBox, QStackedWidget)

from PySide6.QtGui import (QKeySequence, QShortcut, QTextCursor, QImage, QPixmap, QIcon)
from PySide6.QtCore import Qt, QTimer

from gui import retzius_arch
from gui import scale_objects
from gui.cutie.utils.palette import custom_palette_np, custom_names
from gui.gui_utils import *
from gui.loader_dialog import LoaderDialog
from gui.ritm import controller


def _class_button(cls_id, name, rgb):
    """A checkable class button in the class's own colour, labelled '<id>  <name>'.
    Shared by the segmentation classes and the scale references so the two pickers are
    literally the same control."""
    r, g, b = (int(c) for c in rgb)
    fg = 'black' if (r * 299 + g * 587 + b * 114) / 1000 > 140 else 'white'
    btn = QPushButton(f'{cls_id}  {name}')
    btn.setCheckable(True)
    btn.setStyleSheet(
        f'QPushButton {{ background-color: rgb({r},{g},{b}); color: {fg};'
        f' border: 2px solid #808080; border-radius: 4px; padding: 4px 8px; }}'
        f'QPushButton:checked {{ border: 3px solid #ffffff; font-weight: bold; }}')
    return btn


class GUI(QWidget):

    def __init__(self, controller, cfg: DictConfig) -> None:
        super().__init__()

        # callbacks to be set by the controller
        self.on_mouse_motion_xy = None
        self.click_fn = None
        self.release_fn = None

        self.controller = controller
        self.cfg = cfg
        self.h = controller.h
        self.w = controller.w
        self.T = controller.T

        # zoom state: view a zoom-magnified crop centered on zoom_center (image coords)
        self.zoom = 1.0
        self.zoom_center = None
        self.crop_origin = (0, 0)
        self._pan_last = None  # ponytail: Ctrl+drag pan, no extra pan-mode toggle

        # set up the window
        self.setWindowTitle(f'SurgeNetSeg demo: {cfg["workspace"]}')
        self.setGeometry(100, 100, self.w + 200, self.h + 200)
        self.setWindowIcon(QIcon('docs/icon.png'))

        # set up some buttons
        # Open another video / workspace without leaving the GUI (loader dialog).
        self.open_button = QPushButton('Open... (O)')
        self.open_button.setToolTip('Load another raw video or an existing workspace')
        self.open_button.clicked.connect(self.open_loader)
        self.play_button = QPushButton('Play video')
        self.play_button.clicked.connect(self.on_play_video)
        # Playback speed: preview only. Multiplies the play frame-rate; it does NOT change
        # which/how many frames are annotated -- every extracted frame is always annotated.
        self.play_speed_combo = QComboBox()
        for label, mult in (('0.5×', 0.5), ('1×', 1.0), ('2×', 2.0), ('4×', 4.0), ('8×', 8.0)):
            self.play_speed_combo.addItem(label, mult)
        self.play_speed_combo.setCurrentText('1×')
        self.play_speed_combo.setToolTip('Speed for Play / Propagate / Track. Higher = faster '
                                         'with less frequent preview redraw. Never changes '
                                         'which frames are annotated.')
        self.play_speed_combo.currentIndexChanged.connect(self.on_play_speed_changed)
        # step one frame; lambdas, not the bare slot: clicked emits a `checked` bool
        # that would land on the `step` argument
        self.prev_frame_button = QPushButton('◀ Frame')
        self.prev_frame_button.clicked.connect(lambda: controller.on_prev_frame())
        self.next_frame_button = QPushButton('Frame ▶')
        self.next_frame_button.clicked.connect(lambda: controller.on_next_frame())
        self.commit_button = QPushButton('Commit (C)')
        self.commit_button.clicked.connect(controller.on_commit)

        self.forward_run_button = QPushButton('Propagate forward')
        self.forward_run_button.clicked.connect(controller.on_forward_propagation)

        self.backward_run_button = QPushButton('Propagate backward')
        self.backward_run_button.clicked.connect(controller.on_backward_propagation)

        # ponytail: SEGMENT / DEPTH / ANCHOR / ROBOT ANCHOR / SUL / LIVE SUL buttons removed
        # from the UI to keep it simple. controller.on_segment/on_depth/on_anchor/
        # on_robot_anchor/on_sul/on_live_sul still exist -- re-add a QPushButton here and a
        # line in overlay_topbox below to bring one back.
        # Mode: mask annotation (paint classes), arch annotation (place/drag the arc) or
        # scale annotation (draw the known-length references). Order must match
        # MainController.MODE_INDEX, which owns the actual switching.
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Mask annotation', 'Arch annotation', 'Scale annotation'])
        self.mode_combo.currentIndexChanged.connect(self.on_mode_combo_changed)
        # arch tracking: its own pass, in either direction (does not touch the masks).
        # lambdas, not partial: clicked emits a `checked` bool that would land on `direction`
        self.track_back_button = QPushButton('TRACK ◀')
        self.track_back_button.setCheckable(True)  # stays down while that run is going
        self.track_back_button.clicked.connect(lambda: controller.on_track_arch('backward'))
        self.track_fwd_button = QPushButton('TRACK ▶')
        self.track_fwd_button.setCheckable(True)   # stays down while that run is going
        self.track_fwd_button.clicked.connect(lambda: controller.on_track_arch('forward'))
        self.reset_arch_button = QPushButton('RESET ARCH')
        self.reset_arch_button.clicked.connect(controller.on_reset_arch)

        # Arch tip height: the tip is often off-image, where there is no handle to grab,
        # so drive the height scalar directly (drag or scroll the slider).
        self.arch_height_slider = QSlider(Qt.Orientation.Horizontal)
        self.arch_height_slider.setRange(0, 500)           # px above the base chord; no bulge down
        self.arch_height_slider.setMinimumWidth(150)       # range/width set the px-per-drag feel
        self.arch_height_slider.setPageStep(10)
        self.arch_height_slider.setEnabled(False)
        self.arch_height_slider.valueChanged.connect(controller.on_arch_height_slider)
        self.arch_height_slider.sliderReleased.connect(controller.on_arch_height_released)
        self.arch_height_label = QLabel('Arch tip: --')
        self.arch_height_label.setMinimumWidth(85)

        # Arch sharpness: the shape exponent p in v(u) = height*(1-|u|^p), in tenths.
        # Low = pointed tip, 2.0 = plain parabola, high = flat crown with steep arms.
        self.arch_power_slider = QSlider(Qt.Orientation.Horizontal)
        self.arch_power_slider.setRange(int(retzius_arch.POWER_RANGE[0] * 10),
                                        int(retzius_arch.POWER_RANGE[1] * 10))
        self.arch_power_slider.setValue(int(retzius_arch.DEFAULT_POWER * 10))
        self.arch_power_slider.setMinimumWidth(110)
        self.arch_power_slider.setEnabled(False)
        self.arch_power_slider.valueChanged.connect(controller.on_arch_power_slider)
        self.arch_power_slider.sliderReleased.connect(controller.on_arch_height_released)
        self.arch_power_label = QLabel('Sharpness: --')
        self.arch_power_label.setMinimumWidth(85)
        # ponytail: kept (unshown) -- main_controller still writes to it
        self.measure_label = QLabel('SUL: -- mm  |  SUL depth: -- mm')

        # --- scale annotation: the known-length references (gui/scale_objects.py) ---
        # Its own TRACK pair and RESET, mirroring the arch's: same job, different objects.
        self.scale_track_back_button = QPushButton('TRACK ◀')
        self.scale_track_back_button.setCheckable(True)
        self.scale_track_back_button.clicked.connect(
            lambda: controller.on_track_scale('backward'))
        self.scale_track_fwd_button = QPushButton('TRACK ▶')
        self.scale_track_fwd_button.setCheckable(True)
        self.scale_track_fwd_button.clicked.connect(
            lambda: controller.on_track_scale('forward'))
        self.reset_scale_button = QPushButton('RESET REFERENCE')
        self.reset_scale_button.clicked.connect(controller.on_reset_scale)

        # Physical length of the active reference. Only the ruler's is editable (you
        # choose how much of the ruler to span); the catheter tip and robot arm are
        # fixed properties of the instrument, so the box goes read-only for those.
        self.scale_mm_box = QDoubleSpinBox()
        self.scale_mm_box.setRange(0.1, 500.0)
        self.scale_mm_box.setDecimals(3)
        self.scale_mm_box.setSingleStep(0.5)
        self.scale_mm_box.setSuffix(' mm')
        self.scale_mm_box.setMinimumWidth(100)
        self.scale_mm_box.setToolTip('Real length this reference spans. Editable for the '
                                     'ruler only; applies to every frame it is drawn on.')
        self.scale_mm_box.setValue(scale_objects.default_mm(1))   # before connecting: the
        self.scale_mm_box.valueChanged.connect(controller.on_scale_mm)  # box starts at 0

        # The 5 tracking points are normally derived from the two endpoints and are not
        # grabbable -- this arms them, so one sitting on a bad patch can be slid along
        # the line. Off by default so a stray click can never nudge a probe.
        self.scale_points_check = QCheckBox('Edit tracking points')
        self.scale_points_check.setToolTip('Let the interior tracking points be dragged. '
                                           'They slide along the line, so the reference '
                                           'stays straight. Off by default.')
        self.scale_points_check.toggled.connect(controller.on_scale_edit_points)

        # Robot arm only: is THIS frame's measured diameter one you believe? Nothing but
        # the frames left ticked is exported, so this is the annotation, not a display
        # option. Enabled only while the robot arm is the active reference.
        self.arm_trust_check = QCheckBox('Export this frame  (D)')
        self.arm_trust_check.setToolTip('Robot arm: export this frame\'s diameter as a '
                                        'scale reference. Set automatically by the clip '
                                        'reconciliation -- frames that disagree with '
                                        'their neighbours are dropped -- and your answer '
                                        'overrides that for good. Frames left off keep '
                                        'their four points, they just are not exported.')
        self.arm_trust_check.setEnabled(False)
        self.arm_trust_check.toggled.connect(controller.on_arm_trust)

        # Robot arm only: re-guess every frame off its own mask and reconcile them against
        # each other, which is the step that makes the millimetres consistent -- a single
        # frame cannot see its own error, because a mask a pixel fat moves both sides of
        # the arm together. See robot_arm.clip_scale. (TRACK is the other way in: it
        # carries the four points from THIS frame instead of re-guessing each one.)
        self.arm_measure_btn = QPushButton('Measure clip')
        self.arm_measure_btn.setToolTip(
            'Robot arm: guess the four points off every frame\'s mask, then reconcile '
            'the diameters. Each frame is exported as the local median of its neighbours '
            'rather than its own reading, and frames that disagree are dropped '
            'automatically. Click the arm once first to say which instrument it is; '
            'frames whose points you placed by hand are left alone.')
        self.arm_measure_btn.setEnabled(False)
        self.arm_measure_btn.clicked.connect(controller.on_arm_measure_clip)

        # The one knob per-frame geometry cannot replace: a mask that is systematically fat
        # or thin reads wide or narrow on EVERY frame, so no amount of smoothing or edge
        # fitting recovers it. Cross-check it against the hand-drawn ruler (which carries
        # none of the mask's bias) and type the correction here.
        self.arm_calib_box = QDoubleSpinBox()
        self.arm_calib_box.setRange(0.50, 2.00)
        self.arm_calib_box.setSingleStep(0.01)
        self.arm_calib_box.setDecimals(3)
        self.arm_calib_box.setValue(1.0)
        self.arm_calib_box.setPrefix('x ')
        self.arm_calib_box.setToolTip(
            'Robot arm: one multiplier for the whole clip, for a segmentation that is '
            'systematically fat or thin. 1.0 leaves the measurement alone. The readout '
            'says how the arm compares with the ruler on frames that have both -- but '
            'only trust that comparison where the two are at a similar depth, since '
            'mm/px genuinely differs between a near object and a far one.')
        self.arm_calib_box.setEnabled(False)
        self.arm_calib_box.valueChanged.connect(controller.on_arm_calib)

        self.scale_info_label = QLabel('Reference: --')
        self.scale_info_label.setMinimumWidth(260)

        # universal progressbar
        self.progressbar = QProgressBar()
        self.progressbar.setMinimum(0)
        self.progressbar.setMaximum(100)
        self.progressbar.setValue(0)
        self.progressbar.setMinimumWidth(140)

        self.reset_frame_button = QPushButton('Reset frame')
        self.reset_frame_button.clicked.connect(controller.on_reset_mask)
        self.reset_object_button = QPushButton('Reset object')
        self.reset_object_button.clicked.connect(controller.on_reset_object)
        # Eraser: draw a polygon around a stray mask to send it back to background.
        # Sits with the class buttons -- it is the "paint background" of that row.
        self.polygon_cut_button = QPushButton('ERASER  (polygon)')
        self.polygon_cut_button.setCheckable(True)
        self.polygon_cut_button.setStyleSheet(
            'QPushButton { border: 2px solid #808080; border-radius: 4px; padding: 4px 8px; }'
            'QPushButton:checked { border: 3px solid #ffffff; font-weight: bold; }')
        self.polygon_cut_button.setToolTip(
            'Polygon mode returns its interior to background instead of painting.\n'
            'Edits this frame only -- the memory bank is untouched. (E)')
        self.polygon_cut_button.clicked.connect(controller.on_toggle_polygon_erase)

        # set up the LCD
        self.lcd = QTextEdit()
        self.lcd.setReadOnly(True)
        self.lcd.setMaximumHeight(28)
        self.lcd.setMaximumWidth(150)
        self.lcd.setText('{: 5d} / {: 5d}'.format(0, controller.T - 1))

        # ID
        self.object_dial = QSpinBox()

        # One always-visible button per class, in its own colour -- same action as the
        # number keys, which stay the fast path. Replaces the old dropdown (two clicks).
        self.class_group = QButtonGroup(self)
        self.class_buttons = {}
        for obj_id in range(1, controller.num_objects + 1):
            btn = _class_button(obj_id, custom_names[obj_id], custom_palette_np[obj_id])
            self.class_group.addButton(btn, obj_id)
            self.class_buttons[obj_id] = btn
        self.class_group.idClicked.connect(controller.hit_number_key)
        self.class_buttons[1].setChecked(True)   # controller.curr_object starts at 1, but is
                                                 # not set yet: the GUI is built before it

        # The scale references get the very same picker -- coloured buttons + number keys --
        # so switching reference feels identical to switching a segmentation class.
        self.scale_class_group = QButtonGroup(self)
        self.scale_class_buttons = {}
        for cid, spec in scale_objects.CLASSES.items():
            btn = _class_button(cid, spec['name'], spec['color'])
            self.scale_class_group.addButton(btn, cid)
            self.scale_class_buttons[cid] = btn
        self.scale_class_group.idClicked.connect(controller.on_scale_class)
        self.scale_class_buttons[1].setChecked(True)

        # ponytail: dial is off the layout (the buttons replace it) but stays alive --
        # main_controller still calls object_dial.setValue()
        self.object_dial.setReadOnly(False)
        self.object_dial.setMinimumSize(50, 30)
        self.object_dial.setMinimum(1)
        self.object_dial.setMaximum(controller.num_objects)
        self.object_dial.editingFinished.connect(controller.on_object_dial_change)

        self.object_color = QLabel()
        self.object_color.setMinimumSize(30, 30)
        self.object_color.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # big, always-visible readout of the class currently being painted -- so it is
        # obvious at a glance which number key / colour is active
        self.current_class_label = QLabel(f'1  {custom_names.get(1, "1")}')
        f = self.current_class_label.font()
        f.setPointSize(f.pointSize() + 2)
        f.setBold(True)
        self.current_class_label.setFont(f)

        # same readout for the active scale reference
        self.current_scale_label = QLabel(f'1  {scale_objects.class_name(1)}')
        self.current_scale_label.setFont(f)

        self.frame_name = QLabel()
        self.frame_name.setMinimumSize(100, 30)
        self.frame_name.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # timeline slider
        self.tl_slider = QSlider(Qt.Orientation.Horizontal)
        self.tl_slider.valueChanged.connect(controller.on_slider_update)
        self.tl_slider.setMinimum(0)
        self.tl_slider.setMaximum(controller.T - 1)
        self.tl_slider.setValue(0)
        self.tl_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.tl_slider.setTickInterval(1)

        # combobox
        self.combo = QComboBox(self)
        
        self.combo.addItem("mask")
        self.combo.addItem("mask overlay")
        self.combo.addItem("image")
        self.combo.setCurrentText('mask overlay')
        self.combo.currentTextChanged.connect(controller.set_vis_mode)

        # Main canvas -> QLabel
        self.main_canvas = QLabel()
        self.main_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_canvas.setMinimumSize(100, 100)

        self.main_canvas.mousePressEvent = self.on_mouse_press
        self.main_canvas.mouseMoveEvent = self.on_mouse_motion
        self.main_canvas.setMouseTracking(True)  # Required for all-time tracking
        self.main_canvas.mouseReleaseEvent = self.on_mouse_release
        self.main_canvas.wheelEvent = self.on_wheel

        # clearing memory
        self.clear_all_mem_button = QPushButton('Reset all memory')
        self.clear_all_mem_button.clicked.connect(controller.on_clear_memory)
        self.clear_non_perm_mem_button = QPushButton('Reset non-permanent memory')
        self.clear_non_perm_mem_button.clicked.connect(controller.on_clear_non_permanent_memory)

        # displaying memory usage
        self.perm_mem_gauge, self.perm_mem_gauge_layout = create_gauge('Permanent memory size')
        self.work_mem_gauge, self.work_mem_gauge_layout = create_gauge('Working memory size')
        self.long_mem_gauge, self.long_mem_gauge_layout = create_gauge('Long-term memory size')
        self.gpu_mem_gauge, self.gpu_mem_gauge_layout = create_gauge(
            'GPU mem. (all proc, w/ caching)')
        self.torch_mem_gauge, self.torch_mem_gauge_layout = create_gauge(
            'GPU mem. (torch, w/o caching)')

        # Parameters setting
        self.work_mem_min, self.work_mem_min_layout = create_parameter_box(
            1, 100, 'Min. working memory frames', callback=controller.on_work_min_change)
        self.work_mem_max, self.work_mem_max_layout = create_parameter_box(
            2, 100, 'Max. working memory frames', callback=controller.on_work_max_change)
        self.long_mem_max, self.long_mem_max_layout = create_parameter_box(
            1000,
            100000,
            'Max. long-term memory size',
            step=1000,
            callback=controller.update_config)
        self.mem_every_box, self.mem_every_box_layout = create_parameter_box(
            1, 100, 'Memory frame every (r)', callback=controller.update_config)

        # import mask/layer
        self.import_mask_button = QPushButton('Import mask')
        self.import_mask_button.clicked.connect(controller.on_import_mask)
        self.import_layer_button = QPushButton('Import layer')
        self.import_layer_button.clicked.connect(controller.on_import_layer)

        # Console on the GUI
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(100)
        self.console.setMaximumHeight(100)

        # Tips for the users
        self.tips = QTextEdit()
        self.tips.setReadOnly(True)
        self.tips.setTextInteractionFlags(Qt.NoTextInteraction)
        self.tips.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.tips.setMinimumWidth(240)   # so the right-hand panel can never be squeezed away
        with open(Path(__file__).parent / 'TIPS.md', 'r') as f:
            self.tips.setMarkdown(f.read())

        # navigator: one grouped toolbar under the canvas, in two rows --
        #   top    = always-visible controls (open clip, frame nav, mode, view, progress)
        #   bottom = the tools for the current Mode; set_mode_index() shows exactly one of
        #            the mask / arch / scale group sets, so they never crowd the window.
        # Each cluster is a titled QGroupBox so it is obvious what every control is for.
        navi = QVBoxLayout()

        fixed = lambda w: w.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        def make_group(title, widgets):
            box = QGroupBox(title)
            row = QHBoxLayout()
            row.setAlignment(Qt.AlignmentFlag.AlignLeft)
            for w in widgets:
                fixed(w)
                row.addWidget(w)
            box.setLayout(row)
            return box

        # --- always-visible groups (top row) ---
        clip_box = make_group('Clip', [self.open_button, self.lcd, self.frame_name])
        frame_box = make_group('Frame', [self.prev_frame_button, self.next_frame_button,
                                         self.play_button, QLabel('Speed:'),
                                         self.play_speed_combo])
        mode_box = make_group('Mode', [self.mode_combo])
        view_box = make_group('View', [self.combo])
        progress_box = QGroupBox('Progress')
        progress_row = QHBoxLayout()
        progress_row.addWidget(self.progressbar)
        progress_box.setLayout(progress_row)

        # --- Class group (mask mode): live readout + colour buttons + switch keys ---
        # 3 per row: one long row of class names forces the whole window wider than the
        # screen, which pushes the right-hand panel out of view
        self.class_grid = QGridLayout()
        self.class_grid.setSpacing(4)
        for i, btn in enumerate(self.class_buttons.values()):
            self.class_grid.addWidget(btn, i // 3, i % 3)
        class_header = QHBoxLayout()
        class_header.setAlignment(Qt.AlignmentFlag.AlignLeft)
        class_header.addWidget(QLabel('Painting:'))
        class_header.addWidget(self.object_color)
        class_header.addWidget(self.current_class_label)
        class_hint = QLabel('Switch: number keys 1–%d  ·  [ prev  /  ] next' %
                            controller.num_objects)
        class_hint.setStyleSheet('color: gray;')
        class_box_layout = QVBoxLayout()
        class_box_layout.addLayout(class_header)
        class_box_layout.addLayout(self.class_grid)
        class_box_layout.addWidget(class_hint)
        self.class_box = QGroupBox('Class')
        self.class_box.setLayout(class_box_layout)

        # --- other mask-mode groups ---
        self.edit_box = make_group('Edit', [self.reset_frame_button, self.reset_object_button,
                                            self.polygon_cut_button])
        # backward then forward, so the ▶ run button sits at the group's right edge
        self.segment_box = make_group('Segment', [self.commit_button, self.backward_run_button,
                                                  self.forward_run_button])

        # --- arch-mode groups ---
        # TRACK is to the arch what Propagate is to segmentation, so the tracking group is
        # laid out to mirror Segment (reset on the left, ◀ then ▶ on the right) and is placed
        # in the same slot of the mode-swapped row below -- TRACK ▶ lands where Propagate ▶ is.
        self.arch_track_box = make_group('Arch tracking', [self.reset_arch_button,
                                                           self.track_back_button,
                                                           self.track_fwd_button])
        arch_shape_layout = QHBoxLayout()
        arch_shape_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        for w in (self.arch_height_label, self.arch_height_slider,
                  self.arch_power_label, self.arch_power_slider):
            arch_shape_layout.addWidget(w)
        self.arch_shape_box = QGroupBox('Arch shape')
        self.arch_shape_box.setLayout(arch_shape_layout)

        # --- scale-mode groups (laid out to mirror the mask page: picker on the left,
        # the "run" group pushed to the right, so TRACK lands where Propagate ▶ is) ---
        scale_grid = QGridLayout()
        scale_grid.setSpacing(4)
        for i, btn in enumerate(self.scale_class_buttons.values()):
            scale_grid.addWidget(btn, i // 3, i % 3)
        scale_header = QHBoxLayout()
        scale_header.setAlignment(Qt.AlignmentFlag.AlignLeft)
        scale_header.addWidget(QLabel('Drawing:'))
        scale_header.addWidget(self.current_scale_label)
        scale_hint = QLabel('Switch: number keys 1–%d  ·  [ prev  /  ] next' %
                            len(scale_objects.CLASSES))
        scale_hint.setStyleSheet('color: gray;')
        scale_box_layout = QVBoxLayout()
        scale_box_layout.addLayout(scale_header)
        scale_box_layout.addLayout(scale_grid)
        scale_box_layout.addWidget(scale_hint)
        self.scale_class_box = QGroupBox('Reference')
        self.scale_class_box.setLayout(scale_box_layout)

        scale_shape_layout = QVBoxLayout()
        mm_row = QHBoxLayout()
        mm_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
        mm_row.addWidget(QLabel('Length:'))
        mm_row.addWidget(self.scale_mm_box)
        mm_row.addWidget(self.scale_points_check)
        mm_row.addWidget(self.arm_measure_btn)
        mm_row.addWidget(self.arm_trust_check)
        mm_row.addWidget(QLabel('Calib:'))
        mm_row.addWidget(self.arm_calib_box)
        scale_shape_layout.addLayout(mm_row)
        scale_shape_layout.addWidget(self.scale_info_label)
        self.scale_shape_box = QGroupBox('Reference size')
        self.scale_shape_box.setLayout(scale_shape_layout)

        self.scale_track_box = make_group('Reference tracking',
                                          [self.reset_scale_button,
                                           self.scale_track_back_button,
                                           self.scale_track_fwd_button])

        # --- always-visible top row ---
        top_row = QHBoxLayout()
        top_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
        for box in (clip_box, frame_box, mode_box, view_box):
            top_row.addWidget(box)
        top_row.addStretch(1)
        top_row.addWidget(progress_box)

        # --- mode-swapped bottom row (one stack, two pages that share the same rectangle) ---
        # The "run" group (Segment / Arch tracking) is pushed to the right on both pages, so
        # switching Mode swaps Propagate <-> TRACK in place instead of shifting the toolbar.
        def make_page(left_boxes, run_box):
            page = QWidget()
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setAlignment(Qt.AlignmentFlag.AlignLeft)
            for b in left_boxes:
                row.addWidget(b)
            row.addStretch(1)
            row.addWidget(run_box)
            page.setLayout(row)
            return page

        self.mode_stack = QStackedWidget()
        self.mode_stack.addWidget(make_page([self.class_box, self.edit_box], self.segment_box))
        self.mode_stack.addWidget(make_page([self.arch_shape_box], self.arch_track_box))
        self.mode_stack.addWidget(make_page([self.scale_class_box, self.scale_shape_box],
                                            self.scale_track_box))

        navi.addLayout(top_row)
        navi.addWidget(self.mode_stack)

        # Drawing area main canvas
        draw_area = QHBoxLayout()
        draw_area.addWidget(self.main_canvas, 4)

        # right area
        right_area = QVBoxLayout()
        right_area.setAlignment(Qt.AlignmentFlag.AlignBottom)
        right_area.addWidget(self.tips)
        # right_area.addStretch(1)

        # Parameters
        right_area.addLayout(self.perm_mem_gauge_layout)
        right_area.addLayout(self.work_mem_gauge_layout)
        right_area.addLayout(self.long_mem_gauge_layout)
        right_area.addLayout(self.gpu_mem_gauge_layout)
        right_area.addLayout(self.torch_mem_gauge_layout)
        right_area.addWidget(self.clear_all_mem_button)
        right_area.addWidget(self.clear_non_perm_mem_button)
        right_area.addLayout(self.work_mem_min_layout)
        right_area.addLayout(self.work_mem_max_layout)
        right_area.addLayout(self.long_mem_max_layout)
        right_area.addLayout(self.mem_every_box_layout)

        # import mask/layer
        import_area = QHBoxLayout()
        import_area.setAlignment(Qt.AlignmentFlag.AlignBottom)
        import_area.addWidget(self.import_mask_button)
        import_area.addWidget(self.import_layer_button)
        right_area.addLayout(import_area)

        # console
        right_area.addWidget(self.console)

        draw_area.addLayout(right_area, 1)

        layout = QVBoxLayout()
        layout.addLayout(draw_area)
        layout.addWidget(self.tl_slider)
        layout.addLayout(navi)
        self.setLayout(layout)

        # The theme's padding is not counted in a button's size hint, so long labels get
        # elided. Widen every button to its own text once the whole tree exists.
        for btn in self.findChildren(QPushButton):
            fit_button_text(btn)

        # The Mode dropdown swaps the bottom-row stack (0 = mask, 1 = arch, 2 = scale); the
        # top row (clip, frame nav, mode, view, progress) and the timeline are always shown.
        self.set_mode_index(0)

        # timer to play video
        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(controller.on_play_video_timer)

        # timer to update GPU usage
        self.gpu_timer = QTimer()
        self.gpu_timer.setSingleShot(False)
        self.gpu_timer.timeout.connect(controller.on_gpu_timer)
        self.gpu_timer.setInterval(2000)
        self.gpu_timer.start()

        # Objects shortcuts. hit_number_key routes to the scale references while that mode
        # is active, so the range covers whichever picker has more entries.
        for i in range(1, max(controller.num_objects, len(scale_objects.CLASSES)) + 1):
            QShortcut(QKeySequence(str(i)),
                      self).activated.connect(functools.partial(controller.hit_number_key, i))
            QShortcut(QKeySequence(f"Ctrl+{i}"),
                      self).activated.connect(functools.partial(controller.hit_number_key, i))

        # cycle to the previous / next class without reaching for a number key
        QShortcut(QKeySequence(Qt.Key.Key_BracketLeft),
                  self).activated.connect(functools.partial(controller.cycle_object, -1))
        QShortcut(QKeySequence(Qt.Key.Key_BracketRight),
                  self).activated.connect(functools.partial(controller.cycle_object, 1))

        # next/prev frame shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(controller.on_prev_frame)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(controller.on_next_frame)

        # +/- 10 frames shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_Left | Qt.KeyboardModifier.ShiftModifier),
                    self).activated.connect(functools.partial(controller.on_prev_frame, 10))
        QShortcut(QKeySequence(Qt.Key.Key_Right | Qt.KeyboardModifier.ShiftModifier),
                    self).activated.connect(functools.partial(controller.on_next_frame, 10))
        
        # first/last frame shortcuts
        QShortcut(QKeySequence(Qt.Key.Key_Left | Qt.KeyboardModifier.AltModifier),
                    self).activated.connect(functools.partial(controller.on_prev_frame, 999999))
        QShortcut(QKeySequence(Qt.Key.Key_Right | Qt.KeyboardModifier.AltModifier),
                    self).activated.connect(functools.partial(controller.on_next_frame, 999999))
        
        # commit to permanent memory shortcut
        QShortcut(QKeySequence(Qt.Key.Key_C), self).activated.connect(controller.on_commit)

        # forward/backward "run" shortcuts -- mode aware: mask propagation in mask mode,
        # arch tracking in arch mode (so Space never triggers segmentation propagation
        # while a TRACK run is going)
        QShortcut(QKeySequence(Qt.Key.Key_F), self).activated.connect(controller.on_run_forward)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self).activated.connect(controller.on_run_forward)
        QShortcut(QKeySequence(Qt.Key.Key_B), self).activated.connect(controller.on_run_backward)
        
        # Toggle visualization mode
        QShortcut(QKeySequence(Qt.Key.Key_T), self).activated.connect(controller.on_toggle_vis_mode)

        # Robot arm: trust / distrust this frame's measured diameter. The one key you
        # press over and over while scrubbing the video, so it gets its own letter;
        # a no-op unless the robot arm is the active scale reference.
        QShortcut(QKeySequence(Qt.Key.Key_D), self).activated.connect(controller.on_arm_trust)

        # undo last click
        QShortcut(QKeySequence('Ctrl+Z'), self).activated.connect(controller.on_undo)

        # toggle polygon CUT (erase to background)
        QShortcut(QKeySequence(Qt.Key.Key_E), self).activated.connect(
            controller.on_toggle_polygon_erase)

        # open another video / workspace shortcut
        QShortcut(QKeySequence(Qt.Key.Key_O), self).activated.connect(self.open_loader)

        # quit shortcut
        QShortcut(QKeySequence(Qt.Key.Key_Q), self).activated.connect(self.close)

    def set_current_object_id(self, object_id: int):
        self.object_dial.blockSignals(True)
        self.object_dial.setValue(object_id)
        self.object_dial.blockSignals(False)

        btn = self.class_buttons.get(object_id)
        if btn is not None:
            btn.setChecked(True)     # idClicked only fires on user clicks: no loop

        self.set_object_color(object_id)
        self.update_class_name(object_id)

    def update_class_name(self, object_id: int):
        # keep the "Painting:" readout in step with number keys, buttons and [ / ] cycling
        name = custom_names.get(object_id, str(object_id))
        self.current_class_label.setText(f'{object_id}  {name}')

    def set_scale_class_id(self, cls_id: int):
        btn = self.scale_class_buttons.get(cls_id)
        if btn is not None:
            btn.setChecked(True)     # idClicked only fires on user clicks: no loop
        self.current_scale_label.setText(f'{cls_id}  {scale_objects.class_name(cls_id)}')

    def on_mode_combo_changed(self, index):
        self.controller.on_mode_change(index)

    def set_mode_index(self, index: int):
        """Show the tool page for mode index 0/1/2 and keep the dropdown on it.
        Controller-side switches (middle-click, a refused change while propagating)
        land here too, so the dropdown can never drift from the actual mode."""
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(index)
        self.mode_combo.blockSignals(False)
        self.mode_stack.setCurrentIndex(index)

    def resizeEvent(self, event):
        self.controller.show_current_frame()

    def text(self, text):
        self.console.moveCursor(QTextCursor.MoveOperation.End)
        self.console.insertPlainText(text + '\n')

    def set_canvas(self, image):
        height, width, channel = image.shape
        # if the image is RGBA, convert to RGB first by coloring the background green
        if channel == 4:
            image_rgb = image[:, :, :3].copy()
            alpha = image[:, :, 3].astype(np.float32) / 255
            green_bg = np.array([0, 255, 0])
            # soft blending
            image = (image_rgb * alpha[:, :, np.newaxis] + green_bg[np.newaxis, np.newaxis, :] *
                     (1 - alpha[:, :, np.newaxis])).astype(np.uint8)

        # zoom: crop a region around zoom_center and let it scale up to fill the canvas
        if self.zoom > 1.0:
            cw, ch = int(round(width / self.zoom)), int(round(height / self.zoom))
            cx, cy = self.zoom_center or (width / 2, height / 2)
            x0 = int(min(max(0, cx - cw / 2), width - cw))
            y0 = int(min(max(0, cy - ch / 2), height - ch))
            image = np.ascontiguousarray(image[y0:y0 + ch, x0:x0 + cw])
            self.crop_origin = (x0, y0)
            height, width = ch, cw
        else:
            self.crop_origin = (0, 0)

        bytesPerLine = 3 * width

        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
        self.main_canvas.setPixmap(
            QPixmap(
                qImg.scaled(self.main_canvas.size(), Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.FastTransformation)))

        self.main_canvas_size = self.main_canvas.size()
        self.image_size = qImg.size()

    def update_slider(self, value):
        self.lcd.setText('{: 3d} / {: 3d}'.format(value, self.controller.T - 1))
        self.tl_slider.setValue(value)

    def pixel_pos_to_image_pos(self, x, y):
        # Un-scale and un-pad the label coordinates into image coordinates
        oh, ow = self.image_size.height(), self.image_size.width()
        nh, nw = self.main_canvas_size.height(), self.main_canvas_size.width()

        h_ratio = nh / oh
        w_ratio = nw / ow
        dominate_ratio = min(h_ratio, w_ratio)

        # Solve scale
        x /= dominate_ratio
        y /= dominate_ratio

        # Solve padding
        fh, fw = nh / dominate_ratio, nw / dominate_ratio
        x -= (fw - ow) / 2
        y -= (fh - oh) / 2

        # Un-crop: shift back into full-image coordinates when zoomed
        x += self.crop_origin[0]
        y += self.crop_origin[1]

        return x, y

    def is_pos_out_of_bound(self, x, y):
        x, y = self.pixel_pos_to_image_pos(x, y)

        out_of_bound = ((x < 0) or (y < 0) or (x > self.w - 1) or (y > self.h - 1))

        return out_of_bound

    def get_scaled_pos(self, x, y):
        x, y = self.pixel_pos_to_image_pos(x, y)

        x = max(0, min(self.w - 1, x))
        y = max(0, min(self.h - 1, y))

        return x, y

    def forward_propagation_start(self):
        self.backward_run_button.setEnabled(False)
        self.forward_run_button.setText('Pause propagation')

    def backward_propagation_start(self):
        self.forward_run_button.setEnabled(False)
        self.backward_run_button.setText('Pause propagation')

    def pause_propagation(self):
        self.forward_run_button.setEnabled(True)
        self.backward_run_button.setEnabled(True)
        self.clear_all_mem_button.setEnabled(True)
        self.clear_non_perm_mem_button.setEnabled(True)
        self.forward_run_button.setText('Propagate forward')
        self.backward_run_button.setText('propagate backward')
        self.tl_slider.setEnabled(True)

    def process_events(self):
        QApplication.processEvents()

    def on_wheel(self, event):
        # zoom centered on the cursor; wheel up = zoom in
        cx, cy = self.get_scaled_pos(event.position().x(), event.position().y())
        if event.angleDelta().y() > 0:
            self.zoom = min(self.zoom * 1.25, 8.0)
        else:
            self.zoom = max(self.zoom / 1.25, 1.0)
        self.zoom_center = (cx, cy)
        self.controller.show_current_frame()

    def _display_ratio(self):
        oh, ow = self.image_size.height(), self.image_size.width()
        nh, nw = self.main_canvas_size.height(), self.main_canvas_size.width()
        return min(nh / oh, nw / ow)

    def on_mouse_press(self, event):
        # Ctrl+drag pans the zoomed view instead of placing a point
        if (event.modifiers() & Qt.KeyboardModifier.ControlModifier) and self.zoom > 1.0:
            self._pan_last = (event.position().x(), event.position().y())
            return

        ux, uy = self.pixel_pos_to_image_pos(event.position().x(), event.position().y())
        out_of_bound = (ux < 0) or (uy < 0) or (ux > self.w - 1) or (uy > self.h - 1)
        # only the arch tool may act outside the image (its tip can sit past the border)
        if out_of_bound and not getattr(self.controller, 'arch_mode', False):
            return

        ex = max(0, min(self.w - 1, ux))
        ey = max(0, min(self.h - 1, uy))
        if event.button() == Qt.MouseButton.LeftButton:
            action = 'left'
        elif event.button() == Qt.MouseButton.RightButton:
            action = 'right'
        elif event.button() == Qt.MouseButton.MiddleButton:
            action = 'middle'

        self.click_fn(action, ex, ey, ux, uy)

    def on_mouse_motion(self, event):
        if self._pan_last is not None:
            ratio = self._display_ratio()
            dx = (event.position().x() - self._pan_last[0]) / ratio
            dy = (event.position().y() - self._pan_last[1]) / ratio
            cx, cy = self.zoom_center or (self.w / 2, self.h / 2)
            self.zoom_center = (cx - dx, cy - dy)
            self._pan_last = (event.position().x(), event.position().y())
            self.controller.show_current_frame()
            return
        ux, uy = self.pixel_pos_to_image_pos(event.position().x(), event.position().y())
        ex = max(0, min(self.w - 1, ux))
        ey = max(0, min(self.h - 1, uy))
        self.on_mouse_motion_xy(ex, ey, ux, uy)

    def on_mouse_release(self, event):
        self._pan_last = None
        if self.release_fn is not None:
            self.release_fn()

    def on_play_video(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText('Play video')
        else:
            self.timer.start(self._play_interval_ms())
            self.play_button.setText('Stop video')

    def _play_interval_ms(self):
        # Base is 30 ticks/s. For >=1x we keep 30 ticks/s and jump several frames per tick
        # (see controller.on_play_video_timer) so speed is honoured even when a single frame
        # takes longer to render than the interval; for <1x we slow the tick rate instead.
        speed = self.play_speed_combo.currentData() or 1.0
        if speed >= 1.0:
            return max(1, int(round(1000.0 / 30.0)))
        return max(1, int(round(1000.0 / (30.0 * speed))))

    def on_play_speed_changed(self):
        # apply the new speed at once if a video is already playing
        if self.timer.isActive():
            self.timer.start(self._play_interval_ms())

    def ask_reset_arch(self, n_frames: int, has_current: bool, curr_ti: int):
        """Ask what RESET ARCH should clear. Returns 'frame', 'all', or None to cancel.
        Clearing is irreversible and can throw away a whole propagation run, so the scope is
        an explicit choice rather than a guess, and Cancel is the default button."""
        box = QMessageBox(self)
        box.setWindowTitle('Reset arch')
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(f'The arch is set on {n_frames} frame(s).')
        box.setInformativeText('Clearing it cannot be undone.')
        frame_btn = None
        if has_current:
            frame_btn = box.addButton(f'Clear frame {curr_ti} only',
                                      QMessageBox.ButtonRole.DestructiveRole)
        all_btn = box.addButton(f'Clear all {n_frames} frame(s)',
                                QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel_btn)
        for b in box.buttons():          # dialog buttons elide their labels the same way
            fit_button_text(b)
        box.exec()
        clicked = box.clickedButton()
        if clicked is all_btn:
            return 'all'
        if frame_btn is not None and clicked is frame_btn:
            return 'frame'
        return None

    def ask_reset_scale(self, name: str, n_frames: int, has_current: bool, curr_ti: int):
        """Ask what RESET REFERENCE should clear for the active reference. Returns
        'frame', 'all', or None to cancel -- same deal as ask_reset_arch: irreversible,
        so the scope is an explicit choice and Cancel is the default button."""
        box = QMessageBox(self)
        box.setWindowTitle('Reset reference')
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(f'"{name}" is drawn on {n_frames} frame(s).')
        box.setInformativeText('Clearing it cannot be undone. Other references are '
                               'not touched.')
        frame_btn = None
        if has_current:
            frame_btn = box.addButton(f'Clear frame {curr_ti} only',
                                      QMessageBox.ButtonRole.DestructiveRole)
        all_btn = box.addButton(f'Clear all {n_frames} frame(s)',
                                QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel_btn)
        for b in box.buttons():
            fit_button_text(b)
        box.exec()
        clicked = box.clickedButton()
        if clicked is all_btn:
            return 'all'
        if frame_btn is not None and clicked is frame_btn:
            return 'frame'
        return None

    def open_file(self, prompt):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self,
                                                   prompt,
                                                   "",
                                                   "Image files (*)",
                                                   options=options)
        return file_name

    def open_loader(self):
        """Show the video/workspace loader and, if a choice is made, hand it to the
        controller to swap the workspace in place (no restart)."""
        if getattr(self.controller, 'propagating', False):
            self.text('Stop propagation before opening another video.')
            return
        dialog = LoaderDialog(self,
                              raw_videos_root=self.cfg.get('raw_videos_root', './raw_videos'),
                              workspace_root=self.cfg.get('workspace_root', './workspace'),
                              current_workspace=self.cfg.get('workspace'),
                              extra_video_roots=self.cfg.get('extra_video_roots'))
        if dialog.exec() and dialog.selection:
            self.controller.load_workspace(**dialog.selection)

    def rebind_workspace(self, h, w, T, workspace):
        """Re-point the (reused) widgets at a freshly loaded workspace: new frame count,
        new image size, reset zoom, new window title. Called by controller.load_workspace."""
        self.h, self.w, self.T = h, w, T
        self.zoom = 1.0
        self.zoom_center = None
        self.crop_origin = (0, 0)
        self._pan_last = None
        if self.timer.isActive():          # stop playback tied to the old clip
            self.timer.stop()
            self.play_button.setText('Play video')
        self.tl_slider.blockSignals(True)
        self.tl_slider.setMaximum(max(0, T - 1))
        self.tl_slider.setValue(0)
        self.tl_slider.blockSignals(False)
        self.lcd.setText('{: 5d} / {: 5d}'.format(0, T - 1))
        self.setWindowTitle(f'SurgeNetSeg demo: {workspace}')

    def set_polygon_erase(self, enabled: bool):
        # keep the button in sync when CUT is toggled by its shortcut
        self.polygon_cut_button.blockSignals(True)
        self.polygon_cut_button.setChecked(enabled)
        self.polygon_cut_button.blockSignals(False)

    def set_object_color(self, object_id: int):
        r, g, b = custom_palette_np[object_id]
        rgb = f'rgb({r},{g},{b})'
        self.object_color.setFixedSize(50, 30)  # Make it square
        self.object_color.setStyleSheet(f'QLabel {{ background-color: {rgb}; border: 1px solid #d3d3d3; }}')

    def progressbar_update(self, progress: float):
        self.progressbar.setValue(int(progress * 100))
        self.process_events()
