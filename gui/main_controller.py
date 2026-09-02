import os
import shutil
from os import path
import logging
from typing import Literal, Optional

import cv2
# fix conflicts between qt5 and cv2
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

import torch
try:
    from torch import mps
except:
    print('torch.MPS not available.')
from torch import autocast
from torchvision.transforms.functional import to_tensor
import numpy as np
from PIL import Image
from omegaconf import DictConfig, open_dict

from gui.cutie.model.cutie import CUTIE
from gui.cutie.inference.inference_core import InferenceCore

from gui.interaction import *
from gui.interactive_utils import *
from gui.resource_manager import ResourceManager
from gui.gui import GUI
from gui.click_controller import ClickController
from gui.reader import PropagationReader, get_data_loader
from gui.exporter import convert_frames_to_video, convert_mask_to_binary
from gui.cutie.utils.download_models import download_models_if_needed

from gui.cutie.utils.palette import custom_names, custom_palette_np # added
from gui import retzius_arch
from gui import robot_arm
from gui import scale_objects
from gui import gui_bar
from gui.gui_utils import video_roots, workspace_name_for_video

log = logging.getLogger()

# Annotation modes, in the order of the GUI's Mode dropdown / tool-page stack.
MODE_INDEX = {'mask': 0, 'arch': 1, 'scale': 2}


class MainController():

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.initialized = False

        # setting up the workspace
        if cfg["workspace"] is None:
            if cfg["images"] is not None:
                basename = path.basename(cfg["images"])
            elif cfg["video"] is not None:
                basename = workspace_name_for_video(
                    cfg["video"], video_roots(cfg.get('raw_videos_root'),
                                              cfg.get('extra_video_roots')))
            else:
                raise NotImplementedError('Either images, video, or workspace has to be specified')

            cfg["workspace"] = path.join(cfg['workspace_root'], basename)

        # reading arguments
        self.cfg = cfg
        self.num_objects = cfg['num_objects']
        self.device = cfg['device']
        self.amp = cfg['amp']

        # initializing the network(s)
        self.initialize_networks()

        # main components
        self.res_man = ResourceManager(cfg)
        if 'workspace_init_only' in cfg and cfg['workspace_init_only']:
            return
        self.processor = InferenceCore(self.cutie, self.cfg)
        self.gui = GUI(self, self.cfg)

        # initialize control info
        self.length: int = self.res_man.length
        self.interaction: Interaction = None
        self.interaction_type: str = 'Click'
        self.curr_ti: int = 0
        self.curr_object: int = 1
        self.propagating: bool = False
        self.propagate_direction: Literal['forward', 'backward', 'none'] = 'none'
        self.last_ex = self.last_ey = 0

        # current frame info
        self.curr_frame_dirty: bool = False
        self.curr_image_np: np.ndarray = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.curr_image_torch: torch.Tensor = None
        self.curr_mask: np.ndarray = np.zeros((self.h, self.w), dtype=np.uint8)
        self.curr_prob: torch.Tensor = torch.zeros((self.num_objects + 1, self.h, self.w),
                                                   dtype=torch.float).to(self.device)
        self.curr_prob[0] = 1

        # visualization info
        self.vis_mode: str = 'mask overlay'
        self.vis_image: np.ndarray = None
        self.curr_depth_map: np.ndarray = None
        self._depth_cache: dict = {}
        self.save_visualization_mode: str = 'None'
        self.save_soft_mask: bool = False

        self.interacted_prob: torch.Tensor = None
        self.overlay_layer: np.ndarray = None
        self.overlay_layer_torch: torch.Tensor = None

        # the object id used for popup/layer overlay
        self.vis_target_objects = list(range(1, self.num_objects + 1))

        # Retzius arch tool state (ARCH button; geometry in gui/retzius_arch.py)
        # must be set before the first show_current_frame() below, since compose_current_im
        # draws the current frame's arch regardless of whether the tool is active
        self.arch_mode = False          # True while the tool is active (handles editable)
        self.arch_pending = []          # placement clicks so far: side, side, tip
        self.arch_drag = None           # (arch, handle_name) while a handle is dragged
        self.arch_tracking = False      # True while a TRACK run is going
        self._arch_track_dir = None     # which TRACK button is lit during that run
        self._syncing_arch_slider = False   # guards the height slider <-> arch feedback loop
        self._arch_tracker = None       # probe tracker, live only during a TRACK run
        self._arch_current = None       # the arch being carried through that run
        self._arch_us = None            # its probes' arc parameters
        self._arch_holds = 0            # consecutive frames the probes lost the tissue
        self.arch_by_frame = retzius_arch.load(self.res_man.workspace)  # {frame: Arch}

        # Scale-reference tool state (Mode -> "Scale annotation"; gui/scale_objects.py).
        # Same reason as the arch: compose_current_im draws the references on every frame,
        # whether or not the tool is active, so this must exist before the first redraw.
        self.scale_mode = False         # True while the tool is active (handles editable)
        self.scale_class = 1            # which reference the clicks/keys apply to
        self.scale_edit_points = False  # arm the interior tracking points for dragging
        self.scale_pending = []         # the first endpoint click, waiting for the second
        self.scale_drag = None          # (line, handle_name) while a handle is dragged
        self.scale_tracking = False     # True while a reference TRACK run is going
        self._scale_track_dir = None    # which TRACK button is lit during that run
        self._syncing_scale = False     # guards the mm box <-> object feedback loop
        self._scale_states = []         # per-object tracking state, live during a run
        # Robot arm (class 3) is not one of those: it is four points guessed off the
        # segmentation rather than a drawn segment, so it has its own tracker state.
        self.arm_drag = None            # 'e0'|'e1'|'t0'|'t1' while that point is dragged
        self._arm_comp = None           # (frame, HxW bool) cache of the arm's mask blob,
                                        # so the overlay does not re-run the components
        self._arm_state = None          # the arm's tracking state, live during a TRACK run
        # {frame: {class_id: ScaleLine}}, per-class mm, {frame: ArmMeasure}, and the seed
        # click that says WHICH instrument is the arm.
        (self.scale_by_frame, self.scale_mm, self.arm_by_frame, self.arm_seed,
         self.arm_calib, self.arm_track) = scale_objects.load(self.res_man.workspace)
        # the clip-level reconciliation: what is actually exported, and the summary the
        # readout shows. Rebuilt from the per-frame measurements, never persisted.
        self._arm_summary = None
        self._arm_reconcile()

        # GUI-bar boxes (gui/gui_bar.py): hand-drawn rectangles over on-screen junk the
        # extraction missed, tracked along with the references and painted out on demand.
        self.bar_mode = False           # True while COVER GUI is armed (drag draws a box)
        self.bar_pending = None         # the box being dragged right now
        self.bar_anchor = None          # where that drag started (a box grows from it)
        self.bars_by_frame = gui_bar.load(self.res_man.workspace)
        self._bar_states = []           # per-box tracking state, live during a TRACK run

        self.load_current_image_mask()
        self.show_current_frame()

        # initialize stuff
        self.update_memory_gauges()
        self.update_gpu_gauges()
        self.gui.work_mem_min.setValue(self.processor.memory.min_mem_frames)
        self.gui.work_mem_max.setValue(self.processor.memory.max_mem_frames)
        self.gui.long_mem_max.setValue(self.processor.memory.max_long_tokens)
        self.gui.mem_every_box.setValue(self.processor.mem_every)

        # for exporting videos
        self.output_fps = cfg['output_fps']
        self.output_bitrate = cfg['output_bitrate']

        # set callbacks
        self.gui.on_mouse_motion_xy = self.on_mouse_motion_xy
        self.gui.click_fn = self.click_fn
        self.gui.release_fn = self.on_mouse_release

        # Variables for polygon drawing and hovering first point
        self.polygon_points = []
        self.hover_first_point = False
        self.hover_threshold = 8  # pixels
        self.in_polygon_mode = False
        # CUT: a finalized polygon returns its interior to background instead of
        # painting curr_object. This edits only this frame's saved mask -- memory is
        # untouched, so a stray blob can be removed without resetting the memory bank.
        self.polygon_erase = False
        self._polygon_undo = None  # (ti, mask) snapshot, one level, for Ctrl+Z

        # measurement tool state (ANCHOR / SUL buttons; config in gui/segmenter.py)
        self.measure_mode = None        # None | 'anchor' | 'sul' | 'live'
        self.measure_points = []
        self.anchor_px = None           # pixel span of the last ANCHOR (= anchor_mm)
        self.anchor_mm = None           # physical size the last ANCHOR/ROBOT ANCHOR was set to
        self._anchor_target_mm = None   # mm value armed by on_anchor/on_robot_anchor, pending the click
        self.live_sul = False           # LIVE SUL: tracking points across frames
        self.tracker = None

        self.gui.show()
        self.gui.text('Initialized.')
        self.initialized = True

        # try to load the default overlay
        self._try_load_layer('./docs/uiuc.png')
        self.gui.set_object_color(self.curr_object)
        self.update_config()

    def initialize_networks(self) -> None:
        download_models_if_needed()
        self.cutie = CUTIE(self.cfg).eval().to(self.device)
        model_weights = torch.load(self.cfg.weights, map_location=self.device)
        self.cutie.load_weights(model_weights)

        self.click_ctrl = ClickController(self.cfg.ritm_weights, device=self.device)

    def cycle_object(self, delta: int):
        """Step to the next (+1) or previous (-1) class, wrapping around. Bound to ] / [.
        Routes through hit_number_key so buttons, colour, dial and redraw all stay in sync.
        In scale mode it cycles the scale references instead, for the same two keys."""
        if self.propagating:
            return
        n = len(scale_objects.CLASSES) if self.scale_mode else self.num_objects
        curr = self.scale_class if self.scale_mode else self.curr_object
        self.hit_number_key((curr - 1 + delta) % n + 1)

    def hit_number_key(self, number: int):
        if self.scale_mode:
            # the scale references get the number keys while their mode is active, so
            # picking one feels exactly like picking a segmentation class
            if number in scale_objects.CLASSES:
                self.on_scale_class(number)
            return
        if number == self.curr_object:
            return
        self.curr_object = number
        self.gui.object_dial.setValue(number)
        if self.click_ctrl is not None:
            self.click_ctrl.unanchor()
        self.gui.text(f'Current object changed to {number}.')
        self.gui.set_object_color(number)
        self.gui.set_current_object_id(number)
        self.show_current_frame()
    
    def on_toggle_polygon_erase(self):
        """CUT on/off: a finalized polygon subtracts to background instead of painting.

        Only this frame's mask is rewritten -- nothing in the memory bank changes, so
        this is the cheap way to remove a stray blob. Note the correction only sticks
        for later frames if it reaches memory: commit the fixed frame (C) and
        re-propagate, otherwise a propagation pass paints the blob straight back."""
        if self.propagating:
            return
        self.polygon_erase = not self.polygon_erase
        if not self.in_polygon_mode:
            # CUT is meaningless outside polygon mode; turn it on rather than
            # silently arming a mode the next middle-click would reveal
            self.in_polygon_mode = True
            self.polygon_points = []
            self.hover_first_point = False
        self.gui.set_polygon_erase(self.polygon_erase)
        self.gui.text('Polygon CUT ON: finalized polygons return to background.'
                      if self.polygon_erase else
                      'Polygon CUT OFF: finalized polygons paint the current object.')
        self.compose_polygon_overlay()
        self.update_canvas()

    def compose_polygon_overlay(self):
        # Reset to base visualization image
        self.compose_current_im()

        # Draw polygon points and lines
        pts = [(int(px), int(py)) for (px, py) in self.polygon_points]

        if self.polygon_erase:
            # CUT has no object colour -- red reads as "removing", and stays legible
            # against every class colour in the palette
            r, g, b = 255, 0, 0
        else:
            # Get color for the current object
            r, g, b = custom_palette_np[self.curr_object]
            r, g, b = int(r), int(g), int(b)

        # Draw lines between points
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                cv2.line(self.vis_image, pts[i], pts[i + 1], color=(r,g,b), thickness=1)

        # Draw points with hover effect on first point
        for i, pt in enumerate(pts):
            if i == 0 and self.hover_first_point:
                # Hover color: white
                color = (255, 255, 255)
                radius = 6
            else:
                # Normal color: yellow
                color = (r,g,b)
                radius = 4
            cv2.circle(self.vis_image, pt, radius=radius, color=color, thickness=-1)

    def click_fn(self, action: Literal['left', 'right', 'middle'], x: int, y: int,
                 ux: float = None, uy: float = None):
        # x, y are clamped to the image; ux, uy are the raw (unclamped) image coords,
        # which the arch tool uses so its tip can be placed/dragged past the border
        if self.propagating:
            return

        if self.arch_mode:
            # the arch tool owns all clicks; middle-click freezes the arcs and leaves
            if action == 'middle':
                self._exit_arch_mode()
            else:
                self._arch_click(action, x, y,
                                 x if ux is None else ux, y if uy is None else uy)
            return

        if self.scale_mode:
            # same deal for the scale references: they own the clicks, middle-click leaves
            if action == 'middle':
                self.set_mode('mask')
            else:
                self._scale_click(action, x, y)
            return

        if action == 'middle' and (self.measure_mode is not None or self.live_sul):
            # middle-click leaves the measurement/live tools and returns to the
            # original annotation mode (a further middle-click toggles click<->polygon)
            self._exit_measure_mode()
            return

        if self.measure_mode is not None:
            self._measure_click(action, x, y)
            return

        if not hasattr(self, 'in_polygon_mode'):
            self.in_polygon_mode = False  # new flag to track current mode

        if action == 'middle':
            # Toggle polygon mode
            self.in_polygon_mode = not self.in_polygon_mode
            self.polygon_points = []
            self.hover_first_point = False
            if not self.in_polygon_mode and self.polygon_erase:
                # never leave CUT armed behind the user's back: coming back to
                # polygon mode later should paint, as it always has
                self.polygon_erase = False
                self.gui.set_polygon_erase(False)
            mode_text = ('Polygon mode ON (CUT)' if self.polygon_erase else
                         'Polygon mode ON') if self.in_polygon_mode else 'Click mode ON'
            self.gui.text(mode_text)
            self.compose_current_im()
            self.update_canvas()
            return

        if self.in_polygon_mode:
            # In polygon drawing mode
            if action == 'left':
                if self.polygon_points:
                    first_pt = self.polygon_points[0]
                    dist = ((x - first_pt[0])**2 + (y - first_pt[1])**2)**0.5
                    if dist <= self.hover_threshold:
                        # Finalize polygon
                        # Close polygon loop
                        if self.polygon_points[-1] != first_pt:
                            self.polygon_points.append(first_pt)

                        # Create binary mask
                        mask = np.zeros((self.h, self.w), dtype=np.uint8)
                        pts_np = np.array([[(int(px), int(py)) for px, py in self.polygon_points]], dtype=np.int32)
                        cv2.fillPoly(mask, pts_np, color=1)
                        inside = mask > 0

                        # both paths overwrite labels, so keep one level of undo
                        self._polygon_undo = (self.curr_ti, self.curr_mask.copy())

                        if self.polygon_erase:
                            n = int(np.count_nonzero(self.curr_mask[inside]))
                            self.curr_mask[inside] = 0
                            text = f'Polygon cut: {n} px returned to background.'
                        else:
                            self.curr_mask[inside] = self.curr_object
                            text = 'Polygon finalized and added to segmentation.'

                        self.curr_frame_dirty = True
                        self.save_current_mask()

                        # ✅ Update probability map so it's used for propagation
                        self.curr_prob = index_numpy_to_one_hot_torch(self.curr_mask, self.num_objects + 1).to(self.device)

                        # drop any half-finished click interaction so it cannot
                        # re-apply the old probabilities over the edit
                        self.reset_this_interaction()

                        self.polygon_points = []
                        self.hover_first_point = False
                        self.show_current_frame()
                        self.gui.text(text)
                        return
                # Add new point
                self.polygon_points.append((x, y))
                self.gui.text(f'Polygon point added: ({x}, {y})')
                self.compose_polygon_overlay()
                self.update_canvas()
                return

            elif action == 'right':
                # Remove last point
                if self.polygon_points:
                    removed = self.polygon_points.pop()
                    self.gui.text(f'Removed polygon point: {removed}')
                    self.compose_polygon_overlay()
                    self.update_canvas()
                else:
                    self.gui.text('No points to remove.')
                return

            else:
                # Do nothing in polygon mode for other actions
                return

        # Not in polygon mode: do normal click interaction
        last_interaction = self.interaction
        new_interaction = None

        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            if action in ['left', 'right']:
                self.convert_current_image_mask_torch()
                image = self.curr_image_torch
                if (last_interaction is None or last_interaction.tar_obj != self.curr_object):
                    self.complete_interaction()
                    self.click_ctrl.unanchor()
                    new_interaction = ClickInteraction(image, self.curr_prob, (self.h, self.w),
                                                    self.click_ctrl, self.curr_object)
                    if new_interaction is not None:
                        self.interaction = new_interaction

                self.interaction.push_point(x, y, is_neg=(action == 'right'))
                self.interacted_prob = self.interaction.predict().to(self.device, non_blocking=True)
                self.update_interacted_mask()
                self.update_gpu_gauges()

    def on_segment(self):
        # Run best.pth on the current frame and load the result as the mask.
        if self.propagating:
            return
        from gui import segmenter
        self.gui.text('Segmenting current frame (best.pth)...')
        self.gui.process_events()
        self.curr_mask = segmenter.segment(self.curr_image_np, self.device)
        self.curr_image_torch = self.curr_prob = None
        self.reset_this_interaction()
        self.save_current_mask()
        self.show_current_frame()
        self.gui.text('Segmentation done.')

    def on_depth(self):
        if self.propagating:
            return
        from gui import depth_estimator
        if self.curr_ti in self._depth_cache:
            self.curr_depth_map = self._depth_cache[self.curr_ti]
        else:
            self.gui.text('Computing depth map...')
            self.gui.process_events()
            # frames spread across the clip let the static-overlay mask work on a single
            # DEPTH click (it needs several frames to measure temporal variance)
            overlay_frames = self._sample_clip_frames(depth_estimator._OV_N)
            self.curr_depth_map = depth_estimator.estimate_depth(
                self.curr_image_np, self.device, overlay_frames=overlay_frames)
            self._depth_cache[self.curr_ti] = self.curr_depth_map
            self.res_man.save_depth(self.curr_ti, self.curr_depth_map)  # persist like masks
        self.vis_mode = 'depth'
        self.gui.combo.setCurrentText('depth')
        self.show_current_frame()
        self.gui.text('Depth map ready.')

    def _sample_clip_frames(self, n: int):
        # n RGB frames spread evenly across the clip, for the static-overlay mask
        if self.T <= 1:
            return [self.curr_image_np]
        idxs = np.unique(np.linspace(0, self.T - 1, min(n, self.T)).astype(int))
        return [self.res_man.get_image(int(i)) for i in idxs]

    def on_anchor(self):
        from gui import segmenter
        self._arm_anchor(segmenter.ANCHOR_MM)

    def on_robot_anchor(self):
        from gui import segmenter
        self._arm_anchor(segmenter.ROBOT_ANCHOR_MM)

    def _arm_anchor(self, mm):
        self.set_mode('mask')       # the measurement tools own the clicks from here
        self.measure_mode = 'anchor'
        self.measure_points = []
        self._anchor_target_mm = mm
        self.gui.text(f'ANCHOR: left-click 2 points spanning the {mm:g} mm reference '
                      '(right-click to cancel).')

    def on_sul(self):
        # one click pair reports both plain SUL and depth-aware SUL (if DEPTH has been run).
        self.set_mode('mask')       # the measurement tools own the clicks from here
        self.measure_mode = 'sul'
        self.measure_points = []
        hint = '' if self.curr_depth_map is not None else ' (run DEPTH first for a depth-aware result too)'
        if self.anchor_px is None:
            self.gui.text(f'SUL: set ANCHOR first for a mm result. Left-click 2 points{hint}.')
        else:
            self.gui.text(f'SUL: left-click 2 points (right-click to cancel){hint}.')

    def on_live_sul(self):
        self.set_mode('mask')       # the measurement tools own the clicks from here
        self.measure_mode = 'live'
        self.measure_points = []
        self.live_sul = False
        self.tracker = None
        self.gui.text('LIVE SUL: left-click the 2 endpoints, then step/play through '
                      'frames to track them (RESET to stop).')

    def _exit_measure_mode(self):
        # leave anchor/sul/live tools, back to annotation. Keeps the anchor scale.
        self.measure_mode = None
        self.live_sul = False
        self.tracker = None
        self.measure_points = []
        self.show_current_frame()
        self.gui.text('Back to annotation mode.')

    def on_reset_measure(self):
        # clear all measurement state: points, anchor scale, live tracking, and the label
        self.measure_mode = None
        self.measure_points = []
        self.anchor_px = None
        self.anchor_mm = None
        self.live_sul = False
        self.tracker = None
        self.gui.measure_label.setText('SUL: -- mm  |  SUL depth: -- mm')
        self.show_current_frame()
        self.gui.text('Measurements reset.')

    def _start_live_sul(self):
        from gui import segmenter
        self.tracker = segmenter.PointTracker()
        self.tracker.init(self.curr_image_np, self.measure_points)
        self.live_sul = True
        self._draw_measure((0, 255, 0))
        self._update_sul_label(1.0, 'tracked', False)
        # a point seeded ON an instrument takes the instrument as its target template,
        # and no occlusion logic can recover that -- say so rather than track a robot arm
        blocked = self._occluder_mask(self.curr_ti)
        on_arm = (blocked is not None and
                  any(segmenter._blocked_at(blocked, p) for p in self.measure_points))
        self.gui.text('LIVE SUL active, but a point was placed on an instrument -- it will '
                      'track the arm, not the tissue. RESET and re-place it on clear tissue.'
                      if on_arm else 'LIVE SUL active. Step/play through frames.')

    def _live_sul_update(self):
        from gui import segmenter
        # same instrument masks the arch's probes use: a point that lands on a robot
        # arm is occluded, so it holds its last good spot instead of riding the arm
        res = self.tracker.step(self.curr_image_np, self._occluder_mask(self.curr_ti))
        self.measure_points = [r['pos'] for r in res]
        conf = min(r['conf'] for r in res)
        if any(r['failed'] for r in res):
            # lost for >1s: drop the line and stop tracking
            self.live_sul = False
            self.tracker = None
            self.measure_points = []
            self.show_current_frame()
            self.gui.measure_label.setText('SUL: -- mm  |  SUL depth: -- mm')
            self.gui.text('LIVE SUL lost for >1s — line removed, tracking stopped.')
            return
        failed = False
        if any(r['state'] == 'lost' for r in res):
            state, color = 'lost', (255, 0, 0)
        elif any(r['state'] == 'reacquired' for r in res):
            state, color = 'reacquired', (255, 165, 0)
        else:
            state, color = 'tracked', (0, 255, 0)
        self._draw_measure(color)
        self._update_sul_label(conf, state, failed)

    def _update_sul_label(self, conf, state, failed):
        from gui import segmenter
        dist = segmenter.euclidean(*self.measure_points)
        if self.anchor_px:
            mm = dist * segmenter.mm_per_pixel(self.anchor_px, self.anchor_mm)
            txt = f'SUL: {mm:.2f} mm  [{state} {conf * 100:.0f}%]'
        else:
            txt = f'SUL: {dist:.1f} px  [{state} {conf * 100:.0f}%]'
        if failed:
            txt += '  DRIFT/LOST - RESET'
        self.gui.measure_label.setText(txt)

    def _measure_click(self, action, x, y):
        from gui import segmenter
        if action != 'left':                       # right-click undoes the last point
            if self.measure_points:
                self.measure_points.pop()
                self._draw_measure()
            return

        self.measure_points.append((x, y))
        self._draw_measure()
        if len(self.measure_points) < 2:
            return

        mode = self.measure_mode
        if mode == 'live':
            self.measure_mode = None
            self._start_live_sul()
            return

        p, q = self.measure_points
        dist = segmenter.euclidean(p, q)
        dist3d = None
        if mode == 'sul' and self.curr_depth_map is not None:
            dp = segmenter.depth_at(self.curr_depth_map, p)
            dq = segmenter.depth_at(self.curr_depth_map, q)
            dist3d = segmenter.euclidean3d(p, q, dp, dq)
        self.measure_points = []
        if mode == 'anchor':
            self.measure_mode = None
            self.anchor_px = dist
            self.anchor_mm = self._anchor_target_mm
            scale = segmenter.mm_per_pixel(dist, self.anchor_mm)
            self.gui.measure_label.setText(f'Anchor: {dist:.1f} px = {self.anchor_mm:g} mm')
            self.gui.text(f'Anchor = {dist:.1f} px -> {scale:.4f} mm/px '
                          f'({self.anchor_mm:g} mm reference). Use SUL to measure.')
        else:
            # stay armed so you can keep measuring SULs against the same anchor until RESET
            self.measure_mode = mode
            if self.anchor_px:
                to_mm = segmenter.mm_per_pixel(self.anchor_px, self.anchor_mm)
                mm, mm3d = dist * to_mm, (dist3d * to_mm if dist3d is not None else None)
                label = f'SUL: {mm:.2f} mm'
                label += f'  |  SUL depth: {mm3d:.2f} mm' if mm3d is not None else '  |  SUL depth: run DEPTH first'
                self.gui.measure_label.setText(label)
                self.gui.text(f'SUL = {mm:.2f} mm'
                              + (f', SUL depth = {mm3d:.2f} mm' if mm3d is not None else '')
                              + '. Click 2 more points for another, or RESET.')
            else:
                label = f'SUL: {dist:.1f} px (no anchor)'
                label += f'  |  SUL depth: {dist3d:.1f} px (no anchor)' if dist3d is not None else '  |  SUL depth: run DEPTH first'
                self.gui.measure_label.setText(label)
                self.gui.text(f'SUL = {dist:.1f} px'
                              + (f', SUL depth = {dist3d:.1f} px' if dist3d is not None else '')
                              + '. Set ANCHOR to convert to mm.')

    def _draw_measure(self, color=(255, 255, 0)):
        self.compose_current_im()
        pts = [(int(px), int(py)) for px, py in self.measure_points]
        for pt in pts:
            cv2.circle(self.vis_image, pt, 4, color, -1)
        if len(pts) == 2:
            cv2.line(self.vis_image, pts[0], pts[1], color, 2)
        self.update_canvas()

    # --- annotation mode switching -------------------------------------------------
    # Three tools share the canvas and the clicks: mask painting, the Retzius arch and
    # the scale references. set_mode is the only thing that flips between them, so the
    # flags, the Mode dropdown and the visible tool page can never disagree.

    def _mode(self) -> str:
        return 'scale' if self.scale_mode else ('arch' if self.arch_mode else 'mask')

    def set_mode(self, mode: Literal['mask', 'arch', 'scale']):
        if self.propagating:
            self.gui.set_mode_index(MODE_INDEX[self._mode()])   # refuse, don't desync
            self.gui.text('Finish or pause the current run before switching mode.')
            return
        old = self._mode()
        # ponytail: sync the page/dropdown FIRST and even for a no-op switch. The flags and
        # the visible page must never be able to drift apart -- if a switch aborted halfway
        # (a failed save below), re-picking the mode is what puts the GUI back.
        self.gui.set_mode_index(MODE_INDEX[mode])
        if mode == old:
            return
        self.arch_mode = mode == 'arch'      # flags before any I/O, so a save that raises
        self.scale_mode = mode == 'scale'    # cannot leave the tool half-active
        if mode != 'mask':
            self.measure_mode = None         # the tool takes over the clicks
            self.measure_points = []
        try:
            if old == 'arch':                # leave: freeze what is drawn, drop the handles
                self.arch_pending = []
                self.arch_drag = None
                self._save_arches()
            elif old == 'scale':
                self.scale_pending = []
                self.scale_drag = None
                self._save_scales()
        except Exception as e:               # a failed write must not skip the redraw below,
            self.gui.text(f'Warning: could not save {old} annotations: {e}')
        self.show_current_frame()            # or the old tool's handles stay on the canvas
        self.gui.text(self._mode_hint(mode))

    def _mode_hint(self, mode):
        if mode == 'arch':
            if self.arch_by_frame.get(self.curr_ti) is not None:
                return ('ARCH: drag the side handles onto the cave walls and the tip along '
                        'the mid-line; the tip always stays centred between the sides. '
                        'An adjusted frame becomes a keyframe for TRACK. '
                        'Middle-click (or Mode -> Mask annotation) when done.')
            return ('ARCH: left-click the two sides of the cave, then the tip on the '
                    'urethra (the tip snaps to the mid-line and may go past the image '
                    'border). Right-click undoes. Then TRACK to carry it through the video.')
        if mode == 'scale':
            if self.scale_class == scale_objects.ARM_CLASS:
                return self._arm_hint()
            return (f'SCALE: drawing "{scale_objects.class_name(self.scale_class)}" '
                    f'({self._scale_mm(self.scale_class):g} mm). Left-click its two ends; '
                    'drag the end handles to correct. Pick another reference with the '
                    'coloured buttons or keys 1-3. Middle-click when done.')
        return 'Back to mask annotation.'

    def on_mode_change(self, index: int):
        """The Mode dropdown changed. Index order is MODE_INDEX."""
        for name, i in MODE_INDEX.items():
            if i == index:
                self.set_mode(name)
                return

    def on_arch(self):
        # toggle the Retzius arch tool (per-frame parabolic arc; gui/retzius_arch.py)
        self.set_mode('mask' if self.arch_mode else 'arch')

    def _exit_arch_mode(self):
        self.set_mode('mask')

    def _save_arches(self):
        retzius_arch.save(self.res_man.workspace, self.arch_by_frame)

    def _occluder_mask(self, ti):
        """Robot-instrument mask for frame ti, or None if nothing is annotated there.

        Reuses the ordinary segmentation: annotate the instruments as object
        'Non-anatomical' (retzius_arch.OCCLUDER_OBJECT_ID) on a few frames
        and propagate them like any other object -- TRACK then reads those saved masks
        and knows where its probes are blind. Nothing annotated -> None -> the arch
        tracks exactly as it would without any masking."""
        mask = self.res_man.get_mask(ti)
        if mask is None:
            return None
        blocked = (mask == retzius_arch.OCCLUDER_OBJECT_ID)
        return blocked if blocked.any() else None

    def _arch_click(self, action, x, y, ux, uy):
        # handles are grabbed in image coords; shrink the radius when zoomed in so the
        # on-screen grab distance stays roughly constant
        arch = self.arch_by_frame.get(self.curr_ti)
        grab = max(4.0, retzius_arch.GRAB_PX / self.gui.zoom)
        hit = retzius_arch.hit_test([arch] if arch is not None else [], ux, uy, grab)

        if action == 'left':
            if hit is not None:
                self.arch_drag = hit    # dragged in on_mouse_motion_xy, saved on release
            elif arch is not None:
                # once this frame has an arch it can only be adjusted by dragging its
                # own points -- no re-placing and no deleting from the GUI
                self.gui.text('Arch already set on this frame -- drag its handles to '
                              'adjust the points.')
            elif len(self.arch_pending) < 2:
                self.arch_pending.append((x, y))   # the sides live on the image
                if len(self.arch_pending) == 1:
                    self.gui.text('ARCH: first side set -- left-click the other side of the cave.')
                else:
                    self.gui.text('ARCH: sides set -- left-click the tip on the urethra '
                                  '(it snaps to the magenta mid-line, even past the border).')
            else:
                side_a, side_b = self.arch_pending
                arch = retzius_arch.Arch(side_a, side_b, 0.0)
                arch.set_apex(ux, uy)   # raw coords: the tip may start off-image
                self.arch_by_frame[self.curr_ti] = arch
                self.arch_pending = []
                self._save_arches()
                self.gui.text(f'Arch placed on frame {self.curr_ti}. Drag the handles to '
                              'calibrate, then TRACK to carry it through the video.')
        elif action == 'right' and self.arch_pending:
            # only undoes an in-progress placement click; a set arch cannot be deleted
            self.arch_pending.pop()
        self._draw_arch_overlay()

    # --- arch tracking (TRACK buttons) --------------------------------------------
    # A pass of its own, independent of Cutie: it only follows probe points, so it is
    # fast and -- unlike mask propagation -- it never rewrites the saved masks. Run it
    # in either direction from whichever frame the arch is set on.

    def _arch_seed(self, arch, image, blocked):
        """Start probe tracking from arch on image. False if too little of it is visible."""
        us, pts = retzius_arch.sample_probes(arch, self.w, self.h, blocked=blocked)
        if len(us) < 2:
            return False
        from gui import segmenter
        self._arch_tracker = segmenter.PointTracker()
        self._arch_tracker.init(image, pts)
        self._arch_us = np.asarray(us)
        self._arch_current = arch
        self._arch_holds = 0
        return True

    def _arch_step(self, ti, image, blocked):
        """One frame of arch tracking. False once it has lost the tissue for too long."""
        keyframe = self.arch_by_frame.get(ti)
        if keyframe is not None and keyframe.source == 'manual':
            # honor the user's correction: adopt it and re-seed the probes there
            if not self._arch_seed(keyframe, image, blocked):
                self._arch_current = keyframe   # too hidden to re-seed; keep the old probes
                self._arch_holds = 0
            return True

        res = self._arch_tracker.step(image, blocked)
        # probes the tracker still has a fix on; refit then judges whether they AGREE
        # (it returns None if too few of them vote together to trust the fit)
        good = [(u, r) for u, r in zip(self._arch_us, res) if r['state'] != 'lost']
        cand = None
        if len(good) >= 2:
            # trust = appearance x visibility; agreement is applied inside refit.
            # A probe behind an instrument contributes nothing.
            vis = retzius_arch.probe_visibility(blocked, [g[1]['pos'] for g in good])
            cand = retzius_arch.refit(self._arch_current,
                                      [g[0] for g in good],
                                      [g[1]['pos'] for g in good],
                                      [g[1]['conf'] * v for g, v in zip(good, vis)])
        if cand is not None:
            self._arch_current = cand
            self._arch_holds = 0
        else:
            self._arch_current = retzius_arch.Arch(self._arch_current.left,
                                                   self._arch_current.right,
                                                   self._arch_current.height,
                                                   source='hold', conf=0.0,
                                                   power=self._arch_current.power)
            self._arch_holds += 1
        self.arch_by_frame[ti] = self._arch_current
        return self._arch_holds <= retzius_arch.MAX_HOLD_FRAMES

    def _set_track_lit(self, direction: Optional[str] = None):
        """Light the TRACK button that is running (None lights neither).

        The buttons are checkable, so a click toggles them on by itself -- every path
        out of on_track_arch has to put them back, or a refused run leaves one lit."""
        self.gui.track_back_button.setChecked(direction == 'backward')
        self.gui.track_fwd_button.setChecked(direction == 'forward')

    def on_track_arch(self, direction: Literal['forward', 'backward'] = 'backward'):
        # follow the current frame's arch through the video, one frame at a time,
        # re-fitting the (always perfect) parabola to the probes that still agree
        if self.propagating:
            if self.arch_tracking:
                self.arch_tracking = False     # either TRACK button acts as pause
                self._set_track_lit(self._arch_track_dir)  # stays lit until the loop exits
            else:
                self._set_track_lit(None)      # mask propagation owns the run, not TRACK
            return
        arch = self.arch_by_frame.get(self.curr_ti)
        if arch is None:
            self._set_track_lit(None)
            self.gui.text('TRACK: set the arch on this frame first (ARCH button).')
            return
        start = self.curr_ti
        last = 0 if direction == 'backward' else self.T - 1
        if start == last:
            self._set_track_lit(None)
            edge = 'first' if direction == 'backward' else 'last'
            self.gui.text(f'TRACK: already at the {edge} frame.')
            return
        if not self._arch_seed(arch, self.curr_image_np, self._occluder_mask(start)):
            self._set_track_lit(None)
            self.gui.text('TRACK: too little of the arch is visible on this frame to track '
                          '(off-frame, or covered by instruments).')
            return
        self.gui.text(f'TRACK {direction} from frame {start} with {len(self._arch_us)} probes. '
                      'Click TRACK again to pause; gray arc = lost grip, amber = probes '
                      'disagreed. Masks are not touched.')

        step = -1 if direction == 'backward' else 1
        self.propagating = True            # blocks clicks/slider, like mask propagation
        self.arch_tracking = True
        self._arch_track_dir = direction
        self._set_track_lit(direction)     # lit for the whole run, like ARCH stays down
        self.gui.tl_slider.setEnabled(False)
        # Every frame is still tracked; the Speed selector only thins how often the arch
        # preview is redrawn (higher speed -> fewer redraws -> a faster run that needs less
        # watching). A lost-grip frame always redraws so you see where it stopped.
        redraw_every = self._redraw_every()
        n = 0
        stopped = None
        try:
            for t in range(start + step, last + step, step):
                if not self.arch_tracking:
                    stopped = f'paused at frame {t - step}.'
                    break
                image = self.res_man.get_image(t)
                ok = self._arch_step(t, image, self._occluder_mask(t))
                self.curr_ti = t                    # show progress live
                n += 1
                if n % redraw_every == 0 or not ok:
                    self.load_current_image_mask()
                    self.show_current_frame()
                self.gui.progressbar_update(abs(t - start) / abs(last - start))
                self.gui.process_events()
                if not ok:
                    stopped = (f'lost the tissue for {self._arch_holds} frames -- stopped at '
                               f'frame {t}. Correct the arch there (ARCH) and TRACK again.')
                    break
        finally:
            self.propagating = False
            self.arch_tracking = False
            self._arch_track_dir = None
            self._set_track_lit(None)
            self._arch_tracker = None
            self.gui.tl_slider.setEnabled(True)
            self.gui.progressbar_update(0)
            self._save_arches()
            # land on the last tracked frame (the redraw throttle may have skipped it)
            self.load_current_image_mask()
            self.show_current_frame()
        self.gui.text('TRACK: ' + (stopped or
                                   f'arch tracked to frame {last} (from frame {start}). '
                                   'Scrub through, correct any frame (ARCH), TRACK again.'))

    def on_reset_arch(self):
        # The deliberate escape hatch: an arch cannot be deleted by a stray click, but
        # a misplacement (or a propagation that went wrong) has to be undoable somehow.
        if self.propagating:
            self.gui.text('RESET ARCH: wait for propagation to finish or pause it first.')
            return
        n = len(self.arch_by_frame)
        if n == 0:
            self.gui.text('No arch to reset.')
            return
        scope = self.gui.ask_reset_arch(n, self.curr_ti in self.arch_by_frame, self.curr_ti)
        if scope is None:
            return
        if scope == 'frame':
            self.arch_by_frame.pop(self.curr_ti, None)
            msg = f'Arch cleared on frame {self.curr_ti}.'
        else:
            self.arch_by_frame.clear()
            msg = f'Arch cleared on all {n} frame(s).'
        self.arch_pending = []          # a half-placed arch would survive otherwise
        self.arch_drag = None
        self._save_arches()
        self.show_current_frame()       # stays in arch mode: place the new one right away
        self.gui.text(msg + ' Place a new one with ARCH.')

    def on_arch_height_slider(self, value):
        # Drive the tip's height directly. The tip is derived from it, so this reaches a
        # tip that is off-image (where there is no handle to grab) and the parabola stays
        # exact and centred on the mid-line the whole way.
        if self._syncing_arch_slider:
            return                      # we set the slider ourselves; not a user edit
        arch = self.arch_by_frame.get(self.curr_ti)
        if arch is None or self.propagating:
            return
        arch.height = float(value)
        arch.source = 'manual'          # a hand-set height is a keyframe, like a drag
        self._update_arch_height_label(arch)
        self._draw_arch_overlay()

    def on_arch_power_slider(self, value):
        # Shape exponent (in tenths), the one degree of freedom the two sides and the tip
        # do NOT pin down: they fix where the arc starts, ends and peaks, this fixes how
        # sharply it turns over in between.
        if self._syncing_arch_slider:
            return
        arch = self.arch_by_frame.get(self.curr_ti)
        if arch is None or self.propagating:
            return
        arch.power = value / 10.0
        arch.source = 'manual'
        self._update_arch_height_label(arch)
        self._draw_arch_overlay()

    def on_arch_height_released(self):
        # save once at the end of the gesture, not on every tick
        if self.arch_by_frame.get(self.curr_ti) is not None:
            self._save_arches()

    def _update_arch_height_label(self, arch):
        self.gui.arch_height_label.setText(
            'Arch tip: --' if arch is None else f'Arch tip: {arch.height:.0f} px')
        self.gui.arch_power_label.setText(
            'Sharpness: --' if arch is None else f'Sharpness: {arch.power:.1f}')

    def _sync_arch_slider(self):
        # keep the sliders showing the current frame's arch (disabled when there is none)
        arch = self.arch_by_frame.get(self.curr_ti)
        self._syncing_arch_slider = True
        self.gui.arch_height_slider.setEnabled(arch is not None)
        self.gui.arch_power_slider.setEnabled(arch is not None)
        if arch is not None:
            self.gui.arch_height_slider.setValue(int(round(arch.height)))
            self.gui.arch_power_slider.setValue(int(round(arch.power * 10)))
        self._update_arch_height_label(arch)
        self._syncing_arch_slider = False

    def _draw_arch_overlay(self):
        self.compose_current_im()       # arcs (and handles while editing) drawn in compose
        self.update_canvas()
        self._sync_arch_slider()        # dragging the tip handle moves the slider too

    # --- scale references (Mode -> "Scale annotation"; gui/scale_objects.py) ----------
    # Three straight segments of known physical length -- ruler, catheter tip, robot arm --
    # drawn and corrected by hand like the arch, one of each per frame. Everything lands in
    # <workspace>/scale_objects.json, where a depth-estimation loss can read each object's
    # real mm, its endpoints, its tracking points and the mm/px they imply per frame.

    def _scale_mm(self, cls_id):
        """The real length in mm currently set for a reference class."""
        return self.scale_mm.get(cls_id, scale_objects.default_mm(cls_id))

    def _scale_frame(self, ti):
        """{class_id: ScaleLine} on frame ti (empty dict if nothing is drawn there)."""
        return self.scale_by_frame.get(ti, {})

    def _scale_lines(self, ti):
        """The hand-drawn references on frame ti. The robot arm is not among them: it is
        measured off the mask (self.arm_by_frame), not drawn, so it has no ScaleLine to
        drag and none of the drawing/tracking code below should see one."""
        return [l for c, l in self._scale_frame(ti).items() if c != scale_objects.ARM_CLASS]

    def _save_scales(self):
        scale_objects.save(self.res_man.workspace, self.scale_by_frame, self.scale_mm,
                           self.w, self.h, self.arm_by_frame, self.arm_seed,
                           self.arm_calib, self.arm_track)

    def on_scale_class(self, cls_id: int):
        """Pick the reference the clicks and keys apply to (buttons / keys 1-3 / [ ])."""
        if self.propagating or cls_id not in scale_objects.CLASSES:
            return
        self.scale_class = int(cls_id)
        self.scale_pending = []         # a half-drawn object belongs to the old class
        self.arm_drag = None
        self.gui.set_scale_class_id(self.scale_class)
        self._draw_scale_overlay()
        if self.scale_class == scale_objects.ARM_CLASS:
            self.gui.text(self._arm_hint())
            return
        self.gui.text(f'Reference: {scale_objects.class_name(self.scale_class)} '
                      f'({self._scale_mm(self.scale_class):g} mm).')

    def _arm_hint(self):
        return (f'ROBOT ARM ({robot_arm.ARM_MM:g} mm across): FOUR POINTS -- the two where '
                'the arm enters the picture, and the two at the end of its straight shaft '
                '(the wrist, NOT the tip of the tool). 1) Left-click the arm (its '
                f'"{custom_names[robot_arm.ARM_OBJECT_ID]}" mask, segmented and propagated '
                'like any other object) to say which instrument it is: the four points are '
                'guessed off that mask. 2) Drag any of the four squares that landed in the '
                'wrong place -- the two on the border only slide ALONG it, the two at the '
                'shaft end move freely; the centerline runs between the pairs and the '
                'white scale line is the diameter, square to it, which is what gets '
                'exported. 3) TRACK '
                'carries the four points through the video, holding the two sides to the '
                'slope you set here and letting only their height and the tip slide (a '
                'frame you correct becomes a keyframe, and its slope is the one held from '
                'there on), or MEASURE CLIP re-guesses every frame off its mask. Then: D '
                '(or right-click) drops a frame you do not believe, and CALIB is the one '
                'knob for a segmentation that is fat or thin on EVERY frame -- the readout '
                'compares the arm against your ruler where both are drawn.')

    def on_scale_edit_points(self, on: bool):
        """Arm/disarm dragging of the interior tracking points. Off by default: normally
        the 5 points are derived from the two ends and a stray click must not move one."""
        self.scale_edit_points = bool(on)
        self.scale_drag = None
        self._draw_scale_overlay()
        self.gui.text('Tracking points can now be dragged -- they slide along the line, so '
                      'the reference stays straight.' if on else
                      'Tracking points locked to the line again.')

    def on_scale_mm(self, value: float):
        """Real length of the active reference. Only the ruler is editable; the change
        applies to every frame it is drawn on, since it is one physical span."""
        if self._syncing_scale or self.propagating:
            return
        cls_id = self.scale_class
        if scale_objects.CLASSES[cls_id]['fixed_mm']:
            return
        self.scale_mm[cls_id] = float(value)
        for objs in self.scale_by_frame.values():
            if cls_id in objs:
                objs[cls_id].mm = float(value)
        self._save_scales()
        self._draw_scale_overlay()

    def _scale_click(self, action, x, y):
        # handles are grabbed in image coords; shrink the radius when zoomed in so the
        # on-screen grab distance stays roughly constant
        if self.bar_mode:
            self._bar_click(action, x, y)
            return
        if self.scale_class == scale_objects.ARM_CLASS:
            self._arm_click(action, x, y)
            return
        lines = self._scale_lines(self.curr_ti)
        grab = max(4.0, scale_objects.GRAB_PX / self.gui.zoom)
        hit = scale_objects.hit_test(lines, x, y, grab, self.scale_edit_points)

        if action == 'left':
            if hit is not None:
                self.scale_drag = hit   # dragged in on_mouse_motion_xy, saved on release
            elif self.scale_class in self._scale_frame(self.curr_ti):
                # like the arch: once it is drawn on a frame it is corrected by dragging,
                # never silently re-placed (RESET REFERENCE is the way out)
                self.gui.text(f'{scale_objects.class_name(self.scale_class)} is already '
                              'drawn on this frame -- drag its end handles to adjust, or '
                              'use RESET REFERENCE.')
            elif not self.scale_pending:
                self.scale_pending.append((x, y))
                self.gui.text(f'SCALE: first end set -- left-click the other end of the '
                              f'{scale_objects.class_name(self.scale_class)}.')
            else:
                a, self.scale_pending = self.scale_pending[0], []
                line = scale_objects.ScaleLine(self.scale_class, a, (x, y),
                                               mm=self._scale_mm(self.scale_class))
                self.scale_by_frame.setdefault(self.curr_ti, {})[self.scale_class] = line
                self._save_scales()
                self.gui.text(f'{scale_objects.class_name(self.scale_class)} drawn on frame '
                              f'{self.curr_ti}: {line.length_px:.1f} px = {line.mm:g} mm '
                              f'({line.mm_per_px:.4f} mm/px). TRACK to carry it through '
                              'the video.')
        elif action == 'right' and self.scale_pending:
            self.scale_pending.pop()    # only undoes an in-progress placement click
        self._draw_scale_overlay()

    def on_reset_scale(self):
        # The escape hatch, per reference: a drawn object cannot be removed by a stray
        # click, but a misplacement (or a bad TRACK run) has to be undoable somehow.
        if self.propagating:
            self.gui.text('RESET REFERENCE: wait for the run to finish or pause it first.')
            return
        cls_id = self.scale_class
        arm = cls_id == scale_objects.ARM_CLASS
        name = scale_objects.class_name(cls_id)
        frames = (sorted(self.arm_by_frame) if arm else
                  [ti for ti, objs in self.scale_by_frame.items() if cls_id in objs])
        if not frames:
            self.gui.text(f'No {name} to reset.')
            return
        scope = self.gui.ask_reset_scale(name, len(frames), self.curr_ti in frames,
                                         self.curr_ti)
        if scope is None:
            return
        for ti in ([self.curr_ti] if scope == 'frame' else frames):
            if arm:
                self.arm_by_frame.pop(ti, None)
            else:
                self.scale_by_frame.get(ti, {}).pop(cls_id, None)
        self.scale_by_frame = {ti: objs for ti, objs in self.scale_by_frame.items() if objs}
        if arm and scope != 'frame':
            self.arm_seed = None        # a full reset forgets which instrument was the arm
        self._arm_comp = None
        self.scale_pending = []
        self.scale_drag = None
        self.arm_drag = None
        self._save_scales()
        self.show_current_frame()       # stays in scale mode: draw the new one right away
        self.gui.text(f'{name} cleared on '
                      + (f'frame {self.curr_ti}.' if scope == 'frame'
                         else f'all {len(frames)} frame(s).')
                      + (' Click the arm to measure it again.' if arm else
                         ' Draw a new one by clicking its two ends.'))

    def _sync_scale_widgets(self):
        # keep the mm box and the readout showing the active reference on this frame
        cls_id = self.scale_class
        spec = scale_objects.CLASSES[cls_id]
        arm = cls_id == scale_objects.ARM_CLASS
        line = self._scale_frame(self.curr_ti).get(cls_id)
        self._syncing_scale = True
        self.gui.scale_mm_box.setValue(self._scale_mm(cls_id))
        self.gui.scale_mm_box.setReadOnly(spec['fixed_mm'])
        self.gui.scale_mm_box.setEnabled(not spec['fixed_mm'])
        # the tracking-point handles and the trust toggle belong to different tools:
        # only one of the two is ever meaningful, so only one is ever enabled
        self.gui.scale_points_check.setEnabled(not arm)
        self.gui.arm_trust_check.setEnabled(arm)
        self.gui.arm_measure_btn.setEnabled(arm)
        self.gui.arm_calib_box.setEnabled(arm)
        self.gui.arm_calib_box.setValue(float(self.arm_calib))
        self.gui.arm_track_combo.setEnabled(arm)
        self.gui.arm_track_combo.setCurrentIndex(1 if self.arm_track == 'shaft' else 0)
        ann = self.arm_by_frame.get(self.curr_ti) if arm else None
        self.gui.arm_trust_check.setChecked(bool(ann is not None and ann.trusted))
        self._syncing_scale = False
        if arm:
            self.gui.scale_info_label.setText(self._arm_readout(ann))
        elif line is None:
            self.gui.scale_info_label.setText(f'{spec["name"]}: not drawn on this frame')
        else:
            state = '' if line.source == 'manual' else f'  [{line.source} '\
                                                       f'{line.conf * 100:.0f}%]'
            self.gui.scale_info_label.setText(
                f'{spec["name"]}: {line.length_px:.1f} px = {line.mm:g} mm  '
                f'({line.mm_per_px:.4f} mm/px){state}')

    def _arm_readout(self, ann):
        """The one-line status of the robot arm on this frame.

        Deliberately leads with the CLIP, not the frame: the number that matters is the
        reconciled one, and the per-frame reading is only interesting where it disagrees
        with its neighbours (which is the tool telling you the mask is ragged there, not
        the arm changing width)."""
        n = len(self.arm_by_frame)
        if not n:
            return 'Robot arm: click the arm to say which instrument it is, then Measure clip.'
        s = self._arm_summary or {}
        if not s.get('n_exported'):
            tally = f'   [{n} frames measured -- press Measure clip to reconcile them]'
        else:
            tally = (f'   [clip: {s["mm_per_px"]:.4f} mm/px, '
                     f'{s["n_exported"]}/{s["n_measured"]} exported, '
                     f'+-{s["scatter_pct"]:.1f}%'
                     + (f', JUMPED at frame {s["break_at"]}' if s.get('n_broken') else '')
                     + ']')
        if ann is None:
            return f'Robot arm: nothing found on this frame{tally}'
        if not ann.measured:
            return f'Robot arm: the four points are too close together here{tally}'
        by = 'you' if ann.trusted_by == 'user' else 'auto'
        state = f'ON ({by})' if ann.trusted else f'OFF ({by})'
        # the uncalibrated reading is shown only where CALIB has moved it, so the knob's
        # effect on this frame is visible rather than buried in one number
        drift = ('' if abs(self.arm_calib - 1.0) < 1e-9 else
                 f'  (raw {ann.diameter:.1f} px, calib x{self.arm_calib:g})')
        placed = {'manual': '  HAND-PLACED', 'tracked': '  tracked (points)',
                  'shaft': '  tracked (shaft fraction)',
                  'hold': '  HELD (tracking lost its grip)'}.get(ann.source, '')
        return (f'Robot arm: {ann.export_diameter:.1f} px = {robot_arm.ARM_MM:g} mm  '
                f'({ann.export_mm_per_px:.4f} mm/px){drift}{placed}  '
                f'{state}{tally}')

    def _arm_reconcile(self, announce=False):
        """Re-run the clip reconciliation over every measured frame.

        Cheap (it touches no masks -- only the numbers already measured), so it is run
        whenever the set of measurements changes rather than being a step to remember."""
        self._arm_summary = robot_arm.clip_scale(self.arm_by_frame, calib=self.arm_calib)
        if not announce:
            return self._arm_summary
        s = self._arm_summary
        if not s['n_measured']:
            self.gui.text('Robot arm: nothing measured yet. Click the arm, then Measure clip.')
            return s
        msg = (f'Robot arm: {s["n_exported"]}/{s["n_measured"]} frames exported at '
               f'{s["mm_per_px"]:.4f} mm/px (+-{s["scatter_pct"]:.1f}%). '
               f'Per-frame readings scattered {s["raw_scatter_pct"]:.1f}% before '
               f'reconciling; {s["n_rejected"]} frame(s) were dropped.')
        if s['n_broken']:
            # the one rejection worth acting on: the others are noise, this one says the
            # points came off the arm and names the frame to go and fix
            msg += (f' {s["n_broken"]} of those JUMPED away from the reading before them '
                    f'-- first at frame {s["break_at"]} (chain anchored at frame '
                    f'{s["anchor"]}). Go there, put the four points back on the arm, and '
                    'TRACK on from it: everything past a jump is dropped until you do.')
        ref = robot_arm.ruler_check(self.arm_by_frame, self.scale_by_frame)
        if ref:
            msg += (f' Ruler cross-check on {ref["n"]} frame(s): the ruler reads '
                    f'{ref["ref_mm_per_px"]:.4f} mm/px against the arm\'s '
                    f'{ref["arm_mm_per_px"]:.4f} -- x{ref["ratio"]:.3f}. Apply that in '
                    'Calib ONLY if the two are at a similar depth.')
        self.gui.text(msg)
        return s

    def on_arm_measure_clip(self):
        """Guess the four points off every frame's own mask, then reconcile the diameters.
        The alternative to TRACK, which carries one frame's points instead."""
        if self.propagating:
            self.gui.text('Measure clip: wait for the run to finish or pause it first.')
            return
        if self.arm_seed is None:
            self.gui.text('Measure clip: click the arm first, so the tool knows which '
                          'instrument it is.')
            return
        self.gui.text(f'Robot arm: measuring {self.num_frames} frames...')
        self.gui.progressbar_update(0)
        for ti in range(self.num_frames):
            ann = self.arm_by_frame.get(ti)
            if ann is None or not ann.manual:
                self._arm_measure(ti)       # a frame you placed by hand is never redone
            if ti % 25 == 0:
                self.gui.progressbar_update(ti / max(1, self.num_frames))
                self.gui.process_events()
        self._arm_reconcile(announce=True)
        self._save_scales()
        self._draw_scale_overlay()

    def on_arm_calib(self, value):
        """The one knob for a systematically fat or thin segmentation."""
        if self._syncing_scale or self.propagating:
            return
        self.arm_calib = float(value)
        self._arm_reconcile()
        self._save_scales()
        self._draw_scale_overlay()

    def on_arm_track_mode(self, mode: str):
        """Pick which of the two TRACK modes carries the arm. Both track the entry pair;
        they differ in what decides WHERE along the arm the diameter is read.

        points   the tracker carries the tip pair too, and the mask is asked for the width
                 wherever it landed. Follows a physical place on the arm, but it is
                 following two patches on the smooth featureless edges of a cylinder, so
                 the station can slide.
        shaft    the station is a FRACTION along the shaft, recorded when the run is seeded
                 and re-derived from each frame's own mask. Needs no texture and cannot
                 slide; costs you the assumption that the shaft end is found consistently.

        A frame says which mode wrote it (its 'source'), so two runs can be compared
        afterwards without having to remember which was which."""
        if self._syncing_scale or self.propagating:
            return
        self.arm_track = 'shaft' if str(mode).lower().startswith('shaft') else 'points'
        self._save_scales()
        self.gui.text(
            'Robot arm TRACK mode: SHAFT FRACTION -- the diameter is read at the same '
            'fraction along the arm on every frame, off that frame\'s mask. Nothing to '
            'lose grip on. Re-run TRACK to use it.' if self.arm_track == 'shaft' else
            'Robot arm TRACK mode: POINTS -- the tracker carries the tip pair and the '
            'width is read where it lands. Follows a real place on the arm, but the two '
            'points sit on smooth edges and can slide. Re-run TRACK to use it.')

    def _draw_scale_overlay(self):
        self.compose_current_im()       # segments (and handles while editing) in compose
        self.update_canvas()
        self._sync_scale_widgets()

    def _draw_scale(self, editing):
        """Overlay this frame's references onto self.vis_image."""
        scale_objects.draw(self.vis_image, self._scale_lines(self.curr_ti),
                           editing=editing, pending=self.scale_pending,
                           pending_color=scale_objects.CLASSES[self.scale_class]['color'],
                           edit_points=self.scale_edit_points)
        gui_bar.draw(self.vis_image, self.bars_by_frame.get(self.curr_ti, []),
                     pending=self.bar_pending)
        robot_arm.draw(self.vis_image, self.arm_by_frame.get(self.curr_ti),
                       comp=self._arm_component(self.curr_ti),
                       editing=editing and self.scale_class == scale_objects.ARM_CLASS)

    # --- robot arm (scale reference 3; gui/robot_arm.py) -----------------------------
    # Four points: where the arm enters the picture (one per side) and where its straight
    # shaft ends (one per side). Guessed off the mask, dragged where the guess was wrong,
    # carried through the video by TRACK. The diameter is the tip pair measured square to
    # the centerline between the two midpoints. Whether a frame's diameter is trustworthy
    # is yours to say -- the auto verdict only sets the initial state.

    def _arm_component(self, ti):
        """This frame's arm blob (HxW bool), or None. Cached: the overlay is recomposed on
        every redraw and connected components on a full frame is not free."""
        if self._arm_comp is not None and self._arm_comp[0] == ti:
            return self._arm_comp[1]
        ann = self.arm_by_frame.get(ti)
        if ann is None:
            return None
        comp = robot_arm.pick_component(robot_arm.arm_pixels(self.res_man.get_mask(ti)),
                                        self._arm_seeds(ti))
        self._arm_comp = (ti, comp)
        return comp

    def _arm_seeds(self, ti):
        """Points that identify WHICH instrument is the arm, best first: this frame's own
        centerline, the neighbouring frames' (so the choice carries along the video), and
        the click that first pointed the tool at it."""
        seeds = []
        for t in (ti, ti - 1, ti + 1):
            ann = self.arm_by_frame.get(t)
            if ann is not None:
                seeds += [ann.start, ann.end]
        seeds.append(self.arm_seed)
        return seeds

    def _arm_measure(self, ti, seeds=None):
        """Guess the arm's four points on frame ti off its mask and keep the result. seeds
        overrides which instrument to look for -- a click says so outright, and must not
        lose to whatever this frame happened to be pointing at before. Returns the
        ArmMeasure or None (no instrument mask on this frame, or no shaft in it)."""
        ann = robot_arm.measure_frame(self.res_man.get_mask(ti),
                                      self._arm_seeds(ti) if seeds is None else seeds)
        self._arm_comp = None
        if ann is None:
            self.arm_by_frame.pop(ti, None)
        else:
            # a fresh measurement is a new object, so a verdict you gave this frame has
            # to be carried onto it -- otherwise TRACK would silently re-trust it
            ann.adopt_verdict(self.arm_by_frame.get(ti))
            self.arm_by_frame[ti] = ann
        return ann

    def _arm_click(self, action, x, y):
        """Left-click grabs whichever of the four points is under the cursor, otherwise it
        points the tool at an instrument (re-guessing this frame from there). Right-click
        turns this frame's diameter off, the fast way to reject a frame while scrubbing."""
        ann = self.arm_by_frame.get(self.curr_ti)
        grab = max(4.0, robot_arm.GRAB_PX / self.gui.zoom)
        if action == 'right':
            self.on_arm_trust(False)
            return
        if action != 'left':
            return
        handle = None if ann is None else ann.hit(x, y, grab)
        if handle is not None:
            self.arm_drag = handle          # dragged in motion, saved on release
            return
        self.arm_seed = (float(x), float(y))
        ann = self._arm_measure(self.curr_ti, seeds=[self.arm_seed])
        if ann is None:
            # the steps fail for different reasons; say which one, or the fix gets
            # looked for in the wrong place
            self.gui.text('Robot arm: ' + robot_arm.diagnose(
                self.res_man.get_mask(self.curr_ti), [self.arm_seed]))
        else:
            self.gui.text(f'Robot arm on frame {self.curr_ti}: {self._arm_readout(ann)}. '
                          'Drag any of the four squares that landed wrong -- the two on '
                          'the border slide along it, the two at the end of the shaft go '
                          'anywhere; D turns this frame\'s diameter on/off.')
        self._draw_scale_overlay()

    def on_arm_trust(self, on: Optional[bool] = None):
        """Turn this frame's diameter on/off (the "Diameter trusted" box, D, right-click).
        Once you have said either way, no automatic verdict touches this frame again."""
        # D is a global shortcut, so the mode has to be checked here: it must do nothing
        # while you are painting masks, even though the arm is still the picked reference
        if self.propagating or not self.scale_mode \
                or self.scale_class != scale_objects.ARM_CLASS:
            return
        if self._syncing_scale:
            return                      # the checkbox echoing a redraw, not a user click
        ann = self.arm_by_frame.get(self.curr_ti)
        if ann is None:
            self.gui.text('Robot arm: nothing measured on this frame to turn on or off.')
            self._draw_scale_overlay()
            return
        state = ann.toggle(on)
        self._save_scales()
        self._draw_scale_overlay()
        if state:
            self.gui.text(f'Frame {self.curr_ti}: diameter ON -- {ann.diameter:.1f} px '
                          f'= {robot_arm.ARM_MM:g} mm ({ann.mm_per_px:.4f} mm/px), '
                          'exported as a scale reference.')
        else:
            self.gui.text(f'Frame {self.curr_ti}: diameter OFF -- the four points are '
                          'kept, nothing exported from this frame.'
                          + ('' if ann.measured else
                             ' (There is no diameter here anyway: the four points are on '
                             'top of each other.)'))

    def _arm_mask_comp(self, ti, ann):
        """Frame ti's arm blob, identified by where this annotation already sits."""
        return robot_arm.pick_component(
            robot_arm.arm_pixels(self.res_man.get_mask(ti)), [ann.start, ann.end])

    def _arm_seed_track(self, ti, image):
        """Start following the arm from frame ti. False if there is no arm annotated
        there to follow.

        Both modes track the two ENTRY points, which is what anchors the axis. They differ
        in where the measurement is taken: 'points' lets the tracker carry the tip pair
        too, 'shaft' records how far along the shaft the tip sits as a FRACTION and puts it
        back at that fraction on every frame's own mask."""
        ann = self.arm_by_frame.get(ti)
        if ann is None:
            self._arm_state = None
            return False
        from gui import segmenter
        tracker = segmenter.PointTracker()
        tracker.init(image, ann.points())
        frac = None
        if self.arm_track == 'shaft':
            frac = robot_arm.station_frac(ann, self._arm_mask_comp(ti, ann))
        # the slope of the two sides is taken from HERE and held for the whole run: the arm
        # is straight and rigid, so the frame the points were placed on is the one that
        # says which way its edges run (robot_arm.LOCK_SLOPE). Re-seeding on a keyframe
        # therefore adopts the slope you just drew, which is how a wrong one is corrected.
        self._arm_state = {'tracker': tracker, 'ann': ann, 'frac': frac,
                           'dirs': robot_arm.side_dirs(ann) if robot_arm.LOCK_SLOPE
                           else None}
        return True

    def _arm_pass(self, ti, image):
        """One frame of the arm during a TRACK run: carry the four points onto it. Returns
        None normally, or the sentence to stop the run with.

        A frame you placed by hand is a keyframe -- it is never overwritten, and the
        tracker re-seeds there, so a correction propagates from where you made it. A point
        the tracker lost is moved by however far the others went (the arm is rigid, so
        that is the best available guess), the entry pair is put back onto the image
        border it belongs on, and the tip pair is put back onto this frame's mask so the
        diameter is measured rather than carried (robot_arm.snap_width).

        The two sides are also put back onto the SLOPE they were seeded with, before the
        width is read: the arm is a straight shaft that does not bend, so the tracker gets
        to slide a side across the arm and along it, and nothing else. Without that, a
        pixel or two of drift per point per frame -- unavoidable on a smooth shaft with no
        texture to lock onto -- pivots the axis, and the axis is what the mask's chord is
        read square to below. An oblique chord across a cylinder is too wide, so the pivot
        would show up as the arm quietly gaining millimetres over a clip. Drag a point to
        set a different slope; from that keyframe on it is the one that is held
        (robot_arm.LOCK_SLOPE turns the whole thing off).

        The run STOPS if the entry pair has moved further in one frame than the arm
        possibly could (robot_arm.ENTRY_DRIFT_PX). That is the tracker having let go, and
        the frame is not written: everything measured from there on would be measured from
        the wrong place, and it is worth an annotator's minute now instead of a hundred
        frames to be found later.

        No occluder mask goes in: those masks ARE the arm, and the other references use
        them to avoid riding it. These four points are meant to ride it."""
        state = self._arm_state
        if state is None:
            return None
        keyframe = self.arm_by_frame.get(ti)
        if keyframe is not None and keyframe.manual:
            self._arm_seed_track(ti, image)
            return None
        res = state['tracker'].step(image)
        good = [r for r in res if r['state'] != 'lost']
        if not good:
            return None                 # nothing to go on: leave the frame as it was
        drift = np.mean([np.asarray(r['pos']) - np.asarray(p)
                         for r, p in zip(res, state['ann'].points())
                         if r['state'] != 'lost'], axis=0)
        pts = [r['pos'] if r['state'] != 'lost' else (np.asarray(p) + drift)
               for r, p in zip(res, state['ann'].points())]
        ann = state['ann'].moved_to(pts, 'tracked' if len(good) == 4 else 'hold',
                                    shape=(self.h, self.w), dirs=state.get('dirs'))
        moved = robot_arm.entry_drift(state['ann'], ann)
        if moved > robot_arm.ENTRY_DRIFT_PX:
            return (f'the robot arm let go at frame {ti} -- its entry point jumped '
                    f'{moved:.0f} px in one frame, and where the arm crosses the border '
                    f'cannot move more than about {robot_arm.ENTRY_DRIFT_PX:g}. Frame '
                    f'{ti} was not written. Put the four points back on the arm there and '
                    'TRACK again from it.')
        # Either way the WIDTH comes off this frame's mask, never carried along -- a
        # tracker cannot see how far apart a cylinder's two smooth edges have got. The
        # modes differ only in what picks the station:
        #   points  the tracker did; the mask is asked for the width where it landed
        #   shaft   the mask does, at the same fraction along the arm as on the seed frame
        comp = self._arm_mask_comp(ti, ann)
        if state['frac'] is None:
            robot_arm.snap_width(ann, comp)
        else:
            robot_arm.at_frac(ann, comp, state['frac'])
            ann.source = 'shaft' if len(good) == 4 else 'hold'
        ann.adopt_verdict(self.arm_by_frame.get(ti))
        self.arm_by_frame[ti] = ann
        state['ann'] = ann
        self._arm_comp = None   # the run reconciles the clip once, when it finishes
        return None

    # --- GUI-bar boxes (gui/gui_bar.py) ----------------------------------------------
    # A rectangle dragged over on-screen junk the extraction script missed. Tracked by the
    # same TRACK buttons as the references -- but by the box's INSIDE, which is flat and
    # constant, not its edges -- and finally painted black into the frames themselves.

    def on_gui_bar_mode(self, on: bool):
        """Arm/disarm drawing GUI boxes. While armed the drag owns the clicks, so a box
        can never be drawn by accident while a reference is being placed."""
        self.bar_mode = bool(on)
        self.bar_pending = None
        self._draw_scale_overlay()
        self.gui.text('COVER GUI: drag a box over the junk (right-click a box to remove '
                      'it). TRACK carries the boxes through the clip; BLANK FRAMES then '
                      'paints them black into the frames.' if on else
                      'COVER GUI off -- the boxes stay, the clicks go back to the '
                      'references.')

    def _bar_click(self, action, x, y):
        bars = self.bars_by_frame.get(self.curr_ti, [])
        if action == 'left':
            self.bar_anchor = (x, y)
            self.bar_pending = gui_bar.Bar(x, y, x, y)   # grown in motion, kept on release
            return
        if action == 'right':
            # topmost box under the cursor: the last drawn is the one you just misplaced
            hit = next((b for b in reversed(bars) if b.contains(x, y)), None)
            if hit is None:
                self.gui.text('No GUI box under the cursor on this frame.')
            else:
                bars.remove(hit)
                if not bars:
                    self.bars_by_frame.pop(self.curr_ti, None)
                gui_bar.save(self.res_man.workspace, self.bars_by_frame)
                self.gui.text(f'GUI box removed from frame {self.curr_ti}.')
            self._draw_scale_overlay()

    def on_clear_bars(self):
        """Delete the boxes on this frame or on all of them -- the way out of a box drawn
        wrong, or a TRACK run that carried one off the bar. Frames already blanked stay
        blanked: the pixels are gone, only the boxes go."""
        if self.propagating:
            self.gui.text('DELETE BOXES: wait for the run to finish or pause it first.')
            return
        frames = sorted(ti for ti, bars in self.bars_by_frame.items() if bars)
        if not frames:
            self.gui.text('No GUI boxes to delete.')
            return
        scope = self.gui.ask_reset_scale(
            'GUI box(es)', len(frames), self.curr_ti in frames, self.curr_ti,
            title='Delete GUI boxes',
            note='The boxes are deleted, not the picture: frames you have already '
                 'blanked stay blanked.')
        if scope is None:
            return
        for ti in ([self.curr_ti] if scope == 'frame' else frames):
            self.bars_by_frame.pop(ti, None)
        self.bar_pending = None
        gui_bar.save(self.res_man.workspace, self.bars_by_frame)
        self._draw_scale_overlay()
        self.gui.text('GUI boxes deleted on '
                      + (f'frame {self.curr_ti}.' if scope == 'frame'
                         else f'all {len(frames)} frame(s).'))

    def on_blank_bars(self):
        """Paint every frame's boxes black into <workspace>/images. Irreversible: those
        files are what the networks, the overlay and the export all read."""
        if self.propagating:
            self.gui.text('BLANK FRAMES: wait for the run to finish or pause it first.')
            return
        frames = sorted(ti for ti, bars in self.bars_by_frame.items() if bars)
        if not frames:
            self.gui.text('BLANK FRAMES: draw a box first (COVER GUI), then TRACK it '
                          'through the clip.')
            return
        if not self.gui.ask_blank_bars(len(frames)):
            return
        # the originals are copied aside ONCE, the first time anything is blanked: after
        # that images/ is the blanked version and images_backup/ is what came out of the
        # video, so a box drawn wrong is recoverable by copying the folder back
        backup = path.join(self.res_man.workspace, 'images_backup')
        if not path.exists(backup):
            shutil.copytree(self.res_man.image_dir, backup)
            self.gui.text(f'Originals copied to {backup}')
        self.gui.text(f'Blanking {len(frames)} frame(s)...')
        for i, ti in enumerate(frames):
            file = self.res_man.image_path(ti)
            im = np.array(Image.open(file).convert('RGB'))
            gui_bar.blank(im, self.bars_by_frame[ti])
            Image.fromarray(im).save(file, quality=95)
            self.res_man.get_image.invalidate((ti,))
            if i % 25 == 0:
                self.gui.progressbar_update(i / len(frames))
                self.gui.process_events()
        self.gui.progressbar_update(0)
        self._depth_cache.clear()       # the depth maps were computed on the old pixels
        self.load_current_image_mask()
        self.show_current_frame()
        self.gui.text(f'BLANK FRAMES: {len(frames)} frame(s) painted black under their '
                      'boxes; the originals are in images_backup/. The boxes are kept -- '
                      'blanking again is harmless.')

    def _bar_seed(self, ti, image):
        """Take this frame's boxes as the reference: their interiors are what every later
        frame is compared against. False if there are none."""
        bars = self.bars_by_frame.get(ti, [])
        self._bar_states = [{'bar': bar, 'patch': bar.interior(image)} for bar in bars]
        return bool(self._bar_states)

    def _bar_pass(self, ti, image):
        """One frame of a TRACK run: copy each box onto this frame, unless its inside has
        changed. Returns a stop message once it has (None while all is well).

        The box does not move -- the console paints the bar at a fixed place, so what is
        worth watching is not where the box is but WHAT IS UNDER IT. While the bar is
        there the interior is the same pixels frame after frame; the first frame where it
        is not is the frame the bar stopped being there, and the run ends rather than
        blanking tissue for the rest of the clip. A frame whose boxes you drew by hand is
        a keyframe: it is never overwritten, and the comparison re-seeds from it."""
        if not self._bar_states:
            return None
        if any(b.source == 'manual' for b in self.bars_by_frame.get(ti, [])):
            self._bar_seed(ti, image)
            return None
        drift = [gui_bar.changed(st['bar'].interior(image), st['patch'])
                 for st in self._bar_states]
        worst = max(drift)
        if worst > gui_bar.CHANGE_TOL:
            return (f'the inside of a GUI box changed at frame {ti} '
                    f'({worst:.1f} gray levels, limit {gui_bar.CHANGE_TOL:g}) -- the bar '
                    'is not there any more, so nothing past this frame was boxed. Check '
                    'the frame: if the bar IS still there, draw the box again here and '
                    'TRACK on.')
        self.bars_by_frame[ti] = [st['bar'].copy() for st in self._bar_states]
        return None

    # --- reference tracking (the scale page's TRACK buttons) -------------------------
    # Its own pass, like the arch's: only probe points are followed, so it is fast and it
    # never rewrites the saved masks. Every reference drawn on the start frame is carried
    # at once, each with its own tracker and its own lost-grip counter.

    def _scale_seed_one(self, line, image, blocked):
        """Tracking state for one reference, or None if too little of it is visible."""
        ts, pts = scale_objects.sample_probes(line, self.w, self.h, blocked=blocked)
        if len(ts) < 2:
            return None
        from gui import segmenter
        tracker = segmenter.PointTracker()
        tracker.init(image, pts)
        return {'cls': line.cls_id, 'tracker': tracker, 'ts': np.asarray(ts),
                'line': line, 'holds': 0}

    def _scale_seed(self, ti, image, blocked):
        """Seed every reference drawn on frame ti. False if none of them can be tracked."""
        self._scale_states = [s for s in
                              (self._scale_seed_one(l, image, blocked)
                               for l in self._scale_lines(ti)) if s is not None]
        return bool(self._scale_states)

    def _scale_step_one(self, state, ti, image, blocked):
        """One frame for one reference. False once it has lost its grip for too long.

        NOTE: this is where the robot arm's own tracking method plugs in -- for now all
        three references are carried by the same generic probe tracking."""
        keyframe = self._scale_frame(ti).get(state['cls'])
        if keyframe is not None and keyframe.source == 'manual':
            # honor the user's correction: adopt it and re-seed the probes there
            seeded = self._scale_seed_one(keyframe, image, blocked)
            if seeded is None:
                state['line'], state['holds'] = keyframe, 0   # too hidden to re-seed
            else:
                state.update(seeded)
            return True

        res = state['tracker'].step(image, blocked)
        # probes the tracker still has a fix on; refit then judges whether they AGREE
        good = [(t, r) for t, r in zip(state['ts'], res) if r['state'] != 'lost']
        cand = None
        if len(good) >= 2:
            # trust = appearance x visibility; agreement is applied inside refit.
            # A probe behind an instrument contributes nothing.
            vis = retzius_arch.probe_visibility(blocked, [g[1]['pos'] for g in good])
            cand = scale_objects.refit(state['line'],
                                       [g[0] for g in good],
                                       [g[1]['pos'] for g in good],
                                       [g[1]['conf'] * v for g, v in zip(good, vis)])
        if cand is not None:
            state['line'], state['holds'] = cand, 0
        else:
            state['line'] = state['line'].copy(source='hold', conf=0.0)
            state['holds'] += 1
        self.scale_by_frame.setdefault(ti, {})[state['cls']] = state['line']
        return state['holds'] <= scale_objects.MAX_HOLD_FRAMES

    def _scale_step(self, ti, image, blocked):
        """One frame for every reference still alive. False once they have all given up."""
        alive = [s for s in self._scale_states
                 if self._scale_step_one(s, ti, image, blocked)]
        self._scale_states = alive
        return bool(alive)

    def _set_scale_track_lit(self, direction: Optional[str] = None):
        """Light the reference TRACK button that is running (None lights neither)."""
        self.gui.scale_track_back_button.setChecked(direction == 'backward')
        self.gui.scale_track_fwd_button.setChecked(direction == 'forward')

    def on_track_scale(self, direction: Literal['forward', 'backward'] = 'backward'):
        if self.propagating:
            if self.scale_tracking:
                self.scale_tracking = False    # either TRACK button acts as pause
                self._set_scale_track_lit(self._scale_track_dir)
            else:
                self._set_scale_track_lit(None)   # some other run owns the loop
            return
        # the arm rides along with its own tracker: four points, seeded below
        arm_on = self.curr_ti in self.arm_by_frame
        bars_on = bool(self.bars_by_frame.get(self.curr_ti))
        if not self._scale_lines(self.curr_ti) and not arm_on and not bars_on:
            self._set_scale_track_lit(None)
            self.gui.text('TRACK: draw a reference on this frame first (or click the '
                          'robot arm to measure it).')
            return
        start = self.curr_ti
        last = 0 if direction == 'backward' else self.T - 1
        if start == last:
            self._set_scale_track_lit(None)
            edge = 'first' if direction == 'backward' else 'last'
            self.gui.text(f'TRACK: already at the {edge} frame.')
            return
        if not self._scale_seed(start, self.curr_image_np, self._occluder_mask(start)) \
                and not arm_on and not bars_on:
            self._set_scale_track_lit(None)
            self.gui.text('TRACK: too little of the reference(s) is visible on this frame '
                          'to track (off-frame, or covered by instruments).')
            return
        arm_mode = ''
        if arm_on:
            self._arm_seed_track(start, self.curr_image_np)
            # say which mode is ACTUALLY running: 'shaft' needs a shaft it can walk on the
            # seed frame, and without one it quietly becomes 'points' -- which would turn
            # an A/B comparison into a comparison of the same thing twice
            got_frac = (self._arm_state or {}).get('frac') is not None
            arm_mode = ('points -- SHAFT FRACTION could not seed here'
                        if self.arm_track == 'shaft' and not got_frac else
                        'shaft fraction' if got_frac else 'points')
        if bars_on:
            self._bar_seed(start, self.curr_image_np)
        names = ', '.join([scale_objects.class_name(s['cls']) for s in self._scale_states]
                          + ([f'Robot arm ({arm_mode})'] if arm_on else [])
                          + ([f'{len(self._bar_states)} GUI box(es)'] if bars_on else []))
        self.gui.text(f'TRACK {direction} from frame {start}: {names}. Click TRACK again '
                      'to pause; gray = lost grip, amber = the points disagreed. '
                      'Masks are not touched.'
                      + (' The arm is carried from THIS frame, so correct it here first '
                         '-- and each frame keeps its own trust verdict, so scrub back '
                         'through and turn off the ones you do not believe.'
                         if arm_on else ''))

        step = -1 if direction == 'backward' else 1
        self.propagating = True            # blocks clicks/slider, like mask propagation
        self.scale_tracking = True
        self._scale_track_dir = direction
        self._set_scale_track_lit(direction)
        self.gui.tl_slider.setEnabled(False)
        redraw_every = self._redraw_every()
        n = 0
        stopped = None
        try:
            for t in range(start + step, last + step, step):
                if not self.scale_tracking:
                    stopped = f'paused at frame {t - step}.'
                    break
                image = self.res_man.get_image(t)
                ok = self._scale_step(t, image, self._occluder_mask(t))
                if bars_on:
                    # the boxes end the run themselves, at the frame their inside changed
                    stopped = self._bar_pass(t, image)
                    if stopped:
                        self.curr_ti = t
                        break
                    ok = True
                if arm_on:
                    # the arm holds through an ordinary bad frame, but a jumped entry
                    # point ends the run then and there -- see _arm_pass
                    broke = self._arm_pass(t, image)
                    if broke:
                        stopped, self.curr_ti = broke, t
                        break
                    ok = True
                self.curr_ti = t                    # show progress live
                n += 1
                if n % redraw_every == 0 or not ok:
                    self.load_current_image_mask()
                    self.show_current_frame()
                self.gui.progressbar_update(abs(t - start) / abs(last - start))
                self.gui.process_events()
                if not ok:
                    stopped = (f'every reference lost its grip -- stopped at frame {t}. '
                               'Correct one there and TRACK again.')
                    break
        finally:
            if arm_on:
                self._arm_reconcile()      # the new frames change what the clip agrees on
            self.propagating = False
            self.scale_tracking = False
            self._scale_track_dir = None
            self._set_scale_track_lit(None)
            self._scale_states = []
            self._arm_state = None
            self.gui.tl_slider.setEnabled(True)
            self.gui.progressbar_update(0)
            self._save_scales()
            if bars_on:
                gui_bar.save(self.res_man.workspace, self.bars_by_frame)
            self._bar_states = []
            # land on the last tracked frame (the redraw throttle may have skipped it)
            self.load_current_image_mask()
            self.show_current_frame()
        self.gui.text('TRACK: ' + (stopped or
                                   f'references tracked to frame {last} (from frame '
                                   f'{start}). Scrub through, correct any frame, '
                                   'TRACK again.'))

    def on_mouse_release(self):
        if self.bar_pending is not None:
            bar, self.bar_pending = self.bar_pending, None
            if bar.big_enough():
                self.bars_by_frame.setdefault(self.curr_ti, []).append(bar)
                gui_bar.save(self.res_man.workspace, self.bars_by_frame)
                self.gui.text(f'GUI box drawn on frame {self.curr_ti} '
                              f'({bar.w:.0f}x{bar.h:.0f} px). TRACK carries it through the '
                              'clip; BLANK FRAMES paints the boxes black into the frames.')
            self._draw_scale_overlay()
        if self.arch_drag is not None:
            self.arch_drag = None
            self._save_arches()
        if self.scale_drag is not None:
            self.scale_drag = None
            self._save_scales()
        if self.arm_drag:
            # a hand-tuned frame changes what the clip agrees on, so the reconciliation is
            # re-run rather than left stale. It touches no masks, only the numbers.
            self.arm_drag = None
            self._arm_reconcile()
            self._save_scales()

    def load_current_image_mask(self, no_mask: bool = False):
        self.curr_image_np = self.res_man.get_image(self.curr_ti)
        self.curr_image_torch = None
        self.curr_depth_map = self._depth_cache.get(self.curr_ti)

        if not no_mask:
            loaded_mask = self.res_man.get_mask(self.curr_ti)
            if loaded_mask is None:
                self.curr_mask.fill(0)
            else:
                self.curr_mask = loaded_mask.copy()
            self.curr_prob = None

    def convert_current_image_mask_torch(self, no_mask: bool = False):
        if self.curr_image_torch is None:
            self.curr_image_torch = to_tensor(self.curr_image_np).to(self.device, non_blocking=True)

        if self.curr_prob is None and not no_mask:
            self.curr_prob = index_numpy_to_one_hot_torch(self.curr_mask, self.num_objects + 1).to(
                self.device, non_blocking=True)

    def compose_current_im(self):
        self.vis_image = get_visualization(self.vis_mode, self.curr_image_np, self.curr_mask,
                                           self.overlay_layer, self.vis_target_objects,
                                           depth=self.curr_depth_map)
        if self.vis_image is self.curr_image_np:
            # 'image'-style modes return the cached frame itself; copy before any overlay
            # drawing scribbles into the frame cache (which also feeds the networks)
            self.vis_image = self.vis_image.copy()
        arch = self.arch_by_frame.get(self.curr_ti)
        retzius_arch.draw(self.vis_image, [arch] if arch is not None else [],
                          editing=self.arch_mode, pending=self.arch_pending)
        self._draw_scale(editing=self.scale_mode)

    def update_canvas(self):
        self.gui.set_canvas(self.vis_image)

    def update_current_image_fast(self, invalid_soft_mask: bool = False):
        # fast path, uses gpu. Changes the image in-place to avoid copying
        # thus current_image_torch must be voided afterwards
        # do_no_save_soft_mask is an override to solve #41
        if self.vis_mode in ('depth', 'depth overlay', 'mask + depth overlay'):
            # depth has no torch fast path; fall back to numpy (cached depth map)
            self.vis_image = get_visualization(self.vis_mode, self.curr_image_np, self.curr_mask,
                                               self.overlay_layer, self.vis_target_objects,
                                               depth=self.curr_depth_map)
        else:
            self.vis_image = get_visualization_torch(self.vis_mode, self.curr_image_torch,
                                                     self.curr_prob, self.overlay_layer_torch,
                                                     self.vis_target_objects)
        self.curr_image_torch = None
        self.vis_image = np.ascontiguousarray(self.vis_image)
        if self.vis_image is self.curr_image_np:
            # some modes return the cached frame itself; copy before drawing on it
            self.vis_image = self.vis_image.copy()
        # the frame's arc and references ride along during propagation/export too
        # (no editing handles)
        arch = self.arch_by_frame.get(self.curr_ti)
        retzius_arch.draw(self.vis_image, [arch] if arch is not None else [])
        self._draw_scale(editing=False)
        save_visualization = self.save_visualization_mode in [
            'Propagation only (higher quality)', 'Always'
        ]
        if save_visualization and not invalid_soft_mask:
            self.res_man.save_visualization(self.curr_ti, self.vis_mode, self.vis_image)
        if self.save_soft_mask and not invalid_soft_mask:
            self.res_man.save_soft_mask(self.curr_ti, self.curr_prob.cpu().numpy())
        self.gui.set_canvas(self.vis_image)

    def show_current_frame(self, fast: bool = False, invalid_soft_mask: bool = False):
        # Re-compute overlay and show the image
        if fast:
            self.update_current_image_fast(invalid_soft_mask)
        else:
            self.compose_current_im()
            if self.save_visualization_mode == 'Always':
                self.res_man.save_visualization(self.curr_ti, self.vis_mode, self.vis_image)
            self.update_canvas()

        self.gui.update_slider(self.curr_ti)
        self.gui.frame_name.setText(self.res_man.names[self.curr_ti] + '.jpg')
        self._sync_arch_slider()
        self._sync_scale_widgets()

    def set_vis_mode(self):
        self.vis_mode = self.gui.combo.currentText()
        self.show_current_frame()

    def save_current_mask(self):
        # save mask to hard disk
        self.res_man.save_mask(self.curr_ti, self.curr_mask)

    def on_slider_update(self):
        # if we are propagating, the on_run function will take care of everything
        # don't do duplicate work here
        self.curr_ti = self.gui.tl_slider.value()
        if not self.propagating:
            # with self.vis_cond:
            #     self.vis_cond.notify()
            if self.curr_frame_dirty:
                self.save_current_mask()
            self.curr_frame_dirty = False

            self.reset_this_interaction()
            self.arch_pending = []      # placement clicks don't carry across frames
            self.scale_pending = []
            self.curr_ti = self.gui.tl_slider.value()
            self.load_current_image_mask()
            self.show_current_frame()

            if self.live_sul and self.tracker is not None:
                self._live_sul_update()

    def on_run_forward(self):
        """F / Space: the forward "run" for whatever mode is active. In arch mode this is
        arch tracking (which has its own pause via arch_tracking); in mask mode it is
        segmentation propagation. Routing through here keeps the spacebar off segmentation
        propagation while the arch tool is active -- during a TRACK run self.propagating is
        already True, so the old direct binding would hit on_forward_propagation's pause
        branch and yank the lock out from under the tracker."""
        if self.arch_mode:
            self.on_track_arch('forward')
        elif self.scale_mode:
            self.on_track_scale('forward')
        else:
            self.on_forward_propagation()

    def on_run_backward(self):
        """B: the backward "run" for the active mode (arch/reference tracking vs mask
        propagation)."""
        if self.arch_mode:
            self.on_track_arch('backward')
        elif self.scale_mode:
            self.on_track_scale('backward')
        else:
            self.on_backward_propagation()

    def on_forward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
            self.propagate_direction = 'none'
        else:
            self.propagate_fn = self.on_next_frame
            self.gui.forward_propagation_start()
            self.propagate_direction = 'forward'
            self.on_propagate()

    def on_backward_propagation(self):
        if self.propagating:
            # acts as a pause button
            self.propagating = False
            self.propagate_direction = 'none'
        else:
            self.propagate_fn = self.on_prev_frame
            self.gui.backward_propagation_start()
            self.propagate_direction = 'backward'
            self.on_propagate()

    def on_pause(self):
        self.propagating = False
        self.gui.text(f'Propagation stopped at t={self.curr_ti}.')
        self.gui.pause_propagation()

    def on_propagate(self):
        # start to propagate
        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            self.convert_current_image_mask_torch()

            self.gui.text(f'Propagation started at t={self.curr_ti}.')
            self.processor.clear_sensory_memory()
            self.curr_prob = self.processor.step(self.curr_image_torch,
                                                 self.curr_prob[1:],
                                                 idx_mask=False)
            self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)
            # clear
            self.interacted_prob = None
            self.reset_this_interaction()
            # override this for #41
            self.show_current_frame(fast=True, invalid_soft_mask=True)

            self.propagating = True
            self.gui.clear_all_mem_button.setEnabled(False)
            self.gui.clear_non_perm_mem_button.setEnabled(False)
            self.gui.tl_slider.setEnabled(False)

            dataset = PropagationReader(self.res_man, self.curr_ti, self.propagate_direction)
            loader = get_data_loader(dataset, self.cfg.num_read_workers)

            # propagate till the end. Every frame is stepped and saved; the Speed selector
            # only thins out how often we redraw the preview (higher speed -> fewer redraws
            # -> faster), while process_events stays every frame so Pause responds at once.
            redraw_every = self._redraw_every()
            n = 0
            for data in loader:
                if not self.propagating:
                    break
                self.curr_image_np, self.curr_image_torch = data
                self.curr_image_torch = self.curr_image_torch.to(self.device, non_blocking=True)
                self.propagate_fn()

                self.curr_prob = self.processor.step(self.curr_image_torch)
                self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)

                self.save_current_mask()

                n += 1
                if n % redraw_every == 0:
                    self.show_current_frame(fast=True)
                    self.update_memory_gauges()
                self.gui.process_events()

                if self.curr_ti == 0 or self.curr_ti == self.T - 1:
                    break

            self.propagating = False
            self.curr_frame_dirty = False
            self.on_pause()
            self.on_slider_update()
            self.gui.process_events()

    def pause_propagation(self):
        self.propagating = False

    def on_commit(self):
        if self.interacted_prob is None:
            # get mask from disk
            self.load_current_image_mask()
        else:
            # get mask from interaction
            self.complete_interaction()
            self.update_interacted_mask()

        with autocast(self.device, enabled=(self.amp and self.device == 'cuda')):
            self.convert_current_image_mask_torch()
            self.gui.text(f'Permanent memory saved at {self.curr_ti}.')
            self.curr_prob = self.processor.step(self.curr_image_torch,
                                                 self.curr_prob[1:],
                                                 idx_mask=False,
                                                 force_permanent=True)
            self.update_memory_gauges()
            self.update_gpu_gauges()

    def on_undo(self):
        if self.propagating:
            return

        if self.arch_mode:
            # arch mode owns Ctrl+Z the same way it owns clicks: undo an in-progress
            # placement click if there is one, else no-op -- never fall through to
            # unrelated segmentation/polygon undo state while editing the arch
            if self.arch_pending:
                self.arch_pending.pop()
                self.gui.text('Removed last arch placement point.')
                self._draw_arch_overlay()
            return

        if self.scale_mode:
            # same rule for the references: undo an in-progress placement click, and
            # never fall through to unrelated segmentation/polygon undo state
            if self.scale_pending:
                self.scale_pending.pop()
                self.gui.text('Removed last reference placement point.')
                self._draw_scale_overlay()
            return

        if self.in_polygon_mode and self.polygon_points:
            removed = self.polygon_points.pop()
            self.hover_first_point = False
            self.gui.text(f'Removed polygon point: {removed}')
            self.compose_polygon_overlay()
            self.update_canvas()
            return

        if (self.in_polygon_mode and self._polygon_undo is not None
                and self._polygon_undo[0] == self.curr_ti):
            # nothing half-drawn -> undo the last finalized polygon on this frame.
            # One level only, and it is dropped once you leave the frame.
            _, snapshot = self._polygon_undo
            self._polygon_undo = None
            self.curr_mask = snapshot
            self.curr_prob = index_numpy_to_one_hot_torch(self.curr_mask,
                                                          self.num_objects + 1).to(self.device)
            self.curr_frame_dirty = True
            self.save_current_mask()
            self.reset_this_interaction()
            self.show_current_frame()
            self.gui.text('Undid the last finalized polygon.')
            return

        if not isinstance(self.interaction, ClickInteraction):
            self.gui.text('Nothing to undo.')
            return

        undo_mask = self.click_ctrl.undo()
        if undo_mask is None:
            # Revert to the pre-interaction state
            self.curr_prob = self.interaction.prev_mask.clone()
            self.curr_mask = torch_prob_to_numpy_mask(self.curr_prob)
            self.save_current_mask()
            self.interacted_prob = None
            self.interaction = None
            self.show_current_frame()
            return

        self.interaction.obj_mask = undo_mask.to(self.device, non_blocking=True)
        self.interaction.first_click = False
        self.interacted_prob = self.interaction.predict().to(self.device, non_blocking=True)
        self.update_interacted_mask()
        self.update_gpu_gauges()

    def _play_speed(self):
        """Current speed multiplier from the GUI's Speed selector (>=0.5). Drives playback
        step size and the propagate/track preview-redraw cadence. Defaults to 1x."""
        data = self.gui.play_speed_combo.currentData()
        try:
            return float(data) if data else 1.0
        except (TypeError, ValueError):
            return 1.0

    def _redraw_every(self):
        """During propagate/track every frame is still processed and saved; only the preview
        redraw is throttled. At Nx speed we redraw ~every N frames, so a faster setting spends
        less time drawing and finishes sooner (annotation density is unchanged)."""
        return max(1, int(round(self._play_speed())))

    def on_play_video_timer(self):
        # >=1x jumps several frames per tick so playback speeds up even when each frame takes
        # longer to render than the timer interval (preview only -- nothing is re-annotated)
        step = max(1, int(round(self._play_speed())))
        self.curr_ti += step
        if self.curr_ti > self.T - 1:
            self.curr_ti = 0
        self.gui.tl_slider.setValue(self.curr_ti)

    def on_export_visualization(self):
        # NOTE: Save visualization at the end of propagation
        image_folder = path.join(self.cfg['workspace'], 'visualization', self.vis_mode)
        save_folder = self.cfg['workspace']
        if path.exists(image_folder):
            # Sorted so frames will be in order
            output_path = path.join(save_folder, f'visualization_{self.vis_mode}.mp4')
            self.gui.text(f'Exporting visualization -- please wait')
            self.gui.process_events()
            convert_frames_to_video(image_folder,
                                    output_path,
                                    fps=self.output_fps,
                                    bitrate=self.output_bitrate,
                                    progress_callback=self.gui.progressbar_update)
            self.gui.text(f'Visualization exported to {output_path}')
            self.gui.progressbar_update(0)
        else:
            self.gui.text(f'No visualization images found in {image_folder}')

    def on_export_binary(self):
        # export masks in binary format for other applications, e.g., ProPainter
        mask_folder = path.join(self.cfg['workspace'], 'masks')
        save_folder = path.join(self.cfg['workspace'], 'binary_masks')
        if path.exists(mask_folder):
            os.makedirs(save_folder, exist_ok=True)
            self.gui.text(f'Exporting binary masks -- please wait')
            self.gui.process_events()
            convert_mask_to_binary(mask_folder,
                                   save_folder,
                                   self.vis_target_objects,
                                   progress_callback=self.gui.progressbar_update)
            self.gui.text(f'Binary masks exported to {save_folder}')
            self.gui.progressbar_update(0)
        else:
            self.gui.text(f'No masks found in {mask_folder}')

    def on_object_dial_change(self):
        object_id = self.gui.object_dial.value()
        self.hit_number_key(object_id)

    def on_fps_dial_change(self):
        self.output_fps = self.gui.fps_dial.value()

    def on_bitrate_dial_change(self):
        self.output_bitrate = self.gui.bitrate_dial.value()

    def update_interacted_mask(self):
        self.curr_prob = self.interacted_prob
        self.curr_mask = torch_prob_to_numpy_mask(self.interacted_prob)
        self.save_current_mask()
        self.show_current_frame()
        self.curr_frame_dirty = False

    def reset_this_interaction(self):
        self.complete_interaction()
        self.interacted_prob = None
        if self.click_ctrl is not None:
            self.click_ctrl.unanchor()

    def on_reset_mask(self):
        self.curr_mask.fill(0)
        if self.curr_prob is not None:
            self.curr_prob.fill_(0)
        self.curr_frame_dirty = True
        self.save_current_mask()
        self.reset_this_interaction()
        self.show_current_frame()

    def on_reset_object(self):
        self.curr_mask[self.curr_mask == self.curr_object] = 0
        if self.curr_prob is not None:
            self.curr_prob[self.curr_object] = 0
        self.curr_frame_dirty = True
        self.save_current_mask()
        self.reset_this_interaction()
        self.show_current_frame()

    def complete_interaction(self):
        if self.interaction is not None:
            self.interaction = None

    def on_prev_frame(self, step=1):
        new_ti = max(0, self.curr_ti - step)
        self.gui.tl_slider.setValue(new_ti)

    def on_next_frame(self, step=1):
        new_ti = min(self.curr_ti + step, self.length - 1)
        self.gui.tl_slider.setValue(new_ti)

    def update_gpu_gauges(self):
        if 'cuda' in self.device:
            info = torch.cuda.mem_get_info()
            global_free, global_total = info
            global_free /= (2**30)
            global_total /= (2**30)
            global_used = global_total - global_free

            self.gui.gpu_mem_gauge.setFormat(f'{global_used:.1f} GB / {global_total:.1f} GB')
            self.gui.gpu_mem_gauge.setValue(round(global_used / global_total * 100))

            used_by_torch = torch.cuda.max_memory_allocated() / (2**30)
            self.gui.torch_mem_gauge.setFormat(f'{used_by_torch:.1f} GB / {global_total:.1f} GB')
            self.gui.torch_mem_gauge.setValue(round(used_by_torch / global_total * 100 / 1024))
        elif 'mps' in self.device:
            mem_used = mps.current_allocated_memory() / (2**30)
            self.gui.gpu_mem_gauge.setFormat(f'{mem_used:.1f} GB')
            self.gui.gpu_mem_gauge.setValue(0)
            self.gui.torch_mem_gauge.setFormat('N/A')
            self.gui.torch_mem_gauge.setValue(0)
        else:
            self.gui.gpu_mem_gauge.setFormat('N/A')
            self.gui.gpu_mem_gauge.setValue(0)
            self.gui.torch_mem_gauge.setFormat('N/A')
            self.gui.torch_mem_gauge.setValue(0)

    def on_gpu_timer(self):
        self.update_gpu_gauges()

    def update_memory_gauges(self):
        try:
            curr_perm_tokens = self.processor.memory.work_mem.perm_size(0)
            self.gui.perm_mem_gauge.setFormat(f'{curr_perm_tokens} / {curr_perm_tokens}')
            self.gui.perm_mem_gauge.setValue(100)

            max_work_tokens = self.processor.memory.max_work_tokens
            max_long_tokens = self.processor.memory.max_long_tokens

            curr_work_tokens = self.processor.memory.work_mem.non_perm_size(0)
            curr_long_tokens = self.processor.memory.long_mem.non_perm_size(0)

            self.gui.work_mem_gauge.setFormat(f'{curr_work_tokens} / {max_work_tokens}')
            self.gui.work_mem_gauge.setValue(round(curr_work_tokens / max_work_tokens * 100))

            self.gui.long_mem_gauge.setFormat(f'{curr_long_tokens} / {max_long_tokens}')
            self.gui.long_mem_gauge.setValue(round(curr_long_tokens / max_long_tokens * 100))

        except AttributeError as e:
            self.gui.work_mem_gauge.setFormat('Unknown')
            self.gui.long_mem_gauge.setFormat('Unknown')
            self.gui.work_mem_gauge.setValue(0)
            self.gui.long_mem_gauge.setValue(0)

    def on_work_min_change(self):
        if self.initialized:
            self.gui.work_mem_min.setValue(
                min(self.gui.work_mem_min.value(),
                    self.gui.work_mem_max.value() - 1))
            self.update_config()

    def on_work_max_change(self):
        if self.initialized:
            self.gui.work_mem_max.setValue(
                max(self.gui.work_mem_max.value(),
                    self.gui.work_mem_min.value() + 1))
            self.update_config()

    def update_config(self):
        if self.initialized:
            with open_dict(self.cfg):
                self.cfg.long_term['min_mem_frames'] = self.gui.work_mem_min.value()
                self.cfg.long_term['max_mem_frames'] = self.gui.work_mem_max.value()
                self.cfg.long_term['max_num_tokens'] = self.gui.long_mem_max.value()
                self.cfg['mem_every'] = self.gui.mem_every_box.value()

            self.processor.update_config(self.cfg)

    def on_clear_memory(self):
        self.processor.clear_memory()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        elif 'mps' in self.device:
            mps.empty_cache()
        self.processor.update_config(self.cfg)
        self.update_gpu_gauges()
        self.update_memory_gauges()

    def on_clear_non_permanent_memory(self):
        self.processor.clear_non_permanent_memory()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        elif 'mps' in self.device:
            mps.empty_cache()
        self.processor.update_config(self.cfg)
        self.update_gpu_gauges()
        self.update_memory_gauges()

    def load_workspace(self, video: str = None, images: str = None, workspace: str = None):
        """Switch to another video/workspace in place -- no process restart.

        Called from the loader dialog (gui.open_loader). Exactly one of `video`, `images`
        or `workspace` is given. The heavy networks (self.cutie / self.click_ctrl) do not
        depend on the workspace, so they are reused and switching stays fast; only the
        ResourceManager, the InferenceCore (fresh memory) and the per-frame/tool state are
        rebuilt. Workspace derivation matches startup: `workspace=None` lets ResourceManager
        derive ./workspace/<basename>, re-using that folder if it already has frames (so
        opening an already-imported video continues its annotation instead of re-decoding).
        """
        if self.propagating:
            self.gui.text('Stop propagation before loading another video.')
            return

        # flush the frame being edited and the arches to the *current* workspace first
        try:
            if self.curr_frame_dirty:
                self.save_current_mask()
            retzius_arch.save(self.res_man.workspace, self.arch_by_frame)
        except Exception as e:
            self.gui.text(f'Warning: could not fully save current work: {e}')

        self.gui.text('Loading... reading/extracting frames, please wait.')
        self.gui.process_events()

        # remember the current source so a failed load can be fully rolled back
        prev = {k: self.cfg[k] for k in ('video', 'images', 'workspace')}
        with open_dict(self.cfg):
            self.cfg['video'] = video
            self.cfg['images'] = images
            self.cfg['workspace'] = workspace

        # rebuild the ResourceManager; keep the old session intact if this fails
        old_res_man = self.res_man
        try:
            self.res_man = ResourceManager(self.cfg)
        except Exception as e:
            self.res_man = old_res_man
            with open_dict(self.cfg):
                for k, v in prev.items():
                    self.cfg[k] = v
            self.gui.text(f'Failed to load: {e}')
            return
        old_res_man.shutdown()   # flush + stop its save threads (they would otherwise leak)
        del old_res_man

        # fresh inference memory for the new clip
        self.processor = InferenceCore(self.cutie, self.cfg)
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        elif 'mps' in self.device:
            mps.empty_cache()

        self._reset_session_state()

        # re-point the reused GUI widgets at the new workspace and redraw
        self.gui.rebind_workspace(self.h, self.w, self.T, self.res_man.workspace)
        self.gui.set_mode_index(MODE_INDEX[self._mode()])
        self.gui.set_scale_class_id(self.scale_class)
        self.gui.set_polygon_erase(self.polygon_erase)
        self.load_current_image_mask()
        self.show_current_frame()

        # sync memory-parameter boxes + gauges to the new processor
        self.gui.work_mem_min.setValue(self.processor.memory.min_mem_frames)
        self.gui.work_mem_max.setValue(self.processor.memory.max_mem_frames)
        self.gui.long_mem_max.setValue(self.processor.memory.max_long_tokens)
        self.gui.mem_every_box.setValue(self.processor.mem_every)
        self.update_config()
        self.update_memory_gauges()
        self.update_gpu_gauges()

        # the overlay layer was padded to the previous canvas size; re-fit it
        self._try_load_layer('./docs/uiuc.png')

        self.gui.set_object_color(self.curr_object)
        self.gui.set_current_object_id(self.curr_object)
        self.gui.text(f'Loaded workspace: {self.res_man.workspace} ({self.length} frames).')

    def _reset_session_state(self):
        """Reset all per-frame / per-tool state for a freshly (re)loaded workspace.
        Mirrors the per-workspace part of __init__; the networks and the GUI widget tree
        are deliberately left untouched (they are reused across loads)."""
        self.length = self.res_man.length
        self.interaction = None
        self.interaction_type = 'Click'
        self.curr_ti = 0
        self.curr_object = 1
        self.propagating = False
        self.propagate_direction = 'none'
        self.last_ex = self.last_ey = 0

        self.curr_frame_dirty = False
        self.curr_image_np = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.curr_image_torch = None
        self.curr_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        self.curr_prob = torch.zeros((self.num_objects + 1, self.h, self.w),
                                     dtype=torch.float).to(self.device)
        self.curr_prob[0] = 1

        self.vis_image = None
        self.curr_depth_map = None
        self._depth_cache = {}

        self.interacted_prob = None
        # re-fit against the new canvas size on the next _try_load_layer
        self.overlay_layer = None
        self.overlay_layer_torch = None
        self.vis_target_objects = list(range(1, self.num_objects + 1))

        # Retzius arch tool state -- load the new workspace's own arches
        self.arch_mode = False
        self.arch_pending = []
        self.arch_drag = None
        self.arch_tracking = False
        self._arch_track_dir = None
        self._syncing_arch_slider = False
        self._arch_tracker = None
        self._arch_current = None
        self._arch_us = None
        self._arch_holds = 0
        self.arch_by_frame = retzius_arch.load(self.res_man.workspace)

        # Scale-reference tool state -- load the new workspace's own references
        self.scale_mode = False
        self.scale_class = 1
        self.scale_pending = []
        self.scale_drag = None
        self.scale_tracking = False
        self._scale_track_dir = None
        self._syncing_scale = False
        self._scale_states = []
        # scale_edit_points is deliberately NOT reset: it is a UI preference living in the
        # (reused) checkbox, and resetting it here would desync the two
        (self.scale_by_frame, self.scale_mm, self.arm_by_frame, self.arm_seed,
         self.arm_calib, self.arm_track) = scale_objects.load(self.res_man.workspace)
        self._arm_state = None
        self._arm_summary = None
        self._arm_reconcile()

        # GUI-bar boxes -- the new workspace's own, never the last clip's
        self.bar_pending = None
        self.bar_anchor = None
        self._bar_states = []
        self.bars_by_frame = gui_bar.load(self.res_man.workspace)
        # bar_mode is deliberately NOT reset here: like scale_edit_points it lives in the
        # (reused) COVER GUI checkbox, and resetting it would desync the two

        # polygon tool state
        self.polygon_points = []
        self.hover_first_point = False
        self.in_polygon_mode = False
        self.polygon_erase = False
        self._polygon_undo = None

        # measurement tool state
        self.measure_mode = None
        self.measure_points = []
        self.anchor_px = None
        self.anchor_mm = None
        self._anchor_target_mm = None
        self.live_sul = False
        self.tracker = None

    def on_import_mask(self):
        file_name = self.gui.open_file('Mask')
        if len(file_name) == 0:
            return

        mask = self.res_man.import_mask(file_name, size=(self.h, self.w))

        shape_condition = ((len(mask.shape) == 2) and (mask.shape[-1] == self.w)
                           and (mask.shape[-2] == self.h))

        object_condition = (mask.max() <= self.num_objects)

        if not shape_condition:
            self.gui.text(f'Expected ({self.h}, {self.w}). Got {mask.shape} instead.')
        elif not object_condition:
            self.gui.text(f'Expected {self.num_objects} objects. Got {mask.max()} objects instead.')
        else:
            self.gui.text(f'Mask file {file_name} loaded.')
            self.curr_image_torch = self.curr_prob = None
            self.curr_mask = mask
            self.show_current_frame()
            self.save_current_mask()

    def on_import_layer(self):
        file_name = self.gui.open_file('Layer')
        if len(file_name) == 0:
            return

        self._try_load_layer(file_name)

    def _try_load_layer(self, file_name):
        try:
            layer = self.res_man.import_layer(file_name, size=(self.h, self.w))

            self.gui.text(f'Layer file {file_name} loaded.')
            self.overlay_layer = layer
            self.overlay_layer_torch = torch.from_numpy(layer).float().to(self.device) / 255
            self.show_current_frame()
        except FileNotFoundError:
            self.gui.text(f'{file_name} not found.')

    def on_save_soft_mask_toggle(self):
        self.save_soft_mask = self.gui.save_soft_mask_checkbox.isChecked()

    def on_mouse_motion_xy(self, x: int, y: int, ux: float = None, uy: float = None):
        self.last_ex, self.last_ey = x, y

        # Dragging an arch handle: sides follow the clamped cursor, the tip projects the
        # raw cursor onto the mid-line (so it can be pulled past the image border)
        if self.arch_drag is not None:
            retzius_arch.move_handle(*self.arch_drag, x, y,
                                     x if ux is None else ux, y if uy is None else uy)
            self.arch_drag[0].source = 'manual'   # an adjusted frame is a propagation keyframe
            self._draw_arch_overlay()
            return

        # Dragging a scale reference: an end takes the cursor, a tracking point is
        # projected back onto the line (clamped coords -- references stay in the image)
        if self.bar_pending is not None:
            self.bar_pending = gui_bar.Bar(*self.bar_anchor, x, y)
            self._draw_scale_overlay()
            return

        if self.scale_drag is not None:
            scale_objects.move_handle(*self.scale_drag, x, y)
            self.scale_drag[0].source = 'manual'  # an adjusted frame is a TRACK keyframe
            self.scale_drag[0].conf = 1.0
            self._draw_scale_overlay()
            return

        # Dragging one of the robot arm's four points. Everything follows from them, so the
        # centerline, the scale line and the reading all move with the cursor -- you watch
        # the number settle rather than clicking and hoping. The two entry points take the
        # frame size because they only slide ALONG the image edge they are on.
        if self.arm_drag:
            ann = self.arm_by_frame.get(self.curr_ti)
            if ann is None:
                self.arm_drag = None
            else:
                ann.move(self.arm_drag, x, y, (self.h, self.w))
                self._draw_scale_overlay()
            return

        # Check if polygon is being drawn and at least one point exists
        if self.polygon_points:
            # Check distance to first point
            first_pt = self.polygon_points[0]
            dist = ((x - first_pt[0])**2 + (y - first_pt[1])**2)**0.5
            was_hovering = self.hover_first_point
            self.hover_first_point = dist <= self.hover_threshold

            # If hover state changed, update the canvas
            if self.hover_first_point != was_hovering:
                self.compose_polygon_overlay()
                self.update_canvas()

    def on_toggle_vis_mode(self):
        vis_modes = ['image', 'mask', 'mask overlay']
        try:
            next_index = (vis_modes.index(self.vis_mode) + 1) % len(vis_modes)
            self.vis_mode = vis_modes[next_index]
        except ValueError:
            self.vis_mode = 'mask overlay'
        print(f'Visualization mode changed to {self.vis_mode}')

        # Update the dropdown menu to show the current mode
        self.gui.combo.setCurrentText(self.vis_mode)
        self.show_current_frame()

    @property
    def h(self) -> int:
        return self.res_man.h

    @property
    def w(self) -> int:
        return self.res_man.w

    @property
    def T(self) -> int:
        return self.res_man.T
