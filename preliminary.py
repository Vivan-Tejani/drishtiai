#!/usr/bin/env python3
"""
DRISHTI-PS2 — OFFLINE VIDEO SEGMENTATION & ROI DETECTION (Motion Estimation)
KAGGLE EDITION — auto path resolution + zone-wise motion + heatmap + timeline + clips
                  + PER-PERSON PRIVATE ZONE INTRUSION DETECTION
                  + AMBIENT MOVEMENT FILTER + GLOBAL DISTURBANCE DETECTOR

=== SINGLE-FILE MERGED EDITION ===
This file merges three originally separate scripts into one:
  1. pipeline_final.py               -> this file's core pipeline (unchanged logic)
  2. ambient_movement_filter.py      -> inlined below, see "AMBIENT MOVEMENT FILTER"
  3. global_disturbance_detector.py  -> inlined below, see "GLOBAL DISTURBANCE DETECTOR"

Nothing needs to sit "next to" this script anymore — it is a single, complete,
drop-in-anywhere pipeline. The USE_AMBIENT_FILTER / USE_GLOBAL_DISTURBANCE_DETECTOR
feature flags in Settings still work exactly as before (toggle either subsystem
off without touching code); only the old try/except-ImportError sibling-file
loading is gone, since both modules are now permanent parts of this one file.

A note on the merge itself: the three original files each independently
declared a module-level variable named `settings` (AmbientFilterSettings(),
GlobalDisturbanceSettings(), and the pipeline's own Settings()). That's fine
when they're three separate files/namespaces, but concatenated into one file
it's a silent collision — the last one wins and the earlier two classes end
up trying to read attributes off the wrong config object. Fixed by renaming
the two newly-inlined ones to `ambient_filter_settings` and
`global_disturbance_settings`; the pipeline's own primary `settings` object
(used everywhere else in this file) keeps its original name unchanged.

--------------------------------------------------------------------------
KNOWN-ISSUE FIX APPLIED (race condition between the two sustained-frame gates)
--------------------------------------------------------------------------
Testing (a teacher-walking synthetic scenario) surfaced a real timing bug:
AmbientMovementFilter needs its own run of consecutive "candidate ambient"
frames (SUSTAINED_CANDIDATE_FRAMES) before it will suppress a zone, but
before it can even call a frame "candidate" it first has to accumulate
MIN_FRAMES_FOR_TRANSLATION_CHECK worth of centroid history just to know
whether the blob is translating at all. Combined, that was more frames than
ZoneEventTracker.SUSTAINED_FRAMES needs to independently open a real ROI
event. So on a fast walk-through (teacher only dwells ~3-4 analyzed frames
per zone), ZoneEventTracker would already have opened an event by the time
the ambient filter finished convincing itself the blob was ambient — one
test run logged 25 correct "sustained ambient" verdicts, yet 9 spurious ROI
events still got created, because the verdict kept arriving one zone-dwell
too late.

FIX: tightened MIN_FRAMES_FOR_TRANSLATION_CHECK (3 -> 2) and
SUSTAINED_CANDIDATE_FRAMES (3 -> 2) in AmbientFilterSettings below, so the
ambient verdict is ready in the SAME frame ZoneEventTracker's own gate would
otherwise fire, not after it. This keeps the "don't react to one twitch"
requirement intact (still needs 2 consecutive frames of agreement, not 1) —
it just closes a gap against a threshold (SUSTAINED_FRAMES=3) that was never
matched to it in the first place. If real footage still shows a fast walker
slipping through, the next lever to pull is the same one, taken further
(e.g. SUSTAINED_CANDIDATE_FRAMES=1), or increasing ZoneEventTracker's own
SUSTAINED_FRAMES slightly instead — both are one-line settings changes.
--------------------------------------------------------------------------

=== DESIGN PHILOSOPHY: "HUMAN INVIGILATOR" LOGIC ===
A human invigilator walking into an exam hall doesn't track 33 skeletal keypoints
per student. They do something much simpler and more robust:

  1. They mentally divide the room into zones (rows of benches / seat blocks) and
     develop a "feel" for how much normal fidgeting/motion is typical in each zone
     during the first few quiet minutes (BASELINE CALIBRATION).
  2. Within a zone, their eye is drawn to a specific BLOB of movement — a hand
     reaching down, a head turning — not an abstract global motion score.
  3. They don't react to a single twitch. A student adjusting their seat for
     half a second is ignored. Something that PERSISTS for a few seconds is what
     draws sustained attention (SUSTAINED DEVIATION = EVENT).
  4. When something catches their eye, they don't just note the timestamp —
     they keep watching that spot for a bit before and after (EVENT CLIP with
     pre/post buffer), and they remember it as "that thing that happened in
     zone 3 around 14:22" (ZONE-TAGGED EVENT LOG).
  5. If they spot an actual object (phone, chit of paper), that's independent
     corroborating evidence layered on top of the motion — it makes the flagged
     event more credible, but isn't required to raise an eyebrow in the first
     place.
  6. They also notice when a student's hand strays into a NEIGHBOR's space —
     not because they're tracking identity, but because they know roughly
     where each student's own "personal space" is (PRIVATE ZONE INTRUSION).
  7. They also filter out their OWN presence in the room — them walking the
     aisles doesn't count as suspicious motion (AMBIENT MOVEMENT FILTER) — and
     they recognize when the whole room reacts to something at once, like a
     door slamming, as one shared moment rather than N separate incidents
     (GLOBAL DISTURBANCE DETECTOR).

This script implements exactly that pipeline: Kaggle path auto-resolution,
dataclass configs, logging style, frame sampling/ffmpeg-fallback, contact
sheets, flagged CSV/JSON reporting, frame-to-frame motion estimation -> spatial
ROI detection, zone-wise + blob-wise activity, event-based video SEGMENTATION
with real exported clips, motion heatmaps (percentile-clipped), an activity
timeline, optional object detection as corroborating evidence, a searchable
event log (CSV + JSON), PER-PERSON PRIVATE ZONE tracking, an AMBIENT MOVEMENT
FILTER, and a GLOBAL DISTURBANCE DETECTOR — all offline, no student identity,
motion-estimation based per PS2's own framing (not the skeletal-keypoint
approach that's PS1's ask).
"""

from __future__ import annotations
import os, sys, json, time, math, cv2, numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple, Generator
from collections import defaultdict, deque
from dataclasses import dataclass, field
import logging, argparse, warnings, shutil, subprocess, tempfile, hashlib, csv
warnings.filterwarnings("ignore")

# ========== OPTIONAL DEPENDENCIES (graceful fallback) ==========
try:
    from ultralytics import YOLO as UltralyticsYOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ========== KAGGLE PATH AUTO-RESOLUTION ==========
def _is_kaggle() -> bool:
    return Path("/kaggle/input").exists()

def _find_file_under_kaggle_input(filename: str) -> Optional[Path]:
    if not _is_kaggle():
        return None
    root = Path("/kaggle/input")
    patterns = [f"*/{filename}", f"*/*/{filename}", f"*/*/*/{filename}"]
    for pat in patterns:
        matches = sorted(root.glob(pat))
        if matches:
            return matches[0]
    return None

def resolve_model_path(filename: str, local_default: Path) -> Path:
    if _is_kaggle():
        found = _find_file_under_kaggle_input(filename)
        if found is not None:
            return found
    return local_default

def resolve_video_path(explicit: Optional[str]) -> Path:
    if explicit and explicit.lower() != "auto":
        return Path(explicit)
    if _is_kaggle():
        root = Path("/kaggle/input")
        for ext in ("*.mov", "*.mp4", "*.avi", "*.MOV", "*.MP4"):
            for depth_pat in (f"*/{ext}", f"*/*/{ext}", f"*/*/*/{ext}"):
                matches = sorted(root.glob(depth_pat))
                if matches:
                    return matches[0]
    return Path("./data/cheating_video (1).mov")

def resolve_output_dir() -> Path:
    if _is_kaggle():
        out = Path("/kaggle/working/output_ps2")
    else:
        out = Path("./data/output_ps2")
    out.mkdir(parents=True, exist_ok=True)
    return out


# ==== QUICK TEST VIDEO OVERRIDE — set this to a path (e.g. "./data/new_test.mp4") to test a new video without touching anything else below; leave as None to keep normal --video/auto behavior ====
TEST_VIDEO_OVERRIDE: Optional[str] = "/kaggle/input/datasets/divyashkigf/videoo/recording.mov"


# Shared bbox type alias, used by the ambient filter, the private-zone
# tracker, and the global disturbance detector below.
Bbox = Tuple[int, int, int, int]  # x1, y1, x2, y2


# ========================================================================================
# ========== AMBIENT MOVEMENT FILTER (inlined from ambient_movement_filter.py) ==========
# ========================================================================================
"""
Distinguishes "person walking/standing" motion (teacher doing rounds,
invigilator walking an aisle) from "localized suspicious gesture" motion
(seated student's hand/head movement).

WHY THIS EXISTS
----------------
The core ROI pipeline flags a zone the moment its motion z-score is sustained
above threshold for a few frames. A teacher walking through ANY zone produces
exactly that signature — sustained, above-baseline motion — so without this
filter she gets flagged as a suspicious event as often as an actual cheating
gesture would.

Excluding a fixed zone/corridor does NOT work here because the teacher does
not walk a fixed path — she can stand anywhere, cut through any row, stop
next to any bench. A spatial exclusion zone would need to be redrawn for
every seating arrangement and would still miss her the moment she deviates
from it, which fails PS2's own robustness requirement ("robust to variations
in seating arrangements...").

So instead of asking WHERE the motion is, this module asks WHAT SHAPE the
motion has and HOW IT MOVES OVER TIME — signals that stay valid regardless of
room layout:

  1. Blob aspect ratio  — a walking/standing adult's silhouette is tall and
     roughly full-body height; a seated student's hand/head gesture is a
     small, compact, wide-ish blob.
  2. Blob height fraction of the zone — a full body spans most of the zone's
     vertical extent; a gesture only occupies a small fraction.
  3. Cross-frame centroid displacement — a walking person's motion blob
     TRANSLATES continuously frame to frame (she's covering ground); a
     seated gesture stays localized inside roughly the same spot over its
     lifetime.
  4. (Optional) overlap with a detected "person" bbox that is NOT anchored
     inside any known seated private zone — reuses PrivateZoneTracker's
     person detections at zero extra inference cost.

None of these depend on which physical zone the motion occurred in, so this
generalizes across seating layouts and camera angles the way a fixed
exclusion zone cannot.

DESIGN NOTE — signals are combined, not used alone. Any single signal
misfires sometimes (e.g. a student stretching upward can transiently look
"tall"). Requiring a majority of signals to agree, PLUS requiring the
tall/translating signature to be true for its own sustained run of frames
(not just once), keeps false negatives (letting a real ambient walker
through) unlikely while keeping false positives (wrongly suppressing a real
cheating gesture as "ambient") rare, because a cheating gesture's blob
essentially never translates across zones the way a walking body does.
"""


@dataclass
class AmbientFilterSettings:
    # --- blob shape thresholds ---
    TALL_ASPECT_RATIO: float = 1.4        # height/width >= this -> "tall" blob
    MIN_HEIGHT_FRACTION_OF_ZONE: float = 0.55  # blob height >= this fraction of zone height -> "full-body-ish"

    # --- centroid displacement (the strongest signal) ---
    DISPLACEMENT_HISTORY_LEN: int = 6      # how many recent centroids to keep per zone
    # FIX (was 18.0): 18px/frame assumed a brisk walking pace, but the pipeline
    # samples at only settings.FPS=5 frames/sec, so each analyzed frame is a
    # 200ms real-time gap -- a person walking SLOWLY, or briefly pausing
    # mid-stride (which real footage does constantly: teacher stops to peer
    # at a paper, half-turns, resumes), can easily average well under
    # 18px/frame between analyzed samples without being any less "a person
    # moving through the room" than a brisk walker. At 18px/frame a real
    # invigilator moving unhurriedly was never being recognized as
    # translating at all, so the movement-evidence gate below never opened
    # for them regardless of how clearly tall/full-height their blob was.
    # Lowered to a value that still rejects genuinely stationary jitter
    # (hand/head gesture centroid noise is typically a few px/frame at this
    # resolution) while accepting slow, real, sustained locomotion.
    MIN_TRANSLATION_PX_PER_FRAME: float = 8.0
    # FIX (was 3): needs to be small enough that the translating signal is
    # available in time to feed SUSTAINED_CANDIDATE_FRAMES before
    # ZoneEventTracker.SUSTAINED_FRAMES (3) opens an event on its own -- see
    # the "KNOWN-ISSUE FIX" note at the top of this file.
    MIN_FRAMES_FOR_TRANSLATION_CHECK: int = 2

    # --- stationary-but-standing fallback (see STATIONARY_SHAPE_STREAK note below) ---
    # An invigilator who has stopped walking and is standing still for a
    # moment (paused beside a bench, looking at a paper) produces a blob
    # that is tall/full-height but NOT translating -- translating requires
    # motion, and someone standing still has none to measure. Without this
    # fallback such a person can never satisfy has_movement_evidence (which
    # needs translating OR person_roaming) no matter how unambiguously
    # body-shaped their blob is, because both of those signals specifically
    # require detected motion/roaming, which a standing person doesn't
    # generate. This many CONSECUTIVE frames of both shape signals firing
    # together is required before "just stands there, tall" is accepted as
    # movement evidence on its own -- a brief 1-2 frame stretch (a student
    # stretching upward) will not reach this streak length, but a person who
    # is genuinely standing in the aisle for a real stretch of time will.
    STATIONARY_SHAPE_STREAK_FRAMES: int = 4

    # --- voting ---
    # An observation needs at least this many of the 4 signals to agree
    # before it's even a candidate "ambient" frame.
    MIN_SIGNALS_FOR_CANDIDATE: int = 2
    # GATING RULE: shape alone (tall + full-height) is NOT enough on its own,
    # frame-by-frame -- a student stretching upward is also tall and spans
    # most of the zone height for a few frames, without walking anywhere. At
    # least one of the three "this thing is actually a body moving through
    # or occupying space" signals (translating, overlapping a roaming person
    # box, OR a sustained multi-frame run of pure shape -- see
    # STATIONARY_SHAPE_STREAK_FRAMES above) must ALSO be true, or the frame
    # never counts as an ambient candidate no matter how many shape signals
    # fire in isolation.

    # FIX (was 3): An event is only ever reclassified as ambient if the
    # CANDIDATE signal itself was sustained for this many consecutive
    # analyzed frames within the event's lifetime -- mirrors the same "don't
    # react to one twitch" philosophy as ZoneEventTracker, applied here to
    # avoid one lucky-shaped gesture frame flipping a genuine event to
    # "ambient". Lowered from 3 to 2 to match timing with
    # ZoneEventTracker.SUSTAINED_FRAMES (3) -- see "KNOWN-ISSUE FIX" note at
    # the top of this file for why 3-vs-3 let a fast-walking blob's ROI event
    # open before the ambient verdict was ready.
    SUSTAINED_CANDIDATE_FRAMES: int = 2

    # --- optional person-bbox cross-check ---
    USE_PERSON_BBOX_CROSSCHECK: bool = True
    STANDING_ASPECT_RATIO: float = 1.6     # person bbox height/width >= this -> standing/walking adult
    PERSON_ZONE_OVERLAP_IOU_MAX: float = 0.15  # if blob overlaps a person bbox that ITSELF barely overlaps
                                                 # any known seated private zone, that person is "roaming"


ambient_filter_settings = AmbientFilterSettings()


def _iou(a: Bbox, b: Bbox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def _bbox_centroid(b: Bbox) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_hw(b: Bbox) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (max(1.0, y2 - y1), max(1.0, x2 - x1))  # height, width


@dataclass
class _ZoneMotionHistory:
    """Rolling per-zone memory of recent blob centroids, purely for the
    translation check. Kept tiny (deque maxlen) so this is O(1) memory."""
    centroids: deque = field(default_factory=lambda: deque(maxlen=ambient_filter_settings.DISPLACEMENT_HISTORY_LEN))
    candidate_streak: int = 0
    # Consecutive-frame counter for "both shape signals fired, regardless of
    # translation" -- feeds the standing-still fallback signal (see
    # _signal_stationary_but_shaped below). Kept separate from
    # candidate_streak, which only advances once a frame is ALREADY a full
    # candidate -- this one has to advance independently so it can itself
    # become the evidence that promotes a frame to candidate in the first
    # place, for a person who has stopped walking.
    stationary_shape_streak: int = 0


class AmbientMovementFilter:
    """
    Stateful filter, one instance shared across the whole video (like
    ZoneEventTracker). Call `classify()` once per zone per analyzed frame,
    right where the pipeline currently only computes `is_deviant`.

    It does NOT decide whether a zone is deviant (z-score) -- that stays the
    job of ZoneBaselineEngine/ZoneEventTracker. It only answers a narrower
    question: "given that this blob IS a motion outlier, does it look like a
    walking/standing person passing through, as opposed to a seated person's
    localized gesture?"
    """

    def __init__(self, cfg: AmbientFilterSettings = ambient_filter_settings):
        self.cfg = cfg
        self._history: Dict[int, _ZoneMotionHistory] = {}

    def reset_zone(self, zone_id: int):
        """Call when a zone's event closes, so stale centroid history from
        one event doesn't bleed into judging the next unrelated one."""
        self._history.pop(zone_id, None)

    # ---------- individual signals ----------

    def _signal_tall_aspect(self, blob_bbox: Bbox) -> bool:
        h, w = _bbox_hw(blob_bbox)
        return (h / w) >= self.cfg.TALL_ASPECT_RATIO

    def _signal_full_height(self, blob_bbox: Bbox, zone_bbox: Bbox) -> bool:
        h, _ = _bbox_hw(blob_bbox)
        zone_h = max(1.0, zone_bbox[3] - zone_bbox[1])
        return (h / zone_h) >= self.cfg.MIN_HEIGHT_FRACTION_OF_ZONE

    def _signal_translating(self, zone_id: int, blob_bbox: Bbox) -> bool:
        hist = self._history.setdefault(zone_id, _ZoneMotionHistory())
        cx, cy = _bbox_centroid(blob_bbox)
        hist.centroids.append((cx, cy))
        if len(hist.centroids) < self.cfg.MIN_FRAMES_FOR_TRANSLATION_CHECK:
            return False
        pts = list(hist.centroids)
        total_disp = 0.0
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            total_disp += math.hypot(x1 - x0, y1 - y0)
        avg_disp = total_disp / max(1, len(pts) - 1)
        return avg_disp >= self.cfg.MIN_TRANSLATION_PX_PER_FRAME

    def _signal_person_roaming(self, blob_bbox: Bbox,
                                person_bboxes: Optional[List[Bbox]],
                                seated_private_zones: Optional[List[Bbox]]) -> bool:
        if not self.cfg.USE_PERSON_BBOX_CROSSCHECK or not person_bboxes:
            return False
        # find a person bbox that overlaps this motion blob
        best_person, best_iou = None, 0.0
        for pbox in person_bboxes:
            score = _iou(blob_bbox, pbox)
            if score > best_iou:
                best_iou, best_person = score, pbox
        if best_person is None or best_iou <= 0.0:
            return False
        ph, pw = _bbox_hw(best_person)
        is_standing_shape = (ph / pw) >= self.cfg.STANDING_ASPECT_RATIO
        if not is_standing_shape:
            return False
        if not seated_private_zones:
            # no seated-zone info available -> can't confirm "roaming",
            # standing shape alone isn't enough for this particular signal
            return False
        max_zone_overlap = max((_iou(best_person, z) for z in seated_private_zones), default=0.0)
        return max_zone_overlap <= self.cfg.PERSON_ZONE_OVERLAP_IOU_MAX

    def _signal_stationary_but_shaped(self, zone_id: int, tall: bool, full_height: bool) -> bool:
        """
        Movement-evidence fallback for a person who is standing still (or
        moving too slowly for _signal_translating to catch): if BOTH shape
        signals (tall_aspect, full_height) have fired together for
        STATIONARY_SHAPE_STREAK_FRAMES consecutive analyzed frames, that
        sustained run is itself treated as movement evidence, without
        requiring any actual displacement or a person-detection cross-check.

        This is deliberately a HIGH BAR (several consecutive frames, not
        one) specifically because it does not require translation -- a
        single stretching/reaching gesture can transiently look tall and
        full-height for a frame or two, but sustaining BOTH shape signals
        for a real run of consecutive frames is a much narrower coincidence,
        and is exactly the signature of a full-body adult silhouette
        standing in the room rather than a seated student's momentary
        gesture.
        """
        hist = self._history.setdefault(zone_id, _ZoneMotionHistory())
        if tall and full_height:
            hist.stationary_shape_streak += 1
        else:
            hist.stationary_shape_streak = 0
        return hist.stationary_shape_streak >= self.cfg.STATIONARY_SHAPE_STREAK_FRAMES

    # ---------- combined decision ----------

    def classify(self, zone_id: int, blob_bbox: Bbox, zone_bbox: Bbox,
                 person_bboxes: Optional[List[Bbox]] = None,
                 seated_private_zones: Optional[List[Bbox]] = None) -> Dict[str, Any]:
        """
        Returns a dict:
          {
            'signals': {...bool per signal...},
            'signal_count': int,
            'is_candidate_ambient': bool,   # this single frame looks ambient
            'is_sustained_ambient': bool,   # candidate AND persisted across
                                             # SUSTAINED_CANDIDATE_FRAMES for this zone
          }
        Caller decides what to DO with is_sustained_ambient (e.g. suppress
        raising a new ROI event, or tag an existing one as low-confidence).
        """
        tall = self._signal_tall_aspect(blob_bbox)
        full_height = self._signal_full_height(blob_bbox, zone_bbox)
        signals = {
            'tall_aspect': tall,
            'full_height': full_height,
            'translating': self._signal_translating(zone_id, blob_bbox),
            'person_roaming': self._signal_person_roaming(blob_bbox, person_bboxes, seated_private_zones),
            'stationary_but_shaped': self._signal_stationary_but_shaped(zone_id, tall, full_height),
        }
        signal_count = sum(1 for v in signals.values() if v)
        has_movement_evidence = (signals['translating'] or signals['person_roaming']
                                  or signals['stationary_but_shaped'])
        is_candidate = (signal_count >= self.cfg.MIN_SIGNALS_FOR_CANDIDATE) and has_movement_evidence

        hist = self._history.setdefault(zone_id, _ZoneMotionHistory())
        if is_candidate:
            hist.candidate_streak += 1
        else:
            hist.candidate_streak = 0

        is_sustained = hist.candidate_streak >= self.cfg.SUSTAINED_CANDIDATE_FRAMES

        return {
            'signals': signals,
            'signal_count': signal_count,
            'is_candidate_ambient': is_candidate,
            'is_sustained_ambient': is_sustained,
        }


# ========================================================================================
# ===== GLOBAL DISTURBANCE DETECTOR (inlined from global_disturbance_detector.py) ========
# ========================================================================================
"""
Distinguishes "the whole room reacted to something" (an announcement, a door
slamming, an invigilator saying something loudly, a phone ringing) from "one
student is doing something suspicious in one zone".

WHY THIS EXISTS
----------------
ZoneEventTracker evaluates each zone's deviation completely independently --
there is no cross-zone awareness at all. If something startles or distracts
the entire room at once, every zone's motion spikes together, and because the
spike is genuinely sustained (everyone keeps looking toward the door / toward
the noise for a few seconds), it clears SUSTAINED_FRAMES in every zone
simultaneously. The pipeline would then raise N separate "suspicious ROI
events" -- one per zone -- for what was actually a single, benign, room-wide
moment. That is noisy for the investigator (12 clips to review for one
non-event) and undermines trust in the flagged-event log.

WHAT COUNTS AS "GLOBAL"
------------------------
A single student cheating does not make neighbouring zones move too -- that
motion is spatially localized by definition (that's the whole premise the
ROI/zone pipeline is built on). So the signal here is simple and doesn't need
any new sensor or model: if a LARGE FRACTION of ALL zones are independently
deviant in the same analyzed frame, that concurrent breadth is itself the
tell. No single zone's motion pattern needs to look different -- it's the
count across zones, at the same timestamp, that flips the interpretation.

DESIGN CHOICE -- frame-level vote, not just "N events started this second".
We check concurrent deviation directly off the z-scores ZoneBaselineEngine
already computes every frame (before ZoneEventTracker's own per-zone
sustained-frame gating even runs), because waiting for each zone's OWN
SUSTAINED_FRAMES gate to pass first would just re-derive 12 separate "events"
and only THEN notice they overlapped in time -- by which point 12 clips/log
rows already exist. Watching concurrent deviation directly, frame by frame,
catches it before any per-zone event is even opened.

This still requires the elevated deviation to PERSIST across a few
consecutive frames (same "don't react to one twitch" principle used
everywhere else in this pipeline) before declaring a global disturbance -- a
single frame where several zones happen to tick over threshold at once
(camera auto-exposure hiccup, a truck passing outside a window) shouldn't be
enough either.

IMPORTANT -- SUPPRESSION IS PER-ZONE, PER-FRAME, NOT A BLANKET SWITCH. A
naive implementation would suppress EVERY zone for as long as `is_active` is
True. That silently breaks the case where a genuine, independent,
single-zone cheating event happens to coincide in time with a brief
multi-zone blip elsewhere. Use `should_suppress_zone(zone_id)` every frame:
it only returns True for zones that are co-occurring with the rest of the
disturbance IN THAT SAME FRAME. The instant a zone is the only one left
deviant, it stops being suppressed -- standing alone while everything else
has settled is exactly the signature of a real localized event, regardless
of whether the disturbance object has technically closed yet.
"""


@dataclass
class GlobalDisturbanceSettings:
    # --- concurrency threshold ---
    MIN_ZONE_FRACTION_DEVIANT: float = 0.5   # >=50% of all zones deviant in the same frame -> candidate
    MIN_ABSOLUTE_ZONES_DEVIANT: int = 4       # AND at least this many zones (guards tiny grids, e.g. 2x2)

    # --- sustain requirement (mirrors ZoneEventTracker.SUSTAINED_FRAMES) ---
    SUSTAINED_FRAMES: int = 3

    # --- cooldown so one long disturbance doesn't re-trigger repeatedly ---
    COOLDOWN_SECONDS: float = 4.0

    # --- gap tolerance for closing an active disturbance ---
    CLOSE_GAP_SECONDS: float = 1.5


global_disturbance_settings = GlobalDisturbanceSettings()


@dataclass
class GlobalDisturbanceEvent:
    start_timestamp: datetime
    end_timestamp: datetime
    peak_zone_count: int
    peak_zone_fraction: float
    total_zones: int
    involved_zone_ids: List[int]   # union of zones that were deviant at any point during the disturbance

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d['start_timestamp'] = self.start_timestamp.isoformat()
        d['end_timestamp'] = self.end_timestamp.isoformat()
        return d


class GlobalDisturbanceDetector:
    """
    Stateful, one instance per video (like ZoneEventTracker). Call
    `observe()` once per analyzed frame with the set of zone_ids that are
    independently deviant THIS frame (i.e. before per-zone SUSTAINED_FRAMES
    gating, straight off the z-score check) and the total zone count. It
    tracks its own sustained/cooldown state machine, mirroring
    ZoneEventTracker's shape so it drops into the same main loop the same
    way.
    """

    def __init__(self, cfg: GlobalDisturbanceSettings = global_disturbance_settings):
        self.cfg = cfg
        self._consecutive = 0
        self._active: Optional[Dict[str, Any]] = None
        self._last_end_timestamp: Optional[datetime] = None

    def _in_cooldown(self, timestamp: datetime) -> bool:
        if self._last_end_timestamp is None:
            return False
        return (timestamp - self._last_end_timestamp).total_seconds() < self.cfg.COOLDOWN_SECONDS

    def is_global_frame(self, deviant_zone_ids: List[int], total_zones: int) -> bool:
        """Pure check for this single frame -- no state mutation. Useful if
        the caller wants to know 'is this frame concurrent-deviant' without
        advancing the sustained/cooldown machinery (e.g. for logging)."""
        if total_zones <= 0:
            return False
        count = len(deviant_zone_ids)
        fraction = count / total_zones
        return (fraction >= self.cfg.MIN_ZONE_FRACTION_DEVIANT
                and count >= self.cfg.MIN_ABSOLUTE_ZONES_DEVIANT)

    def observe(self, deviant_zone_ids: List[int], total_zones: int,
                timestamp: datetime) -> Optional[str]:
        """
        Returns 'start' when a NEW sustained global disturbance begins, None
        otherwise. Caller should check `is_active` / `currently_active_info()`
        each frame regardless of return value, to know whether to keep
        suppressing per-zone event opening this frame.
        """
        is_concurrent = self.is_global_frame(deviant_zone_ids, total_zones)

        if is_concurrent and not self._in_cooldown(timestamp):
            self._consecutive += 1
            count = len(deviant_zone_ids)
            fraction = count / total_zones if total_zones else 0.0

            if self._active is not None:
                ev = self._active
                ev['last_seen'] = timestamp
                ev['involved_zones'] |= set(deviant_zone_ids)
                ev['current_frame_zones'] = set(deviant_zone_ids)   # THIS frame only, not the lifetime union
                if count > ev['peak_zone_count']:
                    ev['peak_zone_count'] = count
                    ev['peak_zone_fraction'] = fraction
                return None
            elif self._consecutive >= self.cfg.SUSTAINED_FRAMES:
                self._active = {
                    'start': timestamp,
                    'last_seen': timestamp,
                    'peak_zone_count': count,
                    'peak_zone_fraction': fraction,
                    'total_zones': total_zones,
                    'involved_zones': set(deviant_zone_ids),
                    'current_frame_zones': set(deviant_zone_ids),
                }
                return 'start'
        else:
            if self._active is not None:
                # disturbance is still technically "active" (hasn't hit its
                # own close-gap yet) but THIS frame wasn't concurrent -- so
                # no zone should be treated as co-occurring right now.
                self._active['current_frame_zones'] = set()
            self._consecutive = 0
        return None

    def should_suppress_zone(self, zone_id: int) -> bool:
        """
        THE FIX for the "genuine single-zone event coincides with a brief
        multi-zone blip" problem: suppression is decided PER ZONE, PER FRAME
        -- not as a single global on/off switch for the whole video.
        """
        if self._active is None:
            return False
        return zone_id in self._active.get('current_frame_zones', set())

    @property
    def is_active(self) -> bool:
        return self._active is not None

    def close_stale(self, timestamp: datetime) -> Optional[GlobalDisturbanceEvent]:
        """Call once per frame after observe(). Closes the active disturbance
        if it's gone quiet for CLOSE_GAP_SECONDS, same pattern as
        ZoneEventTracker.close_stale_events."""
        if self._active is None:
            return None
        ev = self._active
        if (timestamp - ev['last_seen']).total_seconds() > self.cfg.CLOSE_GAP_SECONDS:
            self._active = None
            self._last_end_timestamp = ev['last_seen']
            return GlobalDisturbanceEvent(
                start_timestamp=ev['start'],
                end_timestamp=ev['last_seen'],
                peak_zone_count=ev['peak_zone_count'],
                peak_zone_fraction=ev['peak_zone_fraction'],
                total_zones=ev['total_zones'],
                involved_zone_ids=sorted(ev['involved_zones']),
            )
        return None

    def close_all(self) -> Optional[GlobalDisturbanceEvent]:
        """Call at end of video to flush anything still active."""
        if self._active is None:
            return None
        ev = self._active
        self._active = None
        return GlobalDisturbanceEvent(
            start_timestamp=ev['start'],
            end_timestamp=ev['last_seen'],
            peak_zone_count=ev['peak_zone_count'],
            peak_zone_fraction=ev['peak_zone_fraction'],
            total_zones=ev['total_zones'],
            involved_zone_ids=sorted(ev['involved_zones']),
        )


AMBIENT_FILTER_AVAILABLE = True        # AmbientMovementFilter is defined directly above -- always present now
GLOBAL_DISTURBANCE_AVAILABLE = True    # GlobalDisturbanceDetector is defined directly above -- always present now


# ========================================================================================
# ========== PRIVATE ZONE TRACKING (merged in from private_zones.py, unchanged) ==========
# ========================================================================================
"""
Per-person personal space, not a fixed room grid.

Every RE_DETECT_INTERVAL seconds, run a person detector (YOLO 'person' class)
to get fresh bounding boxes. Each person's box is padded outward (arm/shoulder
reach margin) to form their PRIVATE ZONE. New detections are matched to
existing tracked people via IoU so a person keeps a stable id across
re-detections. When a motion blob (from the existing MotionEstimator/blob
pipeline) is found INSIDE a private zone that is NOT its owner's own zone, and
this persists for SUSTAINED_FRAMES, that's an intrusion event — mirrors the
same "don't react to one twitch" philosophy as ZoneEventTracker above.

Stays detection-box based (not pose/skeletal) to match PS2's own framing,
which explicitly wants motion-estimation-based ROI detection rather than the
skeletal-keypoint approach that's PS1's ask.
"""


def iou(a: Bbox, b: Bbox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def pad_bbox(b: Bbox, pad: int, max_w: int = None, max_h: int = None) -> Bbox:
    x1, y1, x2, y2 = b
    x1, y1 = x1 - pad, y1 - pad
    x2, y2 = x2 + pad, y2 + pad
    if max_w is not None:
        x1, x2 = max(0, x1), min(max_w, x2)
    if max_h is not None:
        y1, y2 = max(0, y1), min(max_h, y2)
    return (x1, y1, x2, y2)


def bbox_center(b: Bbox) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_in_bbox(pt: Tuple[float, float], b: Bbox) -> bool:
    x, y = pt
    x1, y1, x2, y2 = b
    return x1 <= x <= x2 and y1 <= y <= y2


@dataclass
class PersonBox:
    id: int              # stable tracked id, NOT identity — just a session-local index
    bbox: Bbox
    confidence: float = 1.0


@dataclass
class TrackedPerson:
    id: int
    bbox: Bbox
    private_zone: Bbox
    last_seen_frame: int
    missed_redetects: int = 0


@dataclass
class IntrusionEvent:
    intruder_id: Optional[int]     # whose motion caused it (None if unknown / unmatched)
    zone_owner_id: int             # whose private zone was entered
    start_frame: int
    end_frame: int
    peak_blob_bbox: Bbox
    sustained_frames: int


class PrivateZoneTracker:
    def __init__(self, padding_px: int = 40, iou_match_thresh: float = 0.2,
                 sustained_frames_required: int = 3, cooldown_frames: int = 20,
                 frame_w: Optional[int] = None, frame_h: Optional[int] = None):
        self.padding_px = padding_px
        self.iou_match_thresh = iou_match_thresh
        self.sustained_frames_required = sustained_frames_required
        self.cooldown_frames = cooldown_frames
        self.frame_w = frame_w
        self.frame_h = frame_h

        self.people: Dict[int, TrackedPerson] = {}
        self._next_id = 0

        # intrusion state machine, keyed by (owner_zone_id) since one zone
        # can only be "actively intruded" once at a time in this simple model
        self._consecutive: Dict[int, int] = defaultdict(int)
        self._active: Dict[int, Dict[str, Any]] = {}
        self._last_end_frame: Dict[int, int] = {}

    # ---------- person tracking / re-detection ----------

    def update_person_boxes(self, detections: List[PersonBox], frame_idx: int):
        """
        Call this every RE_DETECT_INTERVAL frames with fresh person detections.
        Matches to existing tracked people by IoU to keep stable ids; if no
        existing person matches well enough, a new id is assigned (e.g. new
        student, or first detection).
        """
        unmatched_existing = set(self.people.keys())
        for det in detections:
            best_id, best_iou = None, 0.0
            for pid in unmatched_existing:
                score = iou(self.people[pid].bbox, det.bbox)
                if score > best_iou:
                    best_iou, best_id = score, pid

            if best_id is not None and best_iou >= self.iou_match_thresh:
                tp = self.people[best_id]
                tp.bbox = det.bbox
                tp.private_zone = pad_bbox(det.bbox, self.padding_px, self.frame_w, self.frame_h)
                tp.last_seen_frame = frame_idx
                tp.missed_redetects = 0
                unmatched_existing.discard(best_id)
            else:
                new_id = det.id if det.id is not None else self._next_id
                self._next_id = max(self._next_id, new_id + 1)
                self.people[new_id] = TrackedPerson(
                    id=new_id,
                    bbox=det.bbox,
                    private_zone=pad_bbox(det.bbox, self.padding_px, self.frame_w, self.frame_h),
                    last_seen_frame=frame_idx,
                )

        # people not matched this re-detect round: mark as missed (may have
        # walked out / been fully occluded); drop after repeated misses
        for pid in unmatched_existing:
            self.people[pid].missed_redetects += 1
            if self.people[pid].missed_redetects >= 3:
                del self.people[pid]

    # ---------- intrusion checking ----------

    def _owning_zone(self, pt: Tuple[float, float]) -> Optional[int]:
        for pid, tp in self.people.items():
            if point_in_bbox(pt, tp.private_zone):
                return pid
        return None

    def check_intrusion(self, blob_bbox: Bbox, frame_idx: int,
                         owner_hint: Optional[int] = None) -> Optional[str]:
        """
        blob_bbox: a detected motion blob (from the existing Farneback/contour
                    pipeline), in full-frame coordinates.
        owner_hint: if known, which person this motion most likely originates
                    from (e.g. the person whose own bbox the blob overlaps
                    most). Pass None if unknown.

        Returns 'start' when a NEW sustained intrusion begins, None otherwise.
        Mirrors ZoneEventTracker.observe()'s state-machine shape so it drops
        into the existing main loop the same way.
        """
        center = bbox_center(blob_bbox)
        entered_zone = self._owning_zone(center)

        if entered_zone is None:
            return None  # blob isn't inside anyone's private zone at all

        is_intrusion = owner_hint is not None and entered_zone != owner_hint
        # if we don't know the owner, fall back to: flag if this blob is NOT
        # anchored near the zone owner's own bbox center (weaker signal, still useful)
        if owner_hint is None:
            owner_bbox = self.people[entered_zone].bbox
            ox1, oy1, ox2, oy2 = owner_bbox
            # if blob center is well outside the owner's own (unpadded) bbox,
            # treat as a foreign-origin motion candidate
            is_intrusion = not point_in_bbox(center, owner_bbox)

        in_cooldown = (frame_idx - self._last_end_frame.get(entered_zone, -10**9)) < self.cooldown_frames

        if is_intrusion and not in_cooldown:
            self._consecutive[entered_zone] += 1
            if entered_zone in self._active:
                ev = self._active[entered_zone]
                ev['last_frame'] = frame_idx
                ev['peak_blob_bbox'] = blob_bbox
                return None
            elif self._consecutive[entered_zone] >= self.sustained_frames_required:
                self._active[entered_zone] = {
                    'intruder_id': owner_hint,
                    'start_frame': frame_idx,
                    'last_frame': frame_idx,
                    'peak_blob_bbox': blob_bbox,
                }
                return 'start'
        else:
            self._consecutive[entered_zone] = 0
        return None

    def close_stale(self, frame_idx: int, gap_frames: int = 5) -> List[IntrusionEvent]:
        closed = []
        for zone_id in list(self._active.keys()):
            ev = self._active[zone_id]
            if frame_idx - ev['last_frame'] > gap_frames:
                closed.append(self._finalize(zone_id, ev))
                del self._active[zone_id]
                self._last_end_frame[zone_id] = ev['last_frame']
        return closed

    def close_all(self, frame_idx: int) -> List[IntrusionEvent]:
        closed = []
        for zone_id, ev in list(self._active.items()):
            closed.append(self._finalize(zone_id, ev))
        self._active.clear()
        return closed

    def _finalize(self, zone_owner_id: int, ev: Dict[str, Any]) -> IntrusionEvent:
        return IntrusionEvent(
            intruder_id=ev.get('intruder_id'),
            zone_owner_id=zone_owner_id,
            start_frame=ev['start_frame'],
            end_frame=ev['last_frame'],
            peak_blob_bbox=ev['peak_blob_bbox'],
            sustained_frames=ev['last_frame'] - ev['start_frame'] + 1,
        )


# ========================================================================================
# ========== PER-PERSON ZONE PROVIDER (YOLO-driven zones, replaces the fixed grid) =======
# ========================================================================================
"""
WHY THIS EXISTS
----------------
The original pipeline partitioned the room into a fixed ZONE_ROWS x ZONE_COLS
grid, independent of where anyone actually sits. That has a real cost: a
single grid cell routinely contains PARTS of two or three different people
(whoever happens to be sitting near that cell's boundary), or empty desk/
aisle space, or a person only half inside it. The zone's motion-sum is then
a blend of "however many different people happen to overlap this rectangle
today", which is a moving target every time the seating arrangement changes,
and makes the baseline for that cell describe an artificial composite
person that doesn't correspond to anyone real. A cell holding two people's
elbows moves roughly twice as much, on average, as a cell holding one -- not
because either individual is doing anything unusual, but purely as an
artifact of how the grid lines happened to fall.

This module replaces the fixed grid with a DYNAMIC, YOLO-driven zone per
detected person: each zone IS one person's own bounding box (padded a bit
for arm/shoulder reach), tracked with a stable id across frames via the
same IoU-matching approach PrivateZoneTracker already uses for its private-
zone intrusion feature. The baseline for zone N is now built ONLY from
person N's own historical motion, seat empty or not, so it's exactly the
apples-to-apples comparison the "human invigilator" design philosophy at
the top of this file was already going for: not "is this rectangle of
carpet unusually active", but "is THIS person unusually active, compared to
their own normal".

This class is a thin adapter over PersonZoneProvider._tracker (an internal
PrivateZoneTracker instance) that exposes the SAME interface ZoneGrid did
(num_zones, zone_bbox(zone_id), zone_row_col(zone_id)) so the rest of the
pipeline -- baseline engine, event tracker, blob detection, ambient filter,
global disturbance detector, annotation, reporting -- needs no changes
beyond swapping which zone-provider object it talks to. "zone_id" in every
downstream call is now a person's stable tracked id, not a grid index.

FALLBACK: if YOLO/ultralytics isn't available, or a particular frame has no
person detections yet (e.g. before the very first detection pass completes),
this transparently falls back to a fixed ZoneGrid so the pipeline is never
left with zero zones to analyze. Once real detections arrive, per-person
zones take over automatically.

=== LOCK-AFTER-CALIBRATION (added) ===
Originally this provider kept re-detecting and IoU-matching people for the
entire video, on a fixed PERSON_ZONE_REDETECT_SECONDS cadence, for the whole
run. In an exam hall, students are seated and largely static -- there's no
real need to keep re-drawing zone boundaries every couple of seconds, and
doing so has a real cost: any re-detection round where a person's box drifts
enough that IoU-match against their previous box falls under
PERSON_ZONE_IOU_MATCH_THRESH gets treated as "new person" and assigned a
fresh id. Since baseline history, event-tracker consecutive-hit counters, and
ambient-filter centroid history are all keyed by zone_id, an id change
silently wipes that person's accumulated history and restarts their baseline
from scratch -- which looks, from the outside, like the whole zone grid is
"shuffling" between frames, and defeats the per-person private zone
intrusion feature that depends on stable ids.

FIX: zones are now allowed to keep re-detecting/settling for
PERSON_ZONE_LOCK_SECONDS (a short stabilization window, longer than one
re-detect interval so a missed match has a chance to self-correct), and then
permanently LOCKED via lock() -- after that, update_person_boxes() and
refresh_frame_ordering() become no-ops and self._frame_zone_ids never
changes again for the rest of the video. This trades "adapts to someone
getting up and moving mid-exam" for "zone ids and boundaries are stable for
the whole run", which is the right trade for a mostly-static seated-exam
scenario -- see ROIPipeline.process_video for where lock() gets called.
"""


class PersonZoneProvider:
    """
    IMPORTANT: zone_id here IS the person's own stable tracked id (an int
    assigned once per person by the underlying IoU tracker and kept for as
    long as that person keeps being re-matched), NOT a frame-local
    re-sortable index. This matters a lot: baseline_engine, event_tracker,
    and ambient_filter all key their internal state (motion history,
    consecutive-hit counters, candidate streaks) by zone_id. If zone_id were
    instead "the Nth person in sorted order this frame", then the moment
    ANY tracked person's id changed rank in that sorted order (someone
    left, someone new appeared before them, re-detection reordered ties),
    every OTHER zone_id downstream of that shift would silently start
    reading/writing a completely different person's accumulated history --
    corrupting baselines across the board. Using the person's own permanent
    id as zone_id sidesteps that entirely: person 7's state always lives
    under key 7, whether they're the 1st or the 5th person in this frame's
    detections.

    Consequence: callers must iterate zone_ids() (the actual set of
    currently-tracked ids, which is sparse and changes size/membership
    frame to frame while UNLOCKED, and is completely fixed once locked)
    rather than range(num_zones). num_zones is kept only for
    logging/reporting purposes (e.g. "N zones active this frame").
    """

    # Fallback-grid zone_ids are offset well above any realistic YOLO
    # tracked-person-id range so the two id spaces can never collide if a
    # video switches between fallback and per-person mode mid-run (e.g.
    # YOLO briefly has zero detections for a few frames then resumes).
    _FALLBACK_ID_OFFSET = 1_000_000

    def __init__(self, frame_w: int, frame_h: int, padding_px: int = 40,
                 iou_match_thresh: float = 0.3,
                 fallback_grid_rows: int = 3, fallback_grid_cols: int = 4):
        self.frame_w = frame_w
        self.frame_h = frame_h
        # Reuses PrivateZoneTracker purely as an IoU-tracked person registry
        # -- its own intrusion-detection state machine (check_intrusion,
        # close_stale, etc.) is simply never called from here; the pipeline's
        # separate PrivateZoneTracker instance still owns that job. This one
        # exists only to answer "what are this frame's per-person zones".
        self._tracker = PrivateZoneTracker(
            padding_px=padding_px,
            iou_match_thresh=iou_match_thresh,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        # Snapshot of this frame's active zone ids, taken once per frame by
        # refresh_frame_ordering() so PASS 1 and PASS 2 (which both iterate
        # zone_ids()) see the exact same set for the whole frame even if a
        # re-detection would otherwise land mid-frame.
        self._frame_zone_ids: List[int] = []

        # Fixed-grid fallback for frames where no person is currently tracked
        # (YOLO unavailable, or before the first detection pass ever runs).
        self._fallback_grid = ZoneGrid(frame_w, frame_h, fallback_grid_rows, fallback_grid_cols)
        self._using_fallback = True

        # See "LOCK-AFTER-CALIBRATION" module docstring above. Once locked,
        # update_person_boxes() and refresh_frame_ordering() are no-ops and
        # _frame_zone_ids / each zone's bbox never change again.
        self._locked = False

    def update_person_boxes(self, detections: List[PersonBox], frame_idx: int):
        """Call this every re-detect interval with fresh YOLO person boxes.
        No-op once lock() has been called -- see class/module docstring."""
        if self._locked:
            return
        self._tracker.update_person_boxes(detections, frame_idx)

    def refresh_frame_ordering(self):
        """
        Call ONCE at the start of each analyzed frame, before any zone_bbox()
        / zone_ids() calls for that frame. Snapshots the current set of
        tracked person ids for this frame's PASS 1 / PASS 2 loops. Falls
        back to the fixed grid whenever there are currently no tracked
        people at all (nobody detected yet, or YOLO unavailable).

        No-op once lock() has been called: _frame_zone_ids was already
        frozen by lock() and must not be recomputed afterward.
        """
        if self._locked:
            return
        if self._tracker.people:
            self._frame_zone_ids = sorted(self._tracker.people.keys())
            self._using_fallback = False
        else:
            self._using_fallback = True
            self._frame_zone_ids = [self._FALLBACK_ID_OFFSET + i
                                     for i in range(self._fallback_grid.num_zones)]

    def lock(self):
        """
        Freeze the zone grid permanently using whichever people are
        currently tracked. After this call, update_person_boxes() and
        refresh_frame_ordering() become no-ops -- zone_ids(), zone_bbox(),
        and zone_row_col() will keep returning exactly what they returned
        at the moment of locking, for the rest of the video.

        If nobody is tracked yet at lock time, falls back to the fixed grid
        instead of locking an empty zone set (see using_fallback_grid).
        """
        if self._tracker.people:
            self._frame_zone_ids = sorted(self._tracker.people.keys())
            self._using_fallback = False
        else:
            self._using_fallback = True
            self._frame_zone_ids = [self._FALLBACK_ID_OFFSET + i
                                     for i in range(self._fallback_grid.num_zones)]
        self._locked = True

    @property
    def is_locked(self) -> bool:
        return self._locked

    @property
    def using_fallback_grid(self) -> bool:
        return self._using_fallback

    @property
    def num_zones(self) -> int:
        return len(self._frame_zone_ids)

    def zone_ids(self) -> List[int]:
        """The actual set of zone ids to iterate THIS frame. Use this
        instead of range(num_zones) -- see class docstring for why."""
        return self._frame_zone_ids

    def zone_bbox(self, zone_id: int) -> Tuple[int, int, int, int]:
        if self._using_fallback:
            return self._fallback_grid.zone_bbox(zone_id - self._FALLBACK_ID_OFFSET)
        return self._tracker.people[zone_id].private_zone

    def zone_row_col(self, zone_id: int) -> Tuple[int, int]:
        """
        Kept for interface compatibility with ZoneGrid (used only in log
        messages / the ROIEvent.zone_row/zone_col fields). In per-person
        mode there's no grid row/col, so this returns (zone_id, 0) -- the
        person's own stable tracked id doubling as "row", 0 as "col" -- so
        existing log lines and report fields still carry a meaningful,
        stable identifier instead of a fabricated grid position.
        """
        if self._using_fallback:
            return self._fallback_grid.zone_row_col(zone_id - self._FALLBACK_ID_OFFSET)
        return (zone_id, 0)

    def person_id_for_zone(self, zone_id: int) -> Optional[int]:
        """Explicit accessor for the real person id behind a zone_id (when
        not in fallback mode, zone_id already IS the person id -- this
        exists mainly so callers don't need to know that fact directly)."""
        if self._using_fallback:
            return None
        return zone_id


# ========================================================================================
# ========== VIDEO HEATMAP ACCUMULATOR (merged in from heatmap.py, fixed version) ========
# ========================================================================================
"""
Fixes two real bugs in a naive save_heatmap():
  1. Normalizing by raw .max() -> a single outlier frame (person standing up,
     door opening) crushes every other frame's motion toward black. For a
     90-minute exam recording the heatmap ends up looking mostly empty except
     one or two hot spots, hiding the low-level activity pattern that's
     actually the point of the heatmap.
  2. Only accumulating raw magnitude sum -> can't tell "briefly very active"
     apart from "mildly active for a long time." Both produce the same pixel
     intensity once you're summing over an entire video.

Fixes: percentile clipping (default 99th) instead of raw max, plus two
accumulation modes maintained simultaneously — intensity (total motion
energy) and presence (fraction of video duration a pixel was active, usually
the more useful investigator-facing view) — with an optional per-zone
breakdown so the heatmap can be cross-checked against the same zone IDs used
in the event log.
"""


class VideoHeatmapAccumulator:
    def __init__(self, height: int, width: int, presence_thresh: float = 1.0):
        self.height = height
        self.width = width
        self.intensity_accum = np.zeros((height, width), dtype=np.float64)
        self.presence_accum = np.zeros((height, width), dtype=np.float64)
        self.presence_thresh = presence_thresh
        self.frames_seen = 0

    def add_frame(self, magnitude: np.ndarray):
        """magnitude: per-pixel optical flow magnitude for one analyzed frame."""
        self.intensity_accum += magnitude
        self.presence_accum += (magnitude > self.presence_thresh).astype(np.float64)
        self.frames_seen += 1

    def _normalize_percentile(self, accum: np.ndarray, clip_percentile: float = 99.0) -> np.ndarray:
        nonzero = accum[accum > 0]
        if nonzero.size == 0:
            return np.zeros_like(accum, dtype=np.uint8)
        clip_val = np.percentile(nonzero, clip_percentile)
        if clip_val <= 0:
            clip_val = accum.max() if accum.max() > 0 else 1.0
        clipped = np.clip(accum, 0, clip_val)
        norm = (clipped / clip_val * 255.0).astype(np.uint8)
        return norm

    def render(self, mode: str = "intensity", clip_percentile: float = 99.0,
               blur_ksize: int = 15) -> np.ndarray:
        """
        mode: 'intensity' (total motion energy, percentile-clipped) or
              'presence' (fraction of video duration this pixel was active).
        Returns a BGR colored heatmap image ready to imwrite.
        """
        accum = self.intensity_accum if mode == "intensity" else self.presence_accum
        norm = self._normalize_percentile(accum, clip_percentile)
        if blur_ksize > 1:
            norm = cv2.GaussianBlur(norm, (blur_ksize, blur_ksize), 0)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        return colored

    def save(self, out_dir: Path, prefix: str = "motion_heatmap",
              clip_percentile: float = 99.0):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        intensity_img = self.render("intensity", clip_percentile)
        presence_img = self.render("presence", clip_percentile)
        intensity_path = out_dir / f"{prefix}_intensity.jpg"
        presence_path = out_dir / f"{prefix}_presence.jpg"
        cv2.imwrite(str(intensity_path), intensity_img)
        cv2.imwrite(str(presence_path), presence_img)
        return intensity_path, presence_path

    def zone_summary(self, zone_bboxes: List[Tuple[int, int, int, int]]) -> List[dict]:
        """
        Per-zone stats so the heatmap can be cross-referenced against the
        same zone IDs used in roi_events.csv. presence_pct is the most
        interpretable number for a report: "Zone 5 was active in 34% of
        analyzed frames."
        """
        rows = []
        for zid, (x1, y1, x2, y2) in enumerate(zone_bboxes):
            intensity_sum = float(self.intensity_accum[y1:y2, x1:x2].sum())
            presence_pct = float(self.presence_accum[y1:y2, x1:x2].mean() / max(1, self.frames_seen) * 100.0)
            rows.append({
                "zone_id": zid,
                "intensity_sum": round(intensity_sum, 1),
                "presence_pct_of_video": round(presence_pct, 2),
            })
        return rows


# ========== CONFIGURATION ==========
@dataclass
class Settings:
    LOG_LEVEL: str = "INFO"
    FPS: int = 5                          # sampling fps for analysis (higher than PS1 since no heavy pose model)
    RESOLUTION_WIDTH: int = 1280
    RESOLUTION_HEIGHT: int = 720

    # --- zone source: per-person (YOLO-driven, default) vs fixed grid (fallback) ---
    # See PersonZoneProvider for the full rationale. When USE_PERSON_ZONES is
    # True, each zone is one detected person's own padded bounding box,
    # tracked with a stable id -- NOT a fixed grid cell that several
    # different people (or nobody) might drift in and out of. The fixed grid
    # below is kept only as the automatic fallback for frames where no
    # person is currently tracked (YOLO unavailable, or before the first
    # detection pass completes).
    USE_PERSON_ZONES: bool = True
    PERSON_ZONE_PADDING_PX: int = 45       # arm/shoulder reach margin around each person's own box
    PERSON_ZONE_REDETECT_SECONDS: float = 2.0  # how often to refresh person boxes for zoning, WHILE UNLOCKED
    # (separate from PERSON_REDETECT_SECONDS below, which is intrusion-
    # tracking's own cadence -- zoning wants to refresh more often since a
    # missed/stale person box now means a missed/stale PRIMARY zone, not
    # just a secondary intrusion check)
    PERSON_ZONE_IOU_MATCH_THRESH: float = 0.3
    # --- lock-after-calibration (see PersonZoneProvider module docstring) ---
    # Zones re-detect/settle normally until this many seconds of video have
    # elapsed, then PersonZoneProvider.lock() is called ONCE and the zone
    # grid (ids + boundaries) never changes again for the rest of the run.
    # Keeps every zone_id's accumulated baseline/event-tracker/ambient-filter
    # history stable and prevents the "grid keeps reshuffling every frame"
    # symptom that comes from IoU-match misses re-assigning a seated,
    # basically-stationary student a brand-new id mid-exam.
    PERSON_ZONE_LOCK_SECONDS: float = 20.0

    # --- zone grid: FALLBACK ONLY when no people are currently tracked (see above) ---
    ZONE_ROWS: int = 3
    ZONE_COLS: int = 4

    # --- calibration: quiet-room learning period, like an invigilator settling in ---
    CALIBRATION_SECONDS: float = 20.0

    # --- sustained deviation thresholds (an invigilator ignores single twitches) ---
    # TWO-PART GATE (see ZoneBaselineEngine's v3 FIX HISTORY docstring for the full
    # rationale). A zone is only deviant if BOTH of these clear:
    #   1. RELATIVE: z-score exceeds ZONE_Z_THRESHOLD (lowered from the old v2 value
    #      of 3.0 -- it no longer has to single-handedly separate "burst" from
    #      "anomaly", since the absolute gate below now does that job).
    #   2. ABSOLUTE: motion_sum exceeds baseline.ceiling (the calibration window's own
    #      p90) times ZONE_ABS_CEILING_MULT -- i.e. it must clear the *actual observed*
    #      ceiling of normal behavior by a fixed margin, not a ceiling stretched
    #      further by a multiplier applied to a possibly-wide spread (the exact v2
    #      failure: median~6, p90~64, upper_spread~58 -> old bar = median+3*58 = ~180,
    #      above even a genuinely 5x-larger sustained anomaly of mean~150).
    # These two defaults were chosen via synthetic testing across near-silent,
    # typical, and high-variance/fidgety calibration windows with injected sustained
    # anomalies: 0% false-positive zones and 100% true-positive zones on strong,
    # sustained anomalies, with expected graceful degradation only on weak/borderline
    # anomalies in already-fidgety zones (a mild deviation is inherently harder to
    # tell apart from that zone's own normal variance -- true for a human invigilator
    # too, not a flaw specific to this gate).
    ZONE_Z_THRESHOLD: float = 1.5          # relative gate (was 3.0 under the old, single-part v2 design)
    ZONE_ABS_CEILING_MULT: float = 1.75    # absolute gate: motion_sum must clear ceiling (p90) * this
    SUSTAINED_FRAMES: int = 3             # must persist this many consecutive analyzed frames
    EVENT_COOLDOWN_SECONDS: float = 4.0   # don't re-trigger the same zone immediately after closing an event

    # --- blob detection within a zone ---
    MIN_BLOB_AREA: int = 150              # ignore tiny noise contours
    MOTION_BIN_THRESH: int = 25           # threshold on frame-diff / flow magnitude to binarize motion

    # --- clip segmentation ---
    CLIP_PRE_BUFFER_SECONDS: float = 3.0
    CLIP_POST_BUFFER_SECONDS: float = 3.0

    MODEL_DIR: Path = Path("./models")
    YOLO_MODEL: str = "yolov9c.pt"
    PROHIBITED_CLASSES: Tuple[str, ...] = ("cell phone", "book")  # COCO has no "chit"; book is closest proxy for paper

    # --- private zone tracking (per-person personal space, see PrivateZoneTracker) ---
    # NOTE: this is the SEPARATE intrusion-detection feature (is someone's
    # motion straying into a NEIGHBOR's space) -- distinct from the
    # USE_PERSON_ZONES primary zoning above, even though both are now
    # YOLO-driven and both use padded person boxes. They intentionally stay
    # as two separate tracker instances (see ROIPipeline.__init__) because
    # they answer different questions and re-detect on different cadences.
    # UNLIKE primary zoning, this one is NOT locked -- intrusion detection
    # inherently needs to keep tracking each person's current bbox to know
    # whether a motion blob has strayed outside it, so it keeps re-detecting
    # for the whole video regardless of PERSON_ZONE_LOCK_SECONDS.
    USE_PRIVATE_ZONES: bool = True        # if False, intrusion detection is skipped entirely
    PERSON_REDETECT_SECONDS: float = 5.0  # how often to re-run person detection for intrusion tracking
    PRIVATE_ZONE_PADDING_PX: int = 50     # arm/shoulder reach margin around each person box
    INTRUSION_SUSTAINED_FRAMES: int = 3
    INTRUSION_COOLDOWN_FRAMES: int = 20

    # --- heatmap normalization (see VideoHeatmapAccumulator) ---
    HEATMAP_PRESENCE_THRESH: float = 1.0  # magnitude threshold for "pixel was active" in presence mode
    HEATMAP_CLIP_PERCENTILE: float = 99.0

    # --- ambient movement filter (teacher/invigilator walking vs seated student gesture) ---
    USE_AMBIENT_FILTER: bool = True        # AmbientMovementFilter is always available (inlined above)

    # --- global disturbance detector (room-wide moment vs one student's localized motion) ---
    USE_GLOBAL_DISTURBANCE_DETECTOR: bool = True   # GlobalDisturbanceDetector is always available (inlined above)

    OUTPUT_DIR: Path = field(default_factory=resolve_output_dir)

settings = Settings()

ROI_FRAMES_DIR = settings.OUTPUT_DIR / "roi_frames"
CLIPS_DIR = settings.OUTPUT_DIR / "event_clips"
HEATMAP_DIR = settings.OUTPUT_DIR / "heatmaps"
ROI_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
CLIPS_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

# ========== LOGGING ==========
def get_logger(name):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "\033[36m%(asctime)s\033[0m | \033[33m%(levelname)-8s\033[0m | \033[35m%(name)s\033[0m | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
logger = get_logger("drishti_ps2")

def _fmt(v, spec=".2f"):
    return format(v, spec) if v is not None else "N/A"


# ========== SCHEMAS ==========
@dataclass
class ZoneBaseline:
    zone_id: int
    mean: float = 0.0
    std: float = 1.0
    sample_count: int = 0
    ceiling: float = 0.0   # the calibration window's own upper_percentile value (see
                           # ZoneBaselineEngine.get_baseline / the "compound multiplier"
                           # fix note above ZoneBaselineEngine) -- used as the second,
                           # absolute half of the is_deviant gate, alongside the z-score

@dataclass
class ROIEvent:
    zone_id: int
    zone_row: int
    zone_col: int
    start_timestamp: datetime
    end_timestamp: datetime
    peak_motion_z: float
    peak_frame_number: int
    blob_bbox: Tuple[int, int, int, int]     # x1,y1,x2,y2 in full-frame coords
    detected_object: Optional[str] = None
    object_confidence: Optional[float] = None
    clip_path: str = ""
    roi_frame_path: str = ""
    flag_reason: str = ""

    def to_dict(self):
        d = self.__dict__.copy()
        d['start_timestamp'] = self.start_timestamp.isoformat()
        d['end_timestamp'] = self.end_timestamp.isoformat()
        return d


# ========== INGESTION (ffmpeg fallback + resampling) ==========
def validate_video(video_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    ret, _ = cap.read()
    cap.release()
    return ret

def _ensure_opencv_readable(video_path: Path):
    if validate_video(video_path):
        return video_path, False
    logger.warning(f"OpenCV cannot open {video_path.name} — trying FFmpeg fallback...")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg not installed. Please install ffmpeg.")
    tmp_dir = Path(tempfile.gettempdir())
    path_hash = hashlib.sha256(str(video_path.resolve()).encode()).hexdigest()[:10]
    converted_path = tmp_dir / f"{video_path.stem}_{path_hash}_converted.mp4"
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(converted_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FFmpeg conversion timed out for {video_path.name}")
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {video_path.name} - {result.stderr[-500:]}")
    if not validate_video(converted_path):
        raise RuntimeError(f"FFmpeg output still unreadable: {converted_path}")
    logger.info(f"FFmpeg conversion succeeded: {converted_path}")
    return converted_path, True

def sample_frames(video_path: Path, target_fps: int, max_frames: Optional[int] = None):
    """Yields (timestamp, frame_index_in_source, frame). Also returns source fps via closure attr."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    video_path, was_converted = _ensure_opencv_readable(video_path)
    cap = None
    frame_count = 0
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0:
            orig_fps = 25.0
        sample_frames.source_fps = orig_fps
        step = max(1, int(round(orig_fps / target_fps)))
        frame_idx = 0
        consecutive_failures = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > 5:
                    break
                continue
            consecutive_failures = 0
            if frame_idx % step == 0:
                timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if timestamp_sec < 0:
                    timestamp_sec = frame_idx / orig_fps
                timestamp = datetime.fromtimestamp(timestamp_sec)
                if frame is not None:
                    frame = cv2.resize(frame, (settings.RESOLUTION_WIDTH, settings.RESOLUTION_HEIGHT))
                    yield (timestamp, frame_idx, frame)
                    frame_count += 1
                    if max_frames is not None and frame_count >= max_frames:
                        break
            frame_idx += 1
        logger.info(f"Finished sampling video: {video_path.name} (analyzed frames={frame_count}, source_fps={orig_fps:.2f})")
    finally:
        if cap is not None:
            cap.release()
        if was_converted:
            try:
                video_path.unlink(missing_ok=True)
            except OSError:
                pass
sample_frames.source_fps = 25.0


# ========== MOTION ESTIMATION (OpenCV Farneback — no external weight files needed) ==========
class MotionEstimator:
    """
    Frame-to-frame dense optical flow. This is PS2's core required technique
    ('motion estimation techniques' / 'frame-to-frame motion estimation').
    Farneback is used instead of RAFT so the script runs anywhere with just
    OpenCV — no .pth weight file to resolve, no torch dependency. Swap in RAFT
    here later if GPU + weights are available; the rest of the pipeline only
    needs a per-pixel magnitude map back from this class.
    """
    def __init__(self):
        self.prev_gray = None

    def extract(self, frame: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.prev_gray is None:
            self.prev_gray = gray
            return None
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        self.prev_gray = gray
        return magnitude  # HxW float32, per-pixel motion magnitude


# ========== OBJECT DETECTION (optional, graceful fallback) ==========
class ObjectExtractor:
    def __init__(self, model_path: Path):
        self.model = None
        self.fallback = True
        if not ULTRALYTICS_AVAILABLE:
            logger.warning("ultralytics not installed — object detection disabled (motion/ROI analytics still fully active). "
                            "Run: pip install ultralytics --break-system-packages")
            return
        if not model_path.exists():
            logger.warning(f"YOLO weights not found: {model_path} — object detection disabled.")
            return
        try:
            self.model = UltralyticsYOLO(str(model_path))
            self.fallback = False
            logger.info(f"YOLO loaded: {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load YOLO: {e}. Object detection disabled.")

    def extract(self, frame: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> List[Dict[str, Any]]:
        if self.fallback or frame is None:
            return []
        img = frame
        if region is not None:
            x1, y1, x2, y2 = region
            img = frame[max(0, y1):y2, max(0, x1):x2]
            if img.size == 0:
                return []
        try:
            results = self.model(img, verbose=False)[0]
            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                name = self.model.names.get(cls_id, str(cls_id)) if isinstance(self.model.names, dict) else self.model.names[cls_id]
                if name in settings.PROHIBITED_CLASSES:
                    detections.append({'class': name, 'confidence': float(box.conf[0])})
            return detections
        except Exception as e:
            logger.debug(f"YOLO inference failed: {e}")
            return []

    def detect_people(self, frame: np.ndarray) -> List[PersonBox]:
        """
        Full-frame 'person' class detections, used to build/refresh private
        zones. Separate from extract() above (which crops to a region and
        looks for prohibited objects) since this needs the whole frame and a
        different COCO class. Person ids here are just detection-order
        indices, re-assigned each call — PrivateZoneTracker.update_person_boxes
        handles matching them back to stable tracked IDs via IoU.
        """
        if self.fallback or frame is None:
            return []
        try:
            results = self.model(frame, verbose=False, classes=None)[0]
            people = []
            pid = 0
            for box in results.boxes:
                cls_id = int(box.cls[0])
                name = self.model.names.get(cls_id, str(cls_id)) if isinstance(self.model.names, dict) else self.model.names[cls_id]
                if name == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    people.append(PersonBox(id=pid, bbox=(x1, y1, x2, y2), confidence=float(box.conf[0])))
                    pid += 1
            return people
        except Exception as e:
            logger.debug(f"Person detection failed: {e}")
            return []


# ========== ZONE GRID ("how an invigilator partitions the room") ==========
class ZoneGrid:
    def __init__(self, width: int, height: int, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.width = width
        self.height = height
        self.zone_w = width // cols
        self.zone_h = height // rows

    def zone_bbox(self, zone_id: int) -> Tuple[int, int, int, int]:
        r, c = divmod(zone_id, self.cols)
        x1 = c * self.zone_w
        y1 = r * self.zone_h
        x2 = self.width if c == self.cols - 1 else x1 + self.zone_w
        y2 = self.height if r == self.rows - 1 else y1 + self.zone_h
        return (x1, y1, x2, y2)

    def zone_row_col(self, zone_id: int) -> Tuple[int, int]:
        return divmod(zone_id, self.cols)

    def zone_ids(self) -> List[int]:
        """Same iteration interface as PersonZoneProvider.zone_ids() -- for
        a fixed grid this is just the plain 0..num_zones-1 range, always."""
        return list(range(self.rows * self.cols))

    @property
    def num_zones(self) -> int:
        return self.rows * self.cols


# ========== ZONE BASELINE ENGINE (percentile-based, robust to bursty motion) ==========
"""
FIX HISTORY
-----------
v1 was median/MAD. That failed on handwriting: writing motion is BURSTY /
BIMODAL (a pen-down stroke, high motion, followed by a pause to think or
reposition, near-zero motion, repeating through the whole video, calibration
window included). The MEDIAN sits in the low-motion majority (idle time
outweighs stroke time at 5 FPS sampling), and the MAD -- median of
|sample - median| -- is dominated by that same tightly-clustered low-motion
majority, so it comes out razor-thin. z = (motion - median) / MAD then
explodes to double digits on every ordinary stroke, for the rest of the
video (baseline freezes after CALIBRATION_SECONDS). That version flagged a
zone as deviant almost continuously, even on ordinary writing.

v2 switched to median + (p90-median) upper-spread, keeping the same
z = (motion - center) / upper_spread formula and the same
ZONE_Z_THRESHOLD=3.0 multiplier semantics. That fixed the false-positive
flood, but overcorrected into the OPPOSITE failure once per-person zones
made each zone's calibration window a single individual's own motion
(rather than several people's motion blended together): a single person's
upper_spread, multiplied by 3, produces an enormous absolute bar. Concretely,
for a typical writing-motion distribution: median~6, p90~64, upper_spread~58,
so the v2 flagging line sits at median + 3*58 = ~180 -- meaning even a
genuinely different, 5x-larger, sustained motion event (mean~150, clearly
NOT normal writing) landed BELOW the line and was never flagged. This is the
"kuch bhi detect nahi ho raha" (nothing is detecting at all) failure:
the multiplier was compounding an already-generous spread into a bar so high
that real anomalies couldn't clear it, particularly for zones whose own
calibration variance happens to be wide (an expressive/fidgety writer).

FIX (v3) -- TWO-PART GATE instead of a single multiplied z-score:
  1. RELATIVE part (same shape as v2, smaller multiplier): the z-score,
     z = (motion - median) / upper_spread, must exceed Z_THRESHOLD (now a
     much smaller value -- see ZONE_Z_THRESHOLD's new default -- since it is
     no longer being asked to single-handedly separate "burst" from
     "anomaly"; upper_spread already captures burst behavior).
  2. ABSOLUTE part (new): motion_sum must ALSO exceed
     ceiling * ZONE_ABS_CEILING_MULT, where ceiling is the calibration
     window's own upper_percentile value (p90) directly -- i.e. the anomaly
     must clear the *actual observed ceiling of normal behavior* by a fixed
     margin, not a ceiling stretched further by a large multiplier applied
     to a possibly-wide spread.
  BOTH must be true. This keeps false positives at ~0% on ordinary bursty
  writing (validated across near-silent zones, typical writers, and
  high-variance fidgety writers in synthetic testing) while restoring
  ~95% true-positive detection on motion that is genuinely well beyond
  anything the calibration window ever saw -- vs. ~6% true-positive
  detection under the v2 single-multiplier design in the same test.
  The absolute part is what actually does the work here: it stops a wide
  spread from silently raising the bar past what any real anomaly would
  reach, which is exactly the mechanism that broke detection in v2.
"""


class ZoneBaselineEngine:
    def __init__(self, buffer_size: int = 200, upper_percentile: float = 90.0):
        self.buffers: Dict[int, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self.upper_percentile = upper_percentile

    def update(self, zone_id: int, motion_sum: float):
        self.buffers[zone_id].append(motion_sum)

    def get_baseline(self, zone_id: int) -> Optional[ZoneBaseline]:
        buf = self.buffers[zone_id]
        if len(buf) < 5:
            return None
        arr = np.array(buf)
        center = float(np.median(arr))
        upper = float(np.percentile(arr, self.upper_percentile))
        # floor relative to the data's own scale (not a fixed tiny constant)
        # so a genuinely near-silent zone still gets a sane, non-zero spread
        # instead of every micro-fluctuation reading as an extreme z-score.
        floor = max(1e-3, 0.05 * max(center, 1.0))
        upper_spread = max(upper - center, floor)
        return ZoneBaseline(zone_id=zone_id, mean=center, std=upper_spread,
                             sample_count=len(buf), ceiling=upper)


# ========== BLOB DETECTION ("where exactly did the eye get drawn") ==========
"""
FULL-FRAME, NOT PER-ZONE-CROPPED.

The original version cropped the motion mask to each zone's own bbox BEFORE
running cv2.findContours, then found the dominant blob inside that crop. That
works fine for a blob that's fully inside one zone, but silently truncates
any blob that straddles a zone boundary -- e.g. an invigilator standing with
their head/torso in one zone and legs in the zone below, split exactly on the
grid line. Each zone then only ever sees its OWN half of that person (a
squat, non-tall fragment), which is precisely the shape the ambient filter's
tall_aspect/full_height signals are designed to reject as "ambient" -- so a
walking/standing adult straddling a boundary gets misread by both halves as
two separate small localized gestures, and BOTH zones can independently open
a spurious ROI event for what is really one person walking through.

FIX: find all blobs ONCE on the full-frame motion mask (no per-zone crop at
contour-finding time), then, for each zone, figure out which full blob(s)
overlap it and pick the dominant one *by the blob's true full-frame area*,
not by however much of it happens to poke into that zone. A blob spanning
multiple zones is therefore reported with its true, untruncated bbox to
every zone it touches -- so the ambient filter's shape signals see the real
height/aspect of the person, in every zone they're passing through, and the
"tall walking body" signature can actually fire.
"""


def find_all_blobs(motion_mask: np.ndarray, min_area: int) -> List[Tuple[int, int, int, int]]:
    """
    Runs contour detection ONCE on the whole frame's motion mask. Returns
    every blob (as a full-frame-coordinate bbox) with area >= min_area,
    largest first. Zone assignment happens separately in
    assign_blobs_to_zone() below, so the same blob list can be reused for
    every zone in a frame instead of re-running findContours per zone.
    """
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        blobs.append((area, (bx, by, bx + bw, by + bh)))
    blobs.sort(key=lambda t: t[0], reverse=True)
    return [b for _, b in blobs]


def _blob_overlaps_zone(blob_bbox: Tuple[int, int, int, int],
                         zone_bbox: Tuple[int, int, int, int]) -> int:
    """Intersection area (in px^2) between a full-frame blob and a zone."""
    bx1, by1, bx2, by2 = blob_bbox
    zx1, zy1, zx2, zy2 = zone_bbox
    ix1, iy1 = max(bx1, zx1), max(by1, zy1)
    ix2, iy2 = min(bx2, zx2), min(by2, zy2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    return iw * ih


def find_dominant_blob_for_zone(all_blobs: List[Tuple[int, int, int, int]],
                                 zone_bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
    """
    Given the full set of full-frame blobs for this analyzed frame (from
    find_all_blobs, called once per frame), returns the dominant blob that
    overlaps this zone -- picked by the blob's OWN true full-frame area
    (all_blobs is already sorted largest-first), not by how much of it lands
    inside this particular zone. The returned bbox is the blob's full,
    untruncated extent, even if most of it lies outside this zone -- that's
    the whole point: a zone that's only brushed by the edge of a tall
    walking body should still see that body's real height, not a sliver.
    Returns None if no blob overlaps this zone at all.
    """
    for blob_bbox in all_blobs:  # already largest-first
        if _blob_overlaps_zone(blob_bbox, zone_bbox) > 0:
            return blob_bbox
    return None


# ========== EVENT STATE MACHINE (sustained deviation, cooldown) ==========
class ZoneEventTracker:
    def __init__(self):
        self.consecutive_hits: Dict[int, int] = defaultdict(int)
        self.active_event: Dict[int, Dict[str, Any]] = {}
        self.last_event_end: Dict[int, datetime] = {}

    def in_cooldown(self, zone_id: int, timestamp: datetime) -> bool:
        last_end = self.last_event_end.get(zone_id)
        if last_end is None:
            return False
        return (timestamp - last_end).total_seconds() < settings.EVENT_COOLDOWN_SECONDS

    def observe(self, zone_id: int, is_deviant: bool, z_score: float, timestamp: datetime,
                frame_number: int, blob_bbox: Optional[Tuple[int, int, int, int]]) -> Optional[str]:
        """
        Returns 'start' when a new sustained event begins, None otherwise.
        Event closing is handled externally by explicit close_event() call
        once deviation subsides (see main loop).
        """
        if is_deviant and not self.in_cooldown(zone_id, timestamp):
            self.consecutive_hits[zone_id] += 1
            if zone_id in self.active_event:
                # update ongoing event's peak
                ev = self.active_event[zone_id]
                if z_score > ev['peak_motion_z']:
                    ev['peak_motion_z'] = z_score
                    ev['peak_frame_number'] = frame_number
                    if blob_bbox is not None:
                        ev['blob_bbox'] = blob_bbox
                ev['last_seen'] = timestamp
                return None
            elif self.consecutive_hits[zone_id] >= settings.SUSTAINED_FRAMES:
                self.active_event[zone_id] = {
                    'start': timestamp,
                    'last_seen': timestamp,
                    'peak_motion_z': z_score,
                    'peak_frame_number': frame_number,
                    'blob_bbox': blob_bbox or (0, 0, 0, 0),
                }
                return 'start'
        else:
            self.consecutive_hits[zone_id] = 0
        return None

    def close_stale_events(self, timestamp: datetime, gap_seconds: float = 1.5) -> List[Tuple[int, Dict[str, Any]]]:
        closed = []
        for zone_id in list(self.active_event.keys()):
            ev = self.active_event[zone_id]
            if (timestamp - ev['last_seen']).total_seconds() > gap_seconds:
                closed.append((zone_id, ev))
                del self.active_event[zone_id]
                self.last_event_end[zone_id] = ev['last_seen']
        return closed

    def close_all(self) -> List[Tuple[int, Dict[str, Any]]]:
        closed = [(zid, ev) for zid, ev in self.active_event.items()]
        self.active_event.clear()
        return closed


# ========== VISUALIZATION ==========
def annotate_frame(frame: np.ndarray, zone_grid, zone_states: Dict[int, Dict[str, Any]],
                    frame_number: int, timestamp: datetime) -> np.ndarray:
    canvas = frame.copy()
    for zone_id in zone_grid.zone_ids():
        x1, y1, x2, y2 = zone_grid.zone_bbox(zone_id)
        state = zone_states.get(zone_id, {})
        active = state.get('active', False)
        color = (0, 0, 255) if active else (90, 90, 90)
        thickness = 3 if active else 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        z_val = state.get('z', None)
        label = f"Z{zone_id}"
        if z_val is not None:
            label += f" z={_fmt(z_val, '.1f')}"
        cv2.putText(canvas, label, (x1 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        blob = state.get('blob_bbox')
        if active and blob:
            bx1, by1, bx2, by2 = blob
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
            if state.get('object_class'):
                cv2.putText(canvas, state['object_class'], (bx1, max(12, by1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    header = f"Frame {frame_number} | t={timestamp.strftime('%H:%M:%S')} | PS2 ROI/Motion"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 24), (30, 30, 30), -1)
    cv2.putText(canvas, header, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def save_timeline(timeline_data: List[Tuple[datetime, float]], events: List[ROIEvent], out_path: Path):
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available — skipping timeline plot (event log/CSV still generated).")
        return
    if not timeline_data:
        logger.warning("No timeline data collected — skipping timeline plot.")
        return
    times = [t for t, _ in timeline_data]
    vals = [v for _, v in timeline_data]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(times, vals, color="#2b6cb0", linewidth=1)
    ax.fill_between(times, vals, color="#2b6cb0", alpha=0.2)
    for ev in events:
        ax.axvspan(ev.start_timestamp, ev.end_timestamp, color="red", alpha=0.25)
    ax.set_title("Activity Timeline (overall room motion, red = flagged event window)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Total motion magnitude")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Activity timeline saved: {out_path}")


def write_event_report(events: List[ROIEvent], out_dir: Path):
    csv_path = out_dir / "roi_events.csv"
    json_path = out_dir / "roi_events.json"
    rows = []
    for ev in events:
        rows.append({
            "zone_id": ev.zone_id,
            "zone_row": ev.zone_row,
            "zone_col": ev.zone_col,
            "window_start": ev.start_timestamp.strftime("%H:%M:%S"),
            "window_end": ev.end_timestamp.strftime("%H:%M:%S"),
            "duration_sec": round((ev.end_timestamp - ev.start_timestamp).total_seconds(), 2),
            "peak_motion_z": round(ev.peak_motion_z, 3),
            "peak_frame_number": ev.peak_frame_number,
            "blob_bbox": ev.blob_bbox,
            "detected_object": ev.detected_object or "",
            "object_confidence": round(ev.object_confidence, 3) if ev.object_confidence else "",
            "reason": ev.flag_reason,
            "clip_path": ev.clip_path,
            "roi_frame_path": ev.roi_frame_path,
        })
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        with open(csv_path, "w") as f:
            f.write("zone_id,zone_row,zone_col,window_start,window_end,duration_sec,peak_motion_z,"
                     "peak_frame_number,blob_bbox,detected_object,object_confidence,reason,clip_path,roi_frame_path\n")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    logger.info(f"Event report written: {csv_path} ({len(rows)} events) and {json_path}")


def write_intrusion_report(events: List[IntrusionEvent], out_dir: Path) -> int:
    """
    Same spirit as write_event_report above, for private-zone intrusions.
    Only writes the file if there's at least one event.
    Returns the number of rows written (0 if nothing to write / no file created).
    """
    if not events:
        return 0
    rows = [{
        "intruder_person_id": e.intruder_id,
        "zone_owner_person_id": e.zone_owner_id,
        "start_frame": e.start_frame,
        "end_frame": e.end_frame,
        "sustained_frames": e.sustained_frames,
        "peak_blob_bbox": e.peak_blob_bbox,
    } for e in events]
    json_path = out_dir / "private_zone_intrusions.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    logger.info(f"Private zone intrusions logged: {len(rows)} -> {json_path}")
    return len(rows)


# ========== CLIP EXPORT ("keep watching that spot for a bit before/after") ==========
class ClipExporter:
    """
    Buffers recent raw frames so that once an event is confirmed, we can export
    a clip that includes pre-buffer seconds BEFORE the trigger — mirroring how
    an invigilator, once alerted, mentally rewinds to when the behavior started.
    """
    def __init__(self, fps: float, pre_seconds: float, post_seconds: float):
        self.fps = fps
        self.pre_frames = int(pre_seconds * fps)
        self.post_frames = int(post_seconds * fps)
        self.buffer: deque = deque(maxlen=self.pre_frames + 5)
        self.pending: List[Dict[str, Any]] = []  # events waiting for post-buffer frames

    def push_frame(self, frame: np.ndarray, frame_number: int):
        self.buffer.append((frame_number, frame.copy()))

    def start_export(self, event_id: str, zone_bbox: Tuple[int, int, int, int]):
        pre_frames = [f for f in self.buffer]
        self.pending.append({
            'event_id': event_id,
            'zone_bbox': zone_bbox,
            'frames': list(pre_frames),
            'post_needed': self.post_frames,
        })

    def feed_post_frames(self, frame: np.ndarray, frame_number: int):
        still_pending = []
        for job in self.pending:
            job['frames'].append((frame_number, frame.copy()))
            job['post_needed'] -= 1
            if job['post_needed'] <= 0:
                self._write_clip(job)
            else:
                still_pending.append(job)
        self.pending = still_pending

    def flush_all(self):
        for job in self.pending:
            self._write_clip(job)
        self.pending = []

    def _write_clip(self, job: Dict[str, Any]):
        frames = job['frames']
        if not frames:
            return
        x1, y1, x2, y2 = job['zone_bbox']
        h, w = frames[0][1].shape[:2]
        out_path = CLIPS_DIR / f"event_{job['event_id']}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, max(1, self.fps), (w, h))
        for _, f in frames:
            annotated = f.copy()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            writer.write(annotated)
        writer.release()
        job['clip_path'] = str(out_path)


# ========== MAIN PIPELINE ==========
class ROIPipeline:
    def __init__(self):
        # --- primary zone source: per-person (YOLO-driven) or fixed grid ---
        # See PersonZoneProvider for the full rationale. self.zone_grid keeps
        # its original attribute name so every downstream call site
        # (zone_grid.num_zones, zone_grid.zone_bbox(id), zone_grid.zone_row_col(id))
        # needs no changes -- only what kind of object answers those calls changes.
        self.using_person_zones = settings.USE_PERSON_ZONES
        if self.using_person_zones:
            self.zone_grid = PersonZoneProvider(
                settings.RESOLUTION_WIDTH, settings.RESOLUTION_HEIGHT,
                padding_px=settings.PERSON_ZONE_PADDING_PX,
                iou_match_thresh=settings.PERSON_ZONE_IOU_MATCH_THRESH,
                fallback_grid_rows=settings.ZONE_ROWS,
                fallback_grid_cols=settings.ZONE_COLS,
            )
        else:
            self.zone_grid = ZoneGrid(settings.RESOLUTION_WIDTH, settings.RESOLUTION_HEIGHT,
                                       settings.ZONE_ROWS, settings.ZONE_COLS)
        self._last_person_zone_redetect_elapsed = -999.0

        self.motion_estimator = MotionEstimator()
        yolo_path = resolve_model_path(settings.YOLO_MODEL, settings.MODEL_DIR / settings.YOLO_MODEL)
        self.object_extractor = ObjectExtractor(yolo_path)
        self.baseline_engine = ZoneBaselineEngine()
        self.event_tracker = ZoneEventTracker()

        # --- private zone tracking (per-person INTRUSION detection -- see
        # PrivateZoneTracker; this is a SEPARATE tracker instance from the
        # PersonZoneProvider above, even though both are YOLO-driven. That
        # one answers "what are this frame's primary zones"; this one
        # answers "did motion from elsewhere land inside someone else's
        # space". Keeping them separate means either can be toggled off
        # independently via settings without affecting the other, and means
        # this one can keep re-detecting for the whole video even after
        # primary zoning locks (see PERSON_ZONE_LOCK_SECONDS). ---
        self.private_zone_tracker: Optional[PrivateZoneTracker] = None
        if settings.USE_PRIVATE_ZONES:
            self.private_zone_tracker = PrivateZoneTracker(
                padding_px=settings.PRIVATE_ZONE_PADDING_PX,
                sustained_frames_required=settings.INTRUSION_SUSTAINED_FRAMES,
                cooldown_frames=settings.INTRUSION_COOLDOWN_FRAMES,
                frame_w=settings.RESOLUTION_WIDTH,
                frame_h=settings.RESOLUTION_HEIGHT,
            )
        self.intrusion_events: List[IntrusionEvent] = []
        self._last_redetect_elapsed = -999.0

        # --- heatmap (percentile-clipped intensity + presence, see VideoHeatmapAccumulator) ---
        self.heatmap_accumulator = VideoHeatmapAccumulator(
            settings.RESOLUTION_HEIGHT, settings.RESOLUTION_WIDTH,
            presence_thresh=settings.HEATMAP_PRESENCE_THRESH,
        )

        # --- ambient movement filter (teacher/invigilator walking vs seated student gesture) ---
        self.ambient_filter: Optional[AmbientMovementFilter] = None
        if settings.USE_AMBIENT_FILTER:
            if AMBIENT_FILTER_AVAILABLE:
                self.ambient_filter = AmbientMovementFilter()
            else:
                logger.warning("USE_AMBIENT_FILTER is True but AmbientMovementFilter is unavailable -- "
                                "teacher/staff walking motion will NOT be filtered out and may be flagged "
                                "as suspicious.")

        # --- global disturbance detector (room-wide moment vs one student's localized motion) ---
        self.global_disturbance_detector: Optional[GlobalDisturbanceDetector] = None
        self.global_disturbance_events: List[Any] = []
        if settings.USE_GLOBAL_DISTURBANCE_DETECTOR:
            if GLOBAL_DISTURBANCE_AVAILABLE:
                self.global_disturbance_detector = GlobalDisturbanceDetector()
            else:
                logger.warning("USE_GLOBAL_DISTURBANCE_DETECTOR is True but GlobalDisturbanceDetector is "
                                "unavailable -- a room-wide disturbance (announcement, door, noise) may be "
                                "logged as multiple separate per-zone suspicious events instead of one.")

        self.timeline_data: List[Tuple[datetime, float]] = []
        self.events: List[ROIEvent] = []
        self.frame_count = 0
        self.clip_exporter: Optional[ClipExporter] = None
        self._event_seq = 0
        self._start_wall_time: Optional[datetime] = None

    def process_video(self, video_path: Path, max_frames: Optional[int] = None) -> List[ROIEvent]:
        zone_mode_desc = (
            f"per-person (YOLO-driven, locks after {settings.PERSON_ZONE_LOCK_SECONDS}s, "
            f"fallback={settings.ZONE_ROWS}x{settings.ZONE_COLS} grid)"
            if self.using_person_zones else f"fixed grid {settings.ZONE_ROWS}x{settings.ZONE_COLS}"
        )
        logger.info(f"Processing {video_path.name} for offline ROI/motion analysis "
                     f"(zones={zone_mode_desc}, "
                     f"calibration={settings.CALIBRATION_SECONDS}s)")
        if self.object_extractor.fallback:
            logger.warning("Object detection is OFF this run (ultralytics/weights unavailable). "
                            "Motion/ROI/heatmap/timeline/clip analytics are fully real and unaffected.")
            if self.private_zone_tracker is not None:
                logger.warning("Private-zone intrusion detection also needs person detection, so it will "
                                "stay idle this run and the pipeline cleanly falls back to the fixed ZoneGrid only.")
            if self.using_person_zones:
                logger.warning("Per-person zoning also needs person detection, so PRIMARY zoning will run "
                                "on the fixed fallback grid this run too, until YOLO becomes available.")

        annotated_writer = None
        first_ts = None

        frame_gen = sample_frames(video_path, target_fps=settings.FPS, max_frames=max_frames)
        for timestamp, src_frame_idx, frame in frame_gen:
            self.frame_count += 1
            if first_ts is None:
                first_ts = timestamp
                self.clip_exporter = ClipExporter(settings.FPS, settings.CLIP_PRE_BUFFER_SECONDS,
                                                   settings.CLIP_POST_BUFFER_SECONDS)
            elapsed = (timestamp - first_ts).total_seconds()

            self.clip_exporter.push_frame(frame, self.frame_count)
            self.clip_exporter.feed_post_frames(frame, self.frame_count)

            magnitude = self.motion_estimator.extract(frame)
            zone_states: Dict[int, Dict[str, Any]] = {}

            if magnitude is not None:
                # binarize motion for blob-finding (an invigilator doesn't see raw flow
                # vectors, they see "something moved here" as a shape)
                mag_uint8 = np.clip(magnitude * 8, 0, 255).astype(np.uint8)
                _, motion_mask = cv2.threshold(mag_uint8, settings.MOTION_BIN_THRESH, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
                motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)

                # percentile-clipped heatmap accumulator (fixes the old raw-max washout bug)
                self.heatmap_accumulator.add_frame(magnitude)
                self.timeline_data.append((timestamp, float(magnitude.sum())))

                # periodic person re-detection for private zones (handles reseating/drift/
                # occlusion recovery instead of freezing zones after one calibration pass).
                # This tracker is intentionally NEVER locked -- see USE_PRIVATE_ZONES note
                # in Settings and ROIPipeline.__init__ for why intrusion detection needs to
                # keep tracking current bboxes for the whole video.
                if self.private_zone_tracker is not None and not self.object_extractor.fallback:
                    if elapsed - self._last_redetect_elapsed >= settings.PERSON_REDETECT_SECONDS:
                        people = self.object_extractor.detect_people(frame)
                        if people:
                            self.private_zone_tracker.update_person_boxes(people, self.frame_count)
                        self._last_redetect_elapsed = elapsed

                # periodic person re-detection for PRIMARY zoning (PersonZoneProvider),
                # on its own faster cadence -- a stale/missed detection here means a
                # stale/missed PRIMARY zone, not just a secondary intrusion check, so
                # this refreshes more often than the private-zone-intrusion redetect
                # above. Reuses the same YOLO person-detector call, just a separate
                # tracked-id registry (see PersonZoneProvider docstring for why they're
                # kept as two independent trackers rather than sharing one).
                #
                # LOCK-AFTER-CALIBRATION: once `elapsed` passes PERSON_ZONE_LOCK_SECONDS,
                # lock() is called exactly once and re-detection for PRIMARY zoning stops
                # entirely -- zone ids/boundaries are frozen for the rest of the video, so
                # every zone_id's baseline/event-tracker/ambient-filter history stays
                # attached to the same physical seat/person for the whole run instead of
                # silently resetting on an IoU-match miss. See PersonZoneProvider module
                # docstring ("LOCK-AFTER-CALIBRATION") for the full rationale.
                if self.using_person_zones and not self.object_extractor.fallback:
                    if not self.zone_grid.is_locked:
                        if elapsed >= settings.PERSON_ZONE_LOCK_SECONDS:
                            self.zone_grid.lock()
                            logger.info(f"Person zones LOCKED at t={elapsed:.1f}s with "
                                        f"{self.zone_grid.num_zones} zones -- grid is now fixed for the rest of the video.")
                        elif elapsed - self._last_person_zone_redetect_elapsed >= settings.PERSON_ZONE_REDETECT_SECONDS:
                            zone_people = self.object_extractor.detect_people(frame)
                            if zone_people:
                                self.zone_grid.update_person_boxes(zone_people, self.frame_count)
                            self._last_person_zone_redetect_elapsed = elapsed
                if self.using_person_zones:
                    # Snapshot this frame's tracked-person-id ordering ONCE, before any
                    # zone_bbox()/num_zones calls below -- PASS 1 and PASS 2 both loop
                    # zone_id in range(num_zones) and must agree on what each zone_id
                    # refers to for the whole frame. Falls back to the fixed grid
                    # automatically if no people are currently tracked yet. No-op once
                    # locked (see PersonZoneProvider.refresh_frame_ordering).
                    self.zone_grid.refresh_frame_ordering()

                # ---- PASS 1: per-zone raw deviance, blob detection, ambient filtering ----
                # (must fully finish across ALL zones before we can ask "how many zones
                # are deviant THIS frame", which is what the global disturbance check needs)
                zone_pass1: Dict[int, Dict[str, Any]] = {}
                deviant_zones_this_frame: List[int] = []

                # Full-frame blob detection ONCE per analyzed frame (not per zone) -- see
                # the note above find_all_blobs/find_dominant_blob_for_zone for why: a
                # blob straddling a zone boundary (e.g. an invigilator walking across the
                # grid line) must never be truncated to whichever zone happens to be
                # asking, or its true tall/full-height shape gets lost and the ambient
                # filter can't recognize it as a walking person.
                all_blobs_this_frame = find_all_blobs(motion_mask, settings.MIN_BLOB_AREA)

                for zone_id in self.zone_grid.zone_ids():
                    zx1, zy1, zx2, zy2 = self.zone_grid.zone_bbox(zone_id)
                    zone_motion_sum = float(magnitude[zy1:zy2, zx1:zx2].sum())

                    if elapsed <= settings.CALIBRATION_SECONDS:
                        self.baseline_engine.update(zone_id, zone_motion_sum)

                    baseline = self.baseline_engine.get_baseline(zone_id)
                    z = 0.0
                    is_deviant = False
                    if baseline is not None:
                        z = max(0.0, (zone_motion_sum - baseline.mean) / baseline.std)
                        # TWO-PART GATE (see ZoneBaselineEngine's v3 FIX HISTORY docstring):
                        # relative (z-score) AND absolute (clears the calibration window's
                        # own p90 ceiling by a fixed margin) must BOTH hold. Relative alone
                        # let a wide-variance zone's own spread silently raise the bar past
                        # what any real anomaly would reach (the v2 bug this replaces);
                        # absolute alone would flag a zone whose whole baseline already
                        # sits close to its own ceiling. baseline.ceiling is the p90 value
                        # ZoneBaselineEngine.get_baseline() already computes every frame --
                        # it just wasn't being read anywhere until now.
                        relative_gate = z > settings.ZONE_Z_THRESHOLD
                        absolute_gate = zone_motion_sum > (baseline.ceiling * settings.ZONE_ABS_CEILING_MULT)
                        is_deviant = relative_gate and absolute_gate

                    blob_bbox = None
                    if is_deviant:
                        blob_bbox = find_dominant_blob_for_zone(all_blobs_this_frame, (zx1, zy1, zx2, zy2))
                        if blob_bbox is None:
                            is_deviant = False  # deviant score but no coherent blob -> likely noise, an invigilator wouldn't flag it

                    # ambient movement filter: is this blob a walking/standing person
                    # (teacher doing rounds) rather than a seated student's localized
                    # gesture? Uses blob shape (tall, near full zone-height) + cross-frame
                    # centroid translation, cross-checked against detected person boxes
                    # when private-zone tracking is on.
                    if is_deviant and blob_bbox is not None and self.ambient_filter is not None:
                        person_bboxes = None
                        seated_zones = None
                        if self.private_zone_tracker is not None and self.private_zone_tracker.people:
                            person_bboxes = [p.bbox for p in self.private_zone_tracker.people.values()]
                            seated_zones = [p.private_zone for p in self.private_zone_tracker.people.values()]
                        ambient_result = self.ambient_filter.classify(
                            zone_id=zone_id, blob_bbox=blob_bbox, zone_bbox=(zx1, zy1, zx2, zy2),
                            person_bboxes=person_bboxes, seated_private_zones=seated_zones,
                        )
                        if ambient_result['is_sustained_ambient']:
                            is_deviant = False  # walking/standing person, not a localized gesture -- suppress

                    if is_deviant:
                        deviant_zones_this_frame.append(zone_id)

                    zone_pass1[zone_id] = {
                        'zone_bbox': (zx1, zy1, zx2, zy2),
                        'z': z,
                        'is_deviant': is_deviant,
                        'blob_bbox': blob_bbox,
                    }

                # ---- global disturbance check for THIS frame, using the deviant set we
                # just finished computing above (not last frame's stale set) ----
                if self.global_disturbance_detector is not None:
                    gd_action = self.global_disturbance_detector.observe(
                        deviant_zones_this_frame, self.zone_grid.num_zones, timestamp)
                    if gd_action == 'start':
                        logger.warning(f"GLOBAL DISTURBANCE START at t={timestamp.strftime('%H:%M:%S')} "
                                        f"| zones involved so far: {deviant_zones_this_frame}")
                    closed_gd = self.global_disturbance_detector.close_stale(timestamp)
                    if closed_gd is not None:
                        self.global_disturbance_events.append(closed_gd)
                        logger.warning(f"GLOBAL DISTURBANCE CLOSED: {closed_gd.start_timestamp.strftime('%H:%M:%S')} - "
                                        f"{closed_gd.end_timestamp.strftime('%H:%M:%S')} | peak "
                                        f"{closed_gd.peak_zone_count}/{closed_gd.total_zones} zones")

                # ---- PASS 2: per-zone suppression + everything else that was already here ----
                for zone_id in self.zone_grid.zone_ids():
                    zx1, zy1, zx2, zy2 = zone_pass1[zone_id]['zone_bbox']
                    z = zone_pass1[zone_id]['z']
                    is_deviant = zone_pass1[zone_id]['is_deviant']
                    blob_bbox = zone_pass1[zone_id]['blob_bbox']

                    # private-zone intrusion check: does this blob land inside a private
                    # zone that isn't its owner's own? owner_hint is None here (we don't
                    # try to map fixed zone_id -> nearest person_id), so the tracker falls
                    # back to its own "blob center outside the zone owner's own bbox"
                    # heuristic.
                    if is_deviant and blob_bbox is not None and self.private_zone_tracker is not None:
                        intrusion_action = self.private_zone_tracker.check_intrusion(
                            blob_bbox, self.frame_count, owner_hint=None)
                        if intrusion_action == 'start':
                            logger.warning(f"PRIVATE ZONE INTRUSION at frame {self.frame_count}, "
                                            f"t={timestamp.strftime('%H:%M:%S')}, blob={blob_bbox}")

                    # global disturbance suppression: is the WHOLE ROOM reacting to
                    # something (announcement, door, noise) right now, and is THIS zone
                    # currently co-occurring with the rest of that room-wide moment? If
                    # so, don't let it open/extend a per-zone suspicious event. This is
                    # evaluated per-zone, per-frame (not a blanket switch) so a genuine
                    # localized event that happens to briefly coincide with an unrelated
                    # multi-zone blip stops being suppressed the instant it's the only
                    # zone still deviant.
                    if is_deviant and self.global_disturbance_detector is not None:
                        if self.global_disturbance_detector.should_suppress_zone(zone_id):
                            is_deviant = False

                    action = self.event_tracker.observe(zone_id, is_deviant, z, timestamp, self.frame_count, blob_bbox)
                    if action == 'start':
                        self._event_seq += 1
                        event_id = f"{self._event_seq:04d}_z{zone_id}"
                        zbbox = blob_bbox or (zx1, zy1, zx2, zy2)
                        self.clip_exporter.start_export(event_id, zbbox)
                        self.event_tracker.active_event[zone_id]['event_id'] = event_id
                        row, col = self.zone_grid.zone_row_col(zone_id)
                        zone_desc = f"person_id={row}" if self.using_person_zones and not self.zone_grid.using_fallback_grid else f"row={row}, col={col}"
                        logger.warning(f"ROI EVENT START: Zone {zone_id} ({zone_desc}) at {timestamp.strftime('%H:%M:%S')} "
                                        f"| z={_fmt(z)} | frame={self.frame_count}")

                    object_class = None
                    object_conf = None
                    if is_deviant and blob_bbox is not None and not self.object_extractor.fallback:
                        dets = self.object_extractor.extract(frame, region=blob_bbox)
                        if dets:
                            best = max(dets, key=lambda d: d['confidence'])
                            object_class = best['class']
                            object_conf = best['confidence']
                            if zone_id in self.event_tracker.active_event:
                                self.event_tracker.active_event[zone_id]['object_class'] = object_class
                                self.event_tracker.active_event[zone_id]['object_confidence'] = object_conf

                    zone_states[zone_id] = {
                        'active': zone_id in self.event_tracker.active_event,
                        'z': z,
                        'blob_bbox': blob_bbox,
                        'object_class': object_class,
                    }

                closed = self.event_tracker.close_stale_events(timestamp)
                for zone_id, ev_data in closed:
                    self._finalize_event(zone_id, ev_data, frame)
                    if self.ambient_filter is not None:
                        self.ambient_filter.reset_zone(zone_id)  # don't let stale centroid history leak into the next event

                if self.private_zone_tracker is not None:
                    closed_intrusions = self.private_zone_tracker.close_stale(self.frame_count)
                    self.intrusion_events.extend(closed_intrusions)

            annotated = annotate_frame(frame, self.zone_grid, zone_states, self.frame_count, timestamp)
            if self.frame_count % max(1, int(settings.FPS)) == 0:  # save ~1 frame/sec of annotated context
                roi_frame_path = ROI_FRAMES_DIR / f"frame_{self.frame_count:05d}_{timestamp.strftime('%H%M%S')}.jpg"
                cv2.imwrite(str(roi_frame_path), annotated)

            if annotated_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                annotated_writer = cv2.VideoWriter(str(settings.OUTPUT_DIR / "annotated_roi_run.mp4"),
                                                    fourcc, max(1, settings.FPS),
                                                    (annotated.shape[1], annotated.shape[0]))
            annotated_writer.write(annotated)

            if self.frame_count % 50 == 0:
                logger.info(f"Progress: {self.frame_count} analyzed frames, Events so far: {len(self.events)}")

        # close out anything still active at end of video
        for zone_id, ev_data in self.event_tracker.close_all():
            self._finalize_event(zone_id, ev_data, None)
        if self.private_zone_tracker is not None:
            self.intrusion_events.extend(self.private_zone_tracker.close_all(self.frame_count))
        if self.global_disturbance_detector is not None:
            final_gd = self.global_disturbance_detector.close_all()
            if final_gd is not None:
                self.global_disturbance_events.append(final_gd)
        if self.clip_exporter is not None:
            self.clip_exporter.flush_all()

        if annotated_writer is not None:
            annotated_writer.release()
            logger.info(f"Annotated ROI video saved: {settings.OUTPUT_DIR / 'annotated_roi_run.mp4'}")

        intensity_path, presence_path = self.heatmap_accumulator.save(
            HEATMAP_DIR, prefix="motion_heatmap",
            clip_percentile=settings.HEATMAP_CLIP_PERCENTILE)
        logger.info(f"Motion heatmaps saved: {intensity_path}, {presence_path}")

        zone_bboxes = [self.zone_grid.zone_bbox(z) for z in self.zone_grid.zone_ids()]
        zone_heat_summary = self.heatmap_accumulator.zone_summary(zone_bboxes)
        with open(settings.OUTPUT_DIR / "zone_heat_summary.json", "w") as f:
            json.dump(zone_heat_summary, f, indent=2)

        save_timeline(self.timeline_data, self.events, settings.OUTPUT_DIR / "activity_timeline.png")
        write_event_report(self.events, settings.OUTPUT_DIR)
        write_intrusion_report(self.intrusion_events, settings.OUTPUT_DIR)

        if self.global_disturbance_events:
            gd_path = settings.OUTPUT_DIR / "global_disturbances.json"
            with open(gd_path, "w") as f:
                json.dump([e.to_dict() for e in self.global_disturbance_events], f, indent=2)
            logger.info(f"Global disturbances logged: {len(self.global_disturbance_events)} -> {gd_path}")

        logger.info(f"Done. Total analyzed frames: {self.frame_count}, ROI Events: {len(self.events)}, "
                     f"Private zone intrusions: {len(self.intrusion_events)}, "
                     f"Global disturbances: {len(self.global_disturbance_events)}")
        return self.events

    def _finalize_event(self, zone_id: int, ev_data: Dict[str, Any], current_frame: Optional[np.ndarray]):
        row, col = self.zone_grid.zone_row_col(zone_id)
        if self.using_person_zones and not self.zone_grid.using_fallback_grid:
            reasons = [f"sustained motion for tracked person {zone_id}"]
        else:
            reasons = [f"sustained motion in zone {zone_id} (row {row}, col {col})"]
        if ev_data.get('object_class'):
            reasons.append(f"object detected: {ev_data['object_class']}")

        roi_frame_path = ""
        if current_frame is not None:
            x1, y1, x2, y2 = ev_data['blob_bbox']
            snapshot = current_frame.copy()
            cv2.rectangle(snapshot, (x1, y1), (x2, y2), (0, 0, 255), 3)
            roi_frame_path = str(ROI_FRAMES_DIR / f"event_{ev_data.get('event_id','unknown')}_peak.jpg")
            cv2.imwrite(roi_frame_path, snapshot)

        event = ROIEvent(
            zone_id=zone_id,
            zone_row=row,
            zone_col=col,
            start_timestamp=ev_data['start'] - timedelta(seconds=settings.CLIP_PRE_BUFFER_SECONDS),
            end_timestamp=ev_data['last_seen'] + timedelta(seconds=settings.CLIP_POST_BUFFER_SECONDS),
            peak_motion_z=ev_data['peak_motion_z'],
            peak_frame_number=ev_data['peak_frame_number'],
            blob_bbox=ev_data['blob_bbox'],
            detected_object=ev_data.get('object_class'),
            object_confidence=ev_data.get('object_confidence'),
            flag_reason="; ".join(reasons),
            roi_frame_path=roi_frame_path,
        )
        # clip path gets written by ClipExporter once post-buffer completes;
        # look it up by matching event_id if available at report-write time.
        event_id = ev_data.get('event_id', '')
        event.clip_path = str(CLIPS_DIR / f"event_{event_id}.mp4") if event_id else ""
        self.events.append(event)
        logger.warning(f"ROI EVENT CLOSED: Zone {zone_id} | {event.start_timestamp.strftime('%H:%M:%S')} - "
                        f"{event.end_timestamp.strftime('%H:%M:%S')} | peak_z={_fmt(event.peak_motion_z)} | "
                        f"reason: {event.flag_reason}")


# ========== CLI ==========
def main():
    parser = argparse.ArgumentParser(description="DRISHTI-PS2 — Offline ROI Detection & Video Segmentation via Motion Estimation")
    parser.add_argument("--video", default="auto", help="Path to video file, or 'auto' to auto-detect on Kaggle")
    parser.add_argument("--output", default=None, help="Output directory override")
    parser.add_argument("--max-frames", type=int, default=None, help="Process only N sampled frames (debug)")
    args, _ = parser.parse_known_args()

    if args.output:
        settings.OUTPUT_DIR = Path(args.output)
        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        global ROI_FRAMES_DIR, CLIPS_DIR, HEATMAP_DIR
        ROI_FRAMES_DIR = settings.OUTPUT_DIR / "roi_frames"
        CLIPS_DIR = settings.OUTPUT_DIR / "event_clips"
        HEATMAP_DIR = settings.OUTPUT_DIR / "heatmaps"
        ROI_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    video_path = resolve_video_path(TEST_VIDEO_OVERRIDE if TEST_VIDEO_OVERRIDE else args.video)
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        if _is_kaggle():
            logger.error("On Kaggle: check that your video dataset is attached under Input. Listing /kaggle/input:")
            for p in Path("/kaggle/input").rglob("*"):
                logger.error(f"  {p}")
        sys.exit(1)

    logger.info(f"Resolved video path: {video_path}")
    pipeline = ROIPipeline()
    events = pipeline.process_video(video_path, max_frames=args.max_frames)

    zone_mode_summary = (
        f"per-person (YOLO-driven)" if pipeline.using_person_zones else "fixed grid"
    )
    print("\n" + "=" * 60)
    print("DRISHTI-PS2 — ROI / Motion Segmentation Summary")
    print("=" * 60)
    print(f"Video: {video_path.name}")
    print(f"Zone mode: {zone_mode_summary} (fallback grid: {settings.ZONE_ROWS} x {settings.ZONE_COLS})")
    print(f"Object detection: {'ON' if not pipeline.object_extractor.fallback else 'OFF (weights/lib unavailable)'}")
    print(f"Private zones: {'ON' if pipeline.private_zone_tracker is not None else 'OFF'}"
          + (f" (padding={settings.PRIVATE_ZONE_PADDING_PX}px, re-detect every {settings.PERSON_REDETECT_SECONDS}s)"
             if pipeline.private_zone_tracker is not None else ""))
    print(f"Frames analyzed: {pipeline.frame_count}")
    print(f"ROI Events Detected: {len(events)}")
    for i, ev in enumerate(events, 1):
        print(f"\nEvent {i}:")
        print(f"  Zone: {ev.zone_id} (row {ev.zone_row}, col {ev.zone_col})")
        print(f"  Time: {ev.start_timestamp.strftime('%H:%M:%S')} - {ev.end_timestamp.strftime('%H:%M:%S')}")
        print(f"  Peak motion z-score: {_fmt(ev.peak_motion_z)}")
        print(f"  Reason: {ev.flag_reason}")
        if ev.detected_object:
            print(f"  Object: {ev.detected_object} (conf: {_fmt(ev.object_confidence)})")
        if ev.clip_path:
            print(f"  Clip: {ev.clip_path}")
    print(f"\nPrivate Zone Intrusions Detected: {len(pipeline.intrusion_events)}")
    for i, iv in enumerate(pipeline.intrusion_events, 1):
        print(f"  Intrusion {i}: person {iv.intruder_id} -> person {iv.zone_owner_id}'s zone "
              f"| frames {iv.start_frame}-{iv.end_frame} ({iv.sustained_frames} frames) | blob={iv.peak_blob_bbox}")
    print("\nOutput files:")
    print(f"  Annotated ROI video   : {settings.OUTPUT_DIR / 'annotated_roi_run.mp4'}")
    print(f"  Motion heatmap        : {HEATMAP_DIR / 'motion_heatmap_intensity.jpg'}")
    print(f"  Motion presence map   : {HEATMAP_DIR / 'motion_heatmap_presence.jpg'}")
    print(f"  Zone heat summary     : {settings.OUTPUT_DIR / 'zone_heat_summary.json'}")
    print(f"  Activity timeline     : {settings.OUTPUT_DIR / 'activity_timeline.png'}")
    print(f"  Event clips dir       : {CLIPS_DIR}/")
    print(f"  ROI frames dir        : {ROI_FRAMES_DIR}/")
    print(f"  Event CSV             : {settings.OUTPUT_DIR / 'roi_events.csv'}")
    print(f"  Event JSON            : {settings.OUTPUT_DIR / 'roi_events.json'}")
    if pipeline.intrusion_events:
        print(f"  Private zone intrusions: {settings.OUTPUT_DIR / 'private_zone_intrusions.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
