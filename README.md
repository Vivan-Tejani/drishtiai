# DRISHTI-PS2 — Results

Offline motion-estimation pipeline for exam-hall surveillance footage (Problem Statement 2, Drishti AI Hackathon 2026).
Team DI2_04 — The Fellowship of the Ring.

Full pipeline code: [`pipeline_final.py`](./pipeline_final.py)

All numbers, images, and clips below are from one actual run (`recording.mov`), pulled directly from `roi_events.csv`, `private_zone_intrusions.json`, and the generated heatmaps/frames/clips — not from the pitch deck.

---

## Run Summary

| | |
|---|---|
| Frames analyzed | 535 |
| ROI events flagged | **4** |
| Private zone intrusions | **3** (all 1-frame, none sustained ≥10 frames) |
| Objects detected (phone/book) | **0** — no event in this run had a corroborating object |
| Zones tracked | 13 (per-person, YOLO-driven) |
| Ground-truth precision/recall | Not available — no manually annotated clips yet |

---

## 1. Live Zone Grid (mid-run annotated frame)

Each box is one tracked person's own zone (per-person zoning, not a fixed grid) — this is the ordinary, un-flagged state most of the video is in.

![zone grid](results/sample_zone_grid_frame.jpg)

Full annotated video: [`results/annotated_roi_run.mp4`](results/annotated_roi_run.mp4)

---

## 2. Flagged Events — Peak Frames

All 4 flagged events from this run, in order. Red box = the blob that triggered the flag.

### Event 1 — Zone 0, peak z-score 2.77, 00:00:04–00:00:10
![event 1](results/event_0001_z0_peak.jpg)
Clip: [`results/event_0001_z0.mp4`](results/event_0001_z0.mp4)

### Event 2 — Zone 10, peak z-score 3.29, 00:00:11–00:00:17
![event 2](results/event_0002_z10_peak.jpg)
Clip: [`results/event_0002_z10.mp4`](results/event_0002_z10.mp4)

### Event 3 — Zone 6, peak z-score 9.06 (highest in this run), 00:00:16–00:00:23
![event 3](results/event_0003_z6_peak.jpg)
Clip: [`results/event_0003_z6.mp4`](results/event_0003_z6.mp4)

### Event 4 — Zone 7, peak z-score 3.41, 00:01:26–00:01:33
![event 4](results/event_0004_z7_peak.jpg)
Clip: [`results/event_0004_z7.mp4`](results/event_0004_z7.mp4)

None of these 4 events had a corroborating object detection (phone/book) — all four were flagged on sustained motion alone.

---

## 3. Motion Heatmaps

99th-percentile-clipped, so one outlier frame doesn't dominate the whole map.

**Intensity** (how strong motion was, per pixel, across the whole run):
![intensity heatmap](results/motion_heatmap_intensity.jpg)

**Presence** (how often a pixel had active motion, regardless of strength):
![presence heatmap](results/motion_heatmap_presence.jpg)

---

## 4. Activity Timeline

Total motion magnitude across the room over time. The 4 flagged windows above sit inside the visible peaks.

![activity timeline](results/activity_timeline.png)

---

## 5. Raw Event Log

`results/roi_events.csv` — every flagged event this run, unedited:

| Zone | Window | Duration | Peak z-score | Reason |
|---|---|---|---|---|
| 0 | 00:00:04–00:00:10 | 6.0s | 2.77 | sustained motion for tracked person 0 |
| 10 | 00:00:11–00:00:17 | 6.0s | 3.288 | sustained motion for tracked person 10 |
| 6 | 00:00:16–00:00:23 | 7.0s | 9.057 | sustained motion for tracked person 6 |
| 7 | 00:01:26–00:01:33 | 7.8s | 3.405 | sustained motion for tracked person 7 |

`results/private_zone_intrusions.json` — 3 intrusions logged, all single-frame (frames 35, 87, 94), none sustained. The pipeline's own intrusion tracker requires `INTRUSION_SUSTAINED_FRAMES` consecutive frames to escalate — these three didn't clear that bar, so they're logged as candidate intrusions, not confirmed ones.

---

## What This Run Does *Not* Show

Being direct about what's actually in this data, not what the pitch says the system can do:

- **No object detections fired** in any of the 4 events — this run has no evidence for the phone/chit-detection layer beyond "it ran and found nothing," since no video content actually had a phone or book in frame.
- **No global disturbance was flagged** — nothing to show for that feature in this particular run.
- **All 3 intrusions were 1-frame candidates**, not the "sustained, high-confidence" intrusions the pitch deck describes — this run doesn't have an example of a sustained intrusion event.
- **No ground-truth labels exist** for this clip, so there's no real precision/recall number to report — the z-scores and timestamps above are the pipeline's own output, not validated against a human-annotated answer key.

---

## Output Folder Reference

```
output_ps2/
├── annotated_roi_run.mp4
├── roi_frames/              # ~1 frame/sec + one peak frame per event
├── event_clips/              # one .mp4 per flagged event, pre/post buffered
├── heatmaps/
│   ├── motion_heatmap_intensity.jpg
│   └── motion_heatmap_presence.jpg
├── zone_heat_summary.json
├── activity_timeline.png
├── roi_events.csv / roi_events.json
├── private_zone_intrusions.json
└── global_disturbances.json  # not generated this run — nothing was flagged
```
