# Physics-Informed 3D Human Pose Estimation

A Python pipeline that takes a video of a human, detects 3D body keypoints using MediaPipe, applies biomechanical constraints, and outputs a physically valid 3D pose sequence.

---

## Demo

> 📽️ *(Demo GIF/screenshot will go here once the pipeline is built)*

---

## Features

- 🎥 Video input support (`.mp4`, `.avi`)
- 🦴 33-joint 3D keypoint detection via MediaPipe Pose
- 📐 Real-world metric scale recovery
- ✅ Biomechanical bone length + joint angle validation
- 🌊 Physics-based temporal smoothing
- 💾 Checkpoint system — never reprocess expensive steps
- 📊 Exports to JSON, CSV, and annotated video

---

## Installation

**Requirements:** Python 3.9+

```bash
# Clone the repo
git clone https://github.com/yourusername/pose-estimation.git
cd pose-estimation

# Install dependencies
pip install numpy opencv-python mediapipe scipy open3d
```

---

## Usage

```bash
python run.py --input data/raw/my_video.mp4
```

**Optional flags:**
```bash
--output   Path to output directory (default: data/outputs/)
--height   Subject reference height in meters (default: 1.7)
--force    Force rerun all steps, ignore checkpoints
```

**Outputs generated:**
```
data/outputs/
├── annotated/          # 2D skeleton overlaid on original video
├── poses_3d/           # 3D keypoint data (JSON + CSV)
└── visualizations/     # 3D animated pose render
```

---

## Project Structure

```
pose_estimation/
│
├── data/
│   ├── raw/                  # Input videos
│   └── outputs/
│       ├── annotated/        # 2D annotated videos
│       ├── poses_3d/         # Saved 3D keypoint data (checkpoints)
│       └── visualizations/   # 3D pose renders
│
├── config.py                 # All parameters and paths
├── camera_model.py           # Pinhole camera math
├── video_loader.py           # Video I/O and frame streaming
├── pose_detector.py          # MediaPipe 3D keypoint detection
├── keypoint_utils.py         # Keypoint cleaning and interpolation
├── scale_recovery.py         # Normalized → metric space conversion
├── pose_visualizer_2d.py     # 2D skeleton annotation
├── bone_constraints.py       # Bone length validation
├── joint_angle_validator.py  # Joint angle validation
├── physics_smoother.py       # Temporal smoothing
├── pose_visualizer_3d.py     # 3D pose rendering (Open3D)
├── exporter.py               # Output serialization
└── run.py                    # Pipeline entry point
```

---

## Pipeline

### How a video flows through the system:

```
my_video.mp4
      ↓
[run.py checks checkpoints]
      ↓
video_loader        →  frame generator + metadata
      ↓
pose_detector       →  (N, 33, 4) raw         ← CHECKPOINT
      ↓
keypoint_utils      →  (N, 33, 4) cleaned
      ↓
scale_recovery      →  (N, 33, 4) in meters   ← CHECKPOINT
      ↓
pose_visualizer_2d  →  annotated .mp4          ← SIDE OUTPUT
      ↓
bone_constraints
joint_angle_validator → (N, 33, 4) validated
      ↓
physics_smoother    →  (N, 33, 4) smooth       ← CHECKPOINT
      ↓
pose_visualizer_3d  →  3D animation .mp4
      ↓
exporter            →  JSON, CSV, summary
```

### Array shape at each stage:

| Stage | Shape | Notes |
|---|---|---|
| Raw detection | `(N, 33, 4)` | x, y, z, confidence — normalized |
| After cleaning | `(N, 33, 4)` | gaps filled via linear interpolation |
| After scale recovery | `(N, 33, 4)` | converted to meters |
| After physics | `(N, 33, 4)` | anatomically valid + temporally smooth |

N = total frames, 33 = MediaPipe joints, 4 = (x, y, z, confidence)

---

## File-by-File Breakdown

### `config.py`
Central configuration. All tunable parameters, paths, and flags. No hardcoded values anywhere else in the project.

---

### `camera_model.py`
Implements the pinhole camera model mathematically.
- `build_intrinsic_matrix(f, cx, cy)` → K matrix
- `project_3d_to_2d(point_3d, K, R, t)` → pixel coordinates
- `unproject_2d_to_ray(point_2d, K)` → 3D ray from pixel

---

### `video_loader.py`
Handles all video I/O via OpenCV. Uses a generator to stream frames one at a time — never loads the full video into RAM.
- `frame_generator(video)` → yields frames one by one
- `get_video_metadata(path)` → fps, width, height, frame count

---

### `pose_detector.py`
Core detection module. Runs MediaPipe Pose per frame and returns native 3D keypoints. MediaPipe outputs X, Y, Z natively — no separate lifting step needed.
- `detect_video(video_generator, detector)` → `(N, 33, 4)`

---

### `keypoint_utils.py`
Pure utility functions for keypoint manipulation.
- `filter_low_confidence(kps, threshold)` → masks unreliable joints
- `interpolate_missing(kps_sequence)` → linear interpolation for gaps
- `compute_joint_angles(kps)` → angle at each joint in degrees

---

### `scale_recovery.py`
Converts MediaPipe's normalized coordinates (0.0–1.0) to real-world meters. Required before any physics constraints can be applied.
- `compute_scale_factor(normalized_height, reference_height_m)` → scale ratio
- `apply_scale(kps_sequence, scale_factor)` → metric-space keypoints

---

### `pose_visualizer_2d.py`
Draws 2D skeleton annotations on original video frames and saves as `.mp4`. Side output for visual sanity checking — does not feed into physics pipeline.
- `annotate_video(video_path, keypoints_sequence, output_path)`

---

### `bone_constraints.py`
Validates and corrects bone lengths against anthropometric priors (in meters).
- `check_bone_lengths(pose_3d, priors)` → flags violations
- `enforce_bone_lengths(pose_3d, priors)` → corrects violations

---

### `joint_angle_validator.py`
Checks joint angles against anatomical limits (e.g. knee: 0–160°, elbow: 0–145°).
- `validate_all_joints(pose_3d)` → dict of joint → angle
- `clamp_joint_angles(pose_3d, limits)` → corrects impossible angles

---

### `physics_smoother.py`
Temporal smoothing using physics-based motion priors. Removes jitter and enforces realistic velocity and acceleration.
- `kalman_smooth(pose_sequence)` → Kalman filter per joint
- `velocity_filter(pose_sequence, max_velocity)` → constrains joint speed

---

### `pose_visualizer_3d.py`
Renders the final 3D pose sequence using **Open3D** (not Matplotlib — avoids memory leaks on long videos).
- `animate_pose_sequence(pose_sequence, output_path)` → saves 3D animation

---

### `exporter.py`
Saves all outputs in structured formats.
- `export_keypoints_json(kps_sequence, path)`
- `export_keypoints_csv(kps_sequence, path)`
- `export_summary_report(metadata, path)`

---

### `run.py`
Top-level entry point. Wires the full pipeline with a checkpoint system — skips expensive steps if outputs already exist on disk.

---

## Checkpoint System

On every run, `run.py` checks for existing outputs before executing each stage:

| Checkpoint File | Skips |
|---|---|
| `poses_3d_raw.json` | MediaPipe detection (most expensive) |
| `poses_3d_metric.json` | Scale recovery |
| `poses_3d_final.json` | Physics validation + smoothing |

To force a full rerun:
```bash
python run.py --input data/raw/my_video.mp4 --force
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array math throughout |
| `opencv-python` | Video I/O, 2D visualization |
| `mediapipe` | 3D keypoint detection |
| `scipy` | Signal filtering, Kalman smoother |
| `open3d` | 3D visualization (memory-safe) |

---

## Roadmap

- [x] Architecture design
- [ ] `camera_model.py`
- [ ] `video_loader.py`
- [ ] `pose_detector.py`
- [ ] `keypoint_utils.py`
- [ ] `scale_recovery.py`
- [ ] `pose_visualizer_2d.py`
- [ ] `bone_constraints.py`
- [ ] `joint_angle_validator.py`
- [ ] `physics_smoother.py`
- [ ] `pose_visualizer_3d.py`
- [ ] `exporter.py`
- [ ] `run.py`
- [ ] End-to-end test on sample video

---

## License

MIT License — feel free to use, modify, and build on this project.
