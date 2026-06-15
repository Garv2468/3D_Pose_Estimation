# 3D Pose Estimation Pipeline: Future Improvements Roadmap

This document outlines structural, algorithmic, and logical enhancements to optimize the processing pipeline, ensure mathematical validity, and enforce robust Object-Oriented Programming (OOP) paradigms.

---

## 1. Structural & OOP Enhancements

### 1.1 Context Managers for Resource Handling
Currently, `VideoLoader` relies on manual execution of `self.release()`. In production environments, if the pipeline crashes midway, the video file remains locked in memory. 

**Improvement:** Implement Python "dunder" methods to turn `VideoLoader` into a context manager. This guarantees resource cleanup.

```python
class VideoLoader:
    def __init__(self, path: str):
        self.path = path
        self.capture = None
        self.metaData = None
        self.capture_video()
        self.load_metadata()

    # Allows usage of 'with VideoLoader(path) as loader:'
    def __enter__(self):
        return self

    # Automatically executes upon exiting the 'with' block, even on error
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
```

### 1.2 Decoupling Extraction from Masking
The `PoseDetector` currently assigns `np.nan` during the frame extraction loop based on `threshold_visibility`. This permanently mutates the raw data before it is saved, destroying the integrity of the checkpoint system.

**Improvement:** `PoseDetector` should extract pure, unfiltered data. Masking should be a vectorized operation applied *after* extraction.

```python
# Future implementation in keypoint_utils.py
def mask_low_confidence(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Vectorized masking of low-confidence coordinates."""
    df_out = df.copy()
    for point in MEDIAPIPE_POINTS:
        # Create a boolean mask where visibility is below threshold
        mask = df_out[f'{point}.visibility'] < threshold
        # Instantly set x, y, z to NaN for those specific rows
        df_out.loc[mask, [f'{point}.x', f'{point}.y', f'{point}.z']] = np.nan
    return df_out
```

---

## 2. Algorithmic & Mathematical Improvements

### 2.1 Preserving DataFrame Schema During Interpolation
The current `interpolate_keypoints` function drops the visibility columns entirely, which will cause `KeyError` exceptions in downstream modules.

**Improvement:** Isolate coordinate columns for interpolation while preserving the visibility scores.

```python
def interpolate_keypoints(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Provided DataFrame is empty.")
    
    df_out = df.copy()
    # Dynamically select only coordinate columns
    coord_cols = [col for col in df_out.columns if 'visibility' not in col]
    
    # Apply interpolation strictly to the spatial geometry
    df_out[coord_cols] = df_out[coord_cols].interpolate(method='linear', limit_direction='both')
    return df_out
```

### 2.2 Advanced Kinematic Smoothing
Linear interpolation assumes constant velocity between missing frames, which violates human biomechanics (limbs accelerate and decelerate non-linearly). 

**Improvement:** Replace linear interpolation with a Savitzky-Golay filter or a discrete Kalman filter [1]. A Kalman filter estimates the true state of a joint by minimizing the mean square error, represented by the state transition equation:

`x_k = A * x_{k-1} + B * u_k + w_k`

This will result in biologically fluid 3D animations rather than robotic, linear snapping.

---

## 3. Logical Corrections & The Scale Dilemma

### 3.1 The Danger of Dropping `scale_recovery.py`
Dropping the scale recovery module fundamentally alters the physical validity of the pipeline. MediaPipe's `pose_world_landmarks` returns coordinates in a synthetic metric space where the origin (0, 0, 0) is placed at the hip midpoint [2]. 

While these values are proportional, they are **scale-ambiguous**. MediaPipe assumes a standard human height. If you run a video of a child and a video of a tall adult through this current pipeline, the `CalculateBoneLength` class will return nearly identical bone lengths for both subjects.

**Improvement:** To extract *professional-grade* biomechanical data, you must reintroduce a scaling mechanism. This requires either:
1. Placing a calibration object of known length in the video frame.
2. Providing a `reference_height_m` configuration variable to mathematically scale the synthetic 3D vectors back to reality:

`V_real = V_synthetic * (Height_actual / Height_synthetic)`