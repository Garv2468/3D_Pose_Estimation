import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import video_loader as vl

MODEL_PATH = "/home/garv/Python/Open_cv/pose_landmarker_heavy.task"

def init_detector():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return vision.PoseLandmarker.create_from_options(options)

def detect_video(cap, detector, fps):
    keypoints_sequence = []
    frame_number = 0
    d = {}
    for frame in vl.frame_generator(cap):
        timestamp_ms = int(frame_number * 1000 / fps)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_world_landmarks:
            keypoints_sequence.append(None)
        else:
            frame_landmark = []
            for i in range(33):
                d = {
                'x' : result.pose_world_landmarks[0][i].x,
                'y' : result.pose_world_landmarks[0][i].y,
                'z' : result.pose_world_landmarks[0][i].z,
                'visibility' : result.pose_world_landmarks[0][i].visibility
                }
                frame_landmark.append(d)
            keypoints_sequence.append(frame_landmark)

        frame_number += 1

    return keypoints_sequence # [[{}, {}, {}, ...], [{}, {}, {}, ...]] frame -> {x, y, z, visibility}

