import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from video_loader import VideoLoader

POINTS = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", 
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER", 
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", 
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", 
    "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", 
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", 
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]

class PoseDetector:
    def __init__(self, path: str, detection_confidence: float = 0.75, tracking_confidence: float = 0.75, threshold_visibility: float = 0.75):
        base_options = python.BaseOptions(model_asset_path=path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.threshold_visibility = threshold_visibility
        self.detector = vision.PoseLandmarker.create_from_options(options)

        self.df = pd.DataFrame()

    def generate_dataframe(self, loader: VideoLoader):
        frames = []
        
        frameNumber = 0
        fps = loader.get_metadata()['fps']

        while True:
            success, frame = loader.read_frame()
            if not success:
                break

            timestampMS = int(frameNumber * 1000 / fps)
            RGBFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mpImg = mp.Image(image_format=mp.ImageFormat.SRGB, data=RGBFrame)

            result = self.detector.detect_for_video(mpImg, timestampMS)
            frame_data = {}

            if not result.pose_world_landmarks:
                for point in POINTS:
                    frame_data[f'{point}.x'] = np.nan
                    frame_data[f'{point}.y'] = np.nan
                    frame_data[f'{point}.z'] = np.nan
                    frame_data[f'{point}.visibility'] = np.nan
            else:
                for i, landmark in enumerate(result.pose_world_landmarks[0]):
                    point_name = POINTS[i]
                    if landmark.visibility >= self.threshold_visibility:
                        frame_data[f'{point_name}.x'] = landmark.x
                        frame_data[f'{point_name}.y'] = landmark.y
                        frame_data[f'{point_name}.z'] = landmark.z
                        frame_data[f'{point_name}.visibility'] = landmark.visibility
                    else:
                        # Use np.nan instead of None for seamless Pandas interpolation
                        frame_data[f'{point_name}.x'] = np.nan
                        frame_data[f'{point_name}.y'] = np.nan
                        frame_data[f'{point_name}.z'] = np.nan
                        frame_data[f'{point_name}.visibility'] = np.nan
                    
            frames.append(frame_data)
            frameNumber += 1
        
        loader.release()
        self.df = pd.DataFrame(frames)

    def get_keypoints(self) -> pd.DataFrame:
        if self.df.empty:
            raise ValueError("KeyPoints not found. Run generate_dataframe() first.")
        return self.df