import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from video_loader import VideoLoader
from constants import TARGET_POINTS_3D

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

        self.df_2d = pd.DataFrame()
        self.df_3d = pd.DataFrame()

        self.target_points_2d = {
            "NOSE": 0, 
            "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
            "LEFT_HIP": 23, "RIGHT_HIP": 24
        }

    def generate_dataframe(self, loader: VideoLoader):
        frames_3d = []
        frames_2d = []
        
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
            frame_data_3d = {}
            frame_data_2d = {}

            if not result.pose_world_landmarks:
                for point_name in TARGET_POINTS_3D.values():
                    frame_data_3d.update({f'{point_name}.x': np.nan, f'{point_name}.y': np.nan, f'{point_name}.z': np.nan, f'{point_name}.visibility': np.nan})
            else:
                for idx, point_name in TARGET_POINTS_3D.items():
                    landmark = result.pose_world_landmarks[0][idx]
                    
                    if landmark.visibility >= self.threshold_visibility:
                        frame_data_3d.update({f'{point_name}.x': landmark.x, f'{point_name}.y': landmark.y, f'{point_name}.z': landmark.z, f'{point_name}.visibility': landmark.visibility})
                    else:
                        frame_data_3d.update({f'{point_name}.x': np.nan, f'{point_name}.y': np.nan, f'{point_name}.z': np.nan, f'{point_name}.visibility': np.nan})

            if not result.pose_landmarks:
                for point in self.target_points_2d.keys():
                    frame_data_2d.update({f'{point}.x': np.nan, f'{point}.y': np.nan, f'{point}.visibility': np.nan})
            else:
                for point_name, i in self.target_points_2d.items():
                    landmark = result.pose_landmarks[0][i]
                    if landmark.visibility >= self.threshold_visibility:
                        frame_data_2d.update({f'{point_name}.x': landmark.x, f'{point_name}.y': landmark.y, f'{point_name}.visibility': landmark.visibility})
                    else:
                        frame_data_2d.update({f'{point_name}.x': np.nan, f'{point_name}.y': np.nan, f'{point_name}.visibility': np.nan})
            
            frames_2d.append(frame_data_2d)
            frames_3d.append(frame_data_3d)
            frameNumber += 1
            
        loader.release()
        self.df_2d = pd.DataFrame(frames_2d)
        self.df_3d = pd.DataFrame(frames_3d)

    def get_keypoints_3d(self) -> pd.DataFrame:
        if self.df_3d.empty:
            raise ValueError("Keypoints not found. Run generate_dataframe() first.")
        return self.df_3d 
    
    def get_keypoints_2d(self) -> pd.DataFrame:
        if self.df_2d.empty:
            raise ValueError("Keypoints not found. Run generate_dataframe() first.")
        return self.df_2d