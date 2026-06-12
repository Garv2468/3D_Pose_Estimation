import cv2
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
        self.key_points = []

    def generate_keypoints(self, loader: VideoLoader):
        self.key_points.clear()
        
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

            if not result.pose_world_landmarks:
                self.key_points.append(None)
            else:
                person = {}

                for i, landmark in enumerate(result.pose_world_landmarks[0]):
                    if landmark.visibility >= self.threshold_visibility:
                        person[POINTS[i]] = {
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z,
                            'visibility': landmark.visibility
                        }
                    else:
                        person[POINTS[i]] = {
                            'x': None,
                            'y': None,
                            'z': None,
                            'visibility': None
                        }
                    
                self.key_points.append(person)

            frameNumber += 1

    def get_keypoints(self) -> list:
        if (len(self.key_points) == 0):
            raise ValueError("KeyPoints not found. Run generate_keypoints() first.")
        return self.key_points