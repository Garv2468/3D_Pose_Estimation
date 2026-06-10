import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import video_loader as vl
import time

l = []
modelPath = r"pose_landmarker_heavy.task"

baseOptions = python.BaseOptions(model_asset_path=modelPath)
options = vision.PoseLandmarkerOptions(base_options=baseOptions
                                       , output_segmentation_masks=False
                                       , running_mode=vision.RunningMode.IMAGE
                                       , num_poses=1
                                       , min_pose_detection_confidence=0.5
                                       , min_tracking_confidence=0.5)
poseEstimator = vision.PoseLandmarker.create_from_options(options)


cap, video_data = vl.load_video("test2.mp4")
print(f"Processing video at {video_data['fps']} FPS.")


for frame in vl.frame_generator(cap):


    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mpImg = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbFrame)

    result = poseEstimator.detect(mpImg)

    if result.pose_world_landmarks:
        world_landmarks = result.pose_world_landmarks[0]
        rightWrist = world_landmarks[16]
        l.append((rightWrist.x, rightWrist.y, rightWrist.z))




    cv2.imshow("Video Playback", frame)







    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vl.release_video(cap)
cv2.destroyAllWindows()
print(l)
