import cv2
import numpy as np
from constants import BONES

def draw_skeleton(frame: np.ndarray, np_coords: dict, frame_num: int, width: int, height: int):
    img = frame
    for bone_name, (joint1, joint2) in BONES.items():
        x1, y1 = np_coords[f"{joint1}.x"][frame_num], np_coords[f"{joint1}.y"][frame_num]
        x2, y2 = np_coords[f"{joint2}.x"][frame_num], np_coords[f"{joint2}.y"][frame_num]
        
        if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
            continue
            
        startX, startY = int(x1 * width), int(y1 * height)
        endX, endY = int(x2 * width), int(y2 * height)
        
        stp = (startX, startY)
        etp = (endX, endY)

        cv2.line(img, stp, etp, (0, 255, 0), 10)
            
    return img

def save_annoted_video(cap, metadata, df, output_path="output.mp4"):
    frame_num = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    width = metadata["width"]
    height = metadata["height"]
    fps = metadata['fps']
    
    output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    np_coords = {col: df[col].to_numpy() for col in df.columns}

    try:
        while True:
            success, frame = cap.read_frame()
            if not success:
                break
            
            img = draw_skeleton(frame, np_coords, frame_num, width, height)
            
            output.write(img)
            frame_num += 1
            
    finally:
        output.release()
        cap.release()