import pandas as pd
import cv2
import numpy as np
from constants import BONES

def draw_skeleton(frame: np.ndarray, df : pd.DataFrame, frame_num : int, width, height):
    img = frame
    for i in BONES:
        if i != "SPINE" :
            stp = (int(df.iloc[frame_num][f"{BONES[i][0]}.x"]*width), int(df.iloc[frame_num][f"{BONES[i][0]}.y"]*height))
            etp = (int(df.iloc[frame_num][f"{BONES[i][1]}.x"]*width), int(df.iloc[frame_num][f"{BONES[i][1]}.y"]*height))
            cv2.line(img, stp, etp, (0, 255, 0), 2)
    return img

def save_annoted_video(cap, metadata, df):
    frame_num = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = metadata["width"]
    height = metadata["height"]
    fps = metadata['fps']
    output = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

    while True :
        success, frame = cap.read_frame()
        if not success:
            break
        img = draw_skeleton(frame, df, frame_num, width, height)
        output.write(img)
        frame_num += 1
    output.release()

