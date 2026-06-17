from scipy.signal import savgol_filter
import pandas as pd

def smooth_keypoints(df: pd.DataFrame):
    for i in df:
        if ".x" in i or ".y" in i or ".z" in i :
            df[i] = savgol_filter(df[i], window_length=9, polyorder=2)
     
    return df