from scipy.signal import savgol_filter
import pandas as pd

def smooth_keypoints(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df_copy = df.copy()
    
    coord_cols = [col for col in df_copy.columns if any(ext in col for ext in ['.x', '.y', '.z'])]
    
    n_frames = len(df_copy)
    window = 9
    if n_frames < window:
        window = n_frames if n_frames % 2 != 0 else n_frames - 1
        
    if window <= 2:
        return df_copy

    df_copy[coord_cols] = savgol_filter(
        df_copy[coord_cols], 
        window_length=window, 
        polyorder=2, 
        axis=0, 
        mode='interp'
    )
     
    return df_copy