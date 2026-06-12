import pandas as pd

def reconstruct_keypoints(interpolated_df: pd.DataFrame) -> list:
    flat_records = interpolated_df.to_dict(orient='records')
    sq = []
    
    for frame in flat_records:
        nested_frame = {}
        for key, val in frame.items():
            joint, parameter = str(key).split('.')
            if joint not in nested_frame:
                nested_frame[joint] = {}
            nested_frame[joint][parameter] = val
            
        sq.append(nested_frame)
        
    return sq

def interpolate_keypoints(raw_sequence: list) -> list:
    if not raw_sequence:
        return []
        
    df = pd.json_normalize(raw_sequence)
    
    df = df.interpolate(method='linear', limit_direction='both')
    
    return reconstruct_keypoints(df)