import numpy as np
import pandas as pd
from constants import DOF1_angles, DOF3_angles, BONES

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

def interpolate_keypoints(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Provided DataFrame is empty.")
    
    df_clean = df.copy()
    coord_cols = [col for col in df_clean.columns if 'visibility' not in col]

    df_clean[coord_cols] = df_clean[coord_cols].interpolate(method='linear', limit_direction='both').ffill().bfill()
    
    df_clean[coord_cols] = df_clean[coord_cols].fillna(0)
    
    return df_clean



class Calculations:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
        #joints required for torso and head movements (not tracked by mediapipe)
        self._make_virtual_joints()

    def _make_virtual_joints(self) -> None:
        self.df['PELVIS.x'] = (self.df['LEFT_HIP.x'] + self.df['RIGHT_HIP.x']) / 2.0
        self.df['PELVIS.y'] = (self.df['LEFT_HIP.y'] + self.df['RIGHT_HIP.y']) / 2.0
        self.df['PELVIS.z'] = (self.df['LEFT_HIP.z'] + self.df['RIGHT_HIP.z']) / 2.0

        self.df['NECK_BASE.x'] = (self.df['LEFT_SHOULDER.x'] + self.df['RIGHT_SHOULDER.x']) / 2.0
        self.df['NECK_BASE.y'] = (self.df['LEFT_SHOULDER.y'] + self.df['RIGHT_SHOULDER.y']) / 2.0
        self.df['NECK_BASE.z'] = (self.df['LEFT_SHOULDER.z'] + self.df['RIGHT_SHOULDER.z']) / 2.0

class CalculateJointAngles(Calculations):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

        self.angles = pd.DataFrame()

        #DOF is degree of freedom
        #these are the global angles (wrt the 3d plane)
        self.calculate_DOF1_angles()
        self.calculate_DOF3_angles()

        #local angles means with respect to other angles
        self.calculate_local_angles() 

        #maybe required later
        self.angles['PELVIS_ROOT_X'] = self.df['PELVIS.x']
        self.angles['PELVIS_ROOT_Y'] = self.df['PELVIS.y']
        self.angles['PELVIS_ROOT_Z'] = self.df['PELVIS.z']

    def _DOF1_angle(self, df: pd.DataFrame, a: str, b: str, c: str) -> np.ndarray:
        u_x = df[f'{a}.x'].values - df[f'{b}.x'].values
        u_y = df[f'{a}.y'].values - df[f'{b}.y'].values
        u_z = df[f'{a}.z'].values - df[f'{b}.z'].values
        
        v_x = df[f'{c}.x'].values - df[f'{b}.x'].values
        v_y = df[f'{c}.y'].values - df[f'{b}.y'].values
        v_z = df[f'{c}.z'].values - df[f'{b}.z'].values

        dot = (u_x * v_x) + (u_y * v_y) + (u_z * v_z)
        u_mod = np.sqrt(u_x**2 + u_y**2 + u_z**2)
        v_mod = np.sqrt(v_x**2 + v_y**2 + v_z**2)

        angle = np.arccos(np.clip(dot / (u_mod * v_mod + 1e-6), -1.0, 1.0))
        return np.degrees(angle)
    
    def _DOF3_angle(self, df: pd.DataFrame, a: str, b: str, plane: str = "sagittal") -> np.ndarray:
        dx = df[f'{b}.x'].values - df[f'{a}.x'].values
        dy = df[f'{b}.y'].values - df[f'{a}.y'].values
        dz = df[f'{b}.z'].values - df[f'{a}.z'].values

        if plane == 'sagittal':
            angles = np.arctan2(dy, dz)
        elif plane == 'coronal':
            angles = np.arctan2(dy, dx)
        elif plane == 'transverse':
            angles = np.arctan2(dz, dx)
        else:
            raise ValueError("Plane must be 'sagittal', 'coronal', or 'transverse'")
            
        return np.degrees(angles)
    
    def _local_angle(self, global_child: pd.Series, global_parent: pd.Series) -> pd.Series:
        raw_local = global_child - global_parent
        
        normalized_local = (raw_local + 180) % 360 - 180
        
        return normalized_local
    
    def calculate_DOF1_angles(self):
        for joint_name, (pt_a, pt_b, pt_c) in DOF1_angles.items():
            self.angles[f'{joint_name}_ANGLE'] = self._DOF1_angle(self.df, pt_a, pt_b, pt_c)

    def calculate_DOF3_angles(self) -> None:
        for joint_name, (pt_a, pt_b, plane) in DOF3_angles.items():
            self.angles[f'{joint_name}_GLOBAL'] = self._DOF3_angle(self.df, pt_a, pt_b, plane=plane)

    def calculate_local_angles(self) -> None:
        self.angles['LEFT_SHOULDER_LOCAL_PITCH'] = self._local_angle(self.angles['LEFT_SHOULDER_PITCH_GLOBAL'], self.angles['SPINE_PITCH_GLOBAL'])
        self.angles['LEFT_SHOULDER_LOCAL_ROLL']  = self._local_angle(self.angles['LEFT_SHOULDER_ROLL_GLOBAL'], self.angles['SPINE_ROLL_GLOBAL'])
        self.angles['RIGHT_SHOULDER_LOCAL_PITCH'] = self._local_angle(self.angles['RIGHT_SHOULDER_PITCH_GLOBAL'], self.angles['SPINE_PITCH_GLOBAL'])
        self.angles['RIGHT_SHOULDER_LOCAL_ROLL']  = self._local_angle(self.angles['RIGHT_SHOULDER_ROLL_GLOBAL'], self.angles['SPINE_ROLL_GLOBAL'])

        self.angles['NECK_LOCAL_PITCH'] = self._local_angle(self.angles['NECK_PITCH_GLOBAL'], self.angles['SPINE_PITCH_GLOBAL'])
        self.angles['NECK_LOCAL_ROLL']  = self._local_angle(self.angles['NECK_ROLL_GLOBAL'], self.angles['SPINE_ROLL_GLOBAL'])
        self.angles['NECK_LOCAL_YAW']   = self._local_angle(self.angles['NECK_YAW_GLOBAL'], self.angles['SPINE_YAW_GLOBAL'])

        self.angles['LEFT_HIP_LOCAL_PITCH']  = self._local_angle(self.angles['LEFT_HIP_PITCH_GLOBAL'], self.angles['SPINE_PITCH_GLOBAL'])
        self.angles['LEFT_HIP_LOCAL_ROLL']   = self._local_angle(self.angles['LEFT_HIP_ROLL_GLOBAL'], self.angles['SPINE_ROLL_GLOBAL'])
        self.angles['RIGHT_HIP_LOCAL_PITCH'] = self._local_angle(self.angles['RIGHT_HIP_PITCH_GLOBAL'], self.angles['SPINE_PITCH_GLOBAL'])
        self.angles['RIGHT_HIP_LOCAL_ROLL']  = self._local_angle(self.angles['RIGHT_HIP_ROLL_GLOBAL'], self.angles['SPINE_ROLL_GLOBAL'])

    def get_angles(self) -> pd.DataFrame:
        if (self.angles.empty):
            raise ValueError("Angles not found. Calculate the angles first.")
        return self.angles
    
class CalculateBoneLength(Calculations):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.bonelength = pd.DataFrame()
        self.compute_bone_length(self.df)

    def _bone_length(self, df: pd.DataFrame, a: str, b: str):
        dx = df[f'{a}.x'].values - df[f'{b}.x'].values
        dy = df[f'{a}.y'].values - df[f'{b}.y'].values
        dz = df[f'{a}.z'].values - df[f'{b}.z'].values

        return np.sqrt(dx**2 + dy**2 + dz**2)

    def compute_bone_length(self, df: pd.DataFrame):
        for bone_name, (joint1, joint2) in BONES.items():
            self.bonelength[f'{bone_name}'] = self._bone_length(df, joint1, joint2)

    def get_bone_lengths(self):
        if (self.bonelength.empty):
            raise ValueError("Lengths not found. Calculate the lengths first.")
        return self.bonelength