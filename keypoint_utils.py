import numpy as np
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

def interpolate_keypoints(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Provided DataFrame is empty.")
    
    return df.interpolate(method='linear', limit_direction='both')

class CalculateJointAngles:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.angles = pd.DataFrame()

        #joints required for torso and head movements (not tracked by mediapipe)
        self._make_virtual_joints()

        #DOF is degree of freedom
        self.DOF1_angles = {
            "LEFT_ELBOW": ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
            "RIGHT_ELBOW": ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
            "LEFT_KNEE": ('LEFT_HIP', 'LEFT_KNEE', 'LEFT_ANKLE'),
            "RIGHT_KNEE": ('RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'),
            }
        self.DOF3_angles = {
            "LEFT_SHOULDER_PITCH": ('LEFT_SHOULDER', 'LEFT_ELBOW', 'sagittal'),
            "LEFT_SHOULDER_ROLL": ('LEFT_SHOULDER', 'LEFT_ELBOW', 'coronal'),
            "RIGHT_SHOULDER_PITCH": ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'sagittal'),
            "RIGHT_SHOULDER_ROLL": ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'coronal'),

            "LEFT_HIP_PITCH": ('LEFT_HIP', 'LEFT_KNEE', 'sagittal'),
            "LEFT_HIP_ROLL": ('LEFT_HIP', 'LEFT_KNEE', 'coronal'),
            "RIGHT_HIP_PITCH": ('RIGHT_HIP', 'RIGHT_KNEE', 'sagittal'),
            "RIGHT_HIP_ROLL": ('RIGHT_HIP', 'RIGHT_KNEE', 'coronal'),

            "LEFT_FOOT_PITCH": ('LEFT_HEEL', 'LEFT_FOOT_INDEX', 'sagittal'),
            "RIGHT_FOOT_PITCH": ('RIGHT_HEEL', 'RIGHT_FOOT_INDEX', 'sagittal'),

            "SPINE_PITCH": ('PELVIS', 'NECK_BASE', 'sagittal'),
            "SPINE_ROLL": ('PELVIS', 'NECK_BASE', 'coronal'),
            "SPINE_YAW": ('PELVIS', 'NECK_BASE', 'transverse'),

            "NECK_PITCH": ('NECK_BASE', 'NOSE', 'sagittal'),
            "NECK_ROLL": ('NECK_BASE', 'NOSE', 'coronal'),
            "NECK_YAW": ('NECK_BASE', 'NOSE', 'transverse')
        }

        #these are the global angles (wrt the 3d plane)
        self.calculate_DOF1_angles()
        self.calculate_DOF3_angles()

        #local angles means with respect to other angles
        self.calculate_local_angles() 

    def _make_virtual_joints(self):
        self.df['PELVIS.x'] = (self.df['LEFT_HIP.x'] + self.df['RIGHT_HIP.x']) / 2.0
        self.df['PELVIS.y'] = (self.df['LEFT_HIP.y'] + self.df['RIGHT_HIP.y']) / 2.0
        self.df['PELVIS.z'] = (self.df['LEFT_HIP.z'] + self.df['RIGHT_HIP.z']) / 2.0

        self.df['NECK_BASE.x'] = (self.df['LEFT_SHOULDER.x'] + self.df['RIGHT_SHOULDER.x']) / 2.0
        self.df['NECK_BASE.y'] = (self.df['LEFT_SHOULDER.y'] + self.df['RIGHT_SHOULDER.y']) / 2.0
        self.df['NECK_BASE.z'] = (self.df['LEFT_SHOULDER.z'] + self.df['RIGHT_SHOULDER.z']) / 2.0

    def _DOF1_angle(self, df: pd.DataFrame, a: str, b: str, c: str) -> np.ndarray:
        A = df[[f'{a}.x', f'{a}.y', f'{a}.z']].values
        B = df[[f'{b}.x', f'{b}.y', f'{b}.z']].values
        C = df[[f'{c}.x', f'{c}.y', f'{c}.z']].values

        u, v = A - B, C - B

        dot = np.sum(u * v, axis=1)
        u_mod = np.linalg.norm(u, axis=1)
        v_mod = np.linalg.norm(v, axis=1)

        angle = np.arccos(np.clip(dot / (u_mod * v_mod), -1.0, 1.0))

        return np.degrees(angle)
    
    def _DOF3_angle(self, df: pd.DataFrame, a: str, b: str, plane: str = "sagittal") -> np.ndarray:
        ax = df[f'{a}.x'].to_numpy(dtype=np.float64)
        ay = df[f'{a}.y'].to_numpy(dtype=np.float64)
        az = df[f'{a}.z'].to_numpy(dtype=np.float64)
        
        bx = df[f'{b}.x'].to_numpy(dtype=np.float64)
        by = df[f'{b}.y'].to_numpy(dtype=np.float64)
        bz = df[f'{b}.z'].to_numpy(dtype=np.float64)

        dx = bx - ax
        dy = by - ay
        dz = bz - az

        #three cross sections of a body
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
        for joint_name, (pt_a, pt_b, pt_c) in self.DOF1_angles.items():
            self.angles[f'{joint_name}_ANGLE'] = self._DOF1_angle(self.df, pt_a, pt_b, pt_c)

    def calculate_DOF3_angles(self) -> None:
        for joint_name, (pt_a, pt_b, plane) in self.DOF3_angles.items():
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
        
        #maybe required later
        self.angles['PELVIS_ROOT_X'] = self.df['PELVIS.x']
        self.angles['PELVIS_ROOT_Y'] = self.df['PELVIS.y']
        self.angles['PELVIS_ROOT_Z'] = self.df['PELVIS.z']

        return self.angles