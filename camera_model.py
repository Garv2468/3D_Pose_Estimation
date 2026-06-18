import pandas as pd
import numpy as np

class CameraProjector:
    def __init__(self, width: int, height: int, focal_length_factor: float = 1.0):
        self.width = width
        self.height = height
        self.focal_length = width * focal_length_factor

    def project_and_anchor(self, df_3d: pd.DataFrame, df_2d: pd.DataFrame):
        df = pd.DataFrame()

        def is_valid(joint_name):
            vis_col = f"{joint_name}.visibility"
            if vis_col in df_2d.columns:
                return df_2d[vis_col].fillna(0) > 0.50
            return np.abs(df_2d[f"{joint_name}.x"]) > 0.01

        hips_valid = is_valid("LEFT_HIP") & is_valid("RIGHT_HIP")
        shoulders_valid = is_valid("LEFT_SHOULDER") & is_valid("RIGHT_SHOULDER")

        hip_anchor_x = (df_2d["LEFT_HIP.x"] + df_2d["RIGHT_HIP.x"]) / 2.0
        hip_anchor_y = (df_2d["LEFT_HIP.y"] + df_2d["RIGHT_HIP.y"]) / 2.0
        
        shoulder_anchor_x = (df_2d["LEFT_SHOULDER.x"] + df_2d["RIGHT_SHOULDER.x"]) / 2.0
        shoulder_anchor_y = (df_2d["LEFT_SHOULDER.y"] + df_2d["RIGHT_SHOULDER.y"]) / 2.0
        
        nose_anchor_x = df_2d["NOSE.x"]
        nose_anchor_y = df_2d["NOSE.y"]

        anchor_x = np.where(hips_valid, hip_anchor_x, 
                            np.where(shoulders_valid, shoulder_anchor_x, nose_anchor_x))
        anchor_y = np.where(hips_valid, hip_anchor_y, 
                            np.where(shoulders_valid, shoulder_anchor_y, nose_anchor_y))

        offset_3d_x = np.where(hips_valid, 0.0, 
                               np.where(shoulders_valid, df_3d["NECK_BASE.x"], df_3d["NOSE.x"]))
        offset_3d_y = np.where(hips_valid, 0.0, 
                               np.where(shoulders_valid, df_3d["NECK_BASE.y"], df_3d["NOSE.y"]))

        dx_2d = df_2d["RIGHT_SHOULDER.x"] - df_2d["LEFT_SHOULDER.x"]
        dy_2d = df_2d["RIGHT_SHOULDER.y"] - df_2d["LEFT_SHOULDER.y"]
        shoulder_width_2d = np.sqrt(dx_2d**2 + dy_2d**2)

        dx_3d = df_3d["RIGHT_SHOULDER.x"] - df_3d["LEFT_SHOULDER.x"]
        dy_3d = df_3d["RIGHT_SHOULDER.y"] - df_3d["LEFT_SHOULDER.y"]
        dz_3d = df_3d["RIGHT_SHOULDER.z"] - df_3d["LEFT_SHOULDER.z"]
        shoulder_width_3d = np.sqrt(dx_3d**2 + dy_3d**2 + dz_3d**2)

        face_torso_2d = np.sqrt((df_2d["NOSE.x"] - shoulder_anchor_x)**2 + (df_2d["NOSE.y"] - shoulder_anchor_y)**2)
        face_torso_3d = np.sqrt((df_3d["NOSE.x"] - df_3d["NECK_BASE.x"])**2 + (df_3d["NOSE.y"] - df_3d["NECK_BASE.y"])**2 + (df_3d["NOSE.z"] - df_3d["NECK_BASE.z"])**2)

        ref_2d = np.maximum(shoulder_width_2d, face_torso_2d)
        ref_3d = np.maximum(shoulder_width_3d, face_torso_3d)

        ref_2d = np.where(ref_2d < 0.05, 0.05, ref_2d)

        dynamic_depth = ref_3d / ref_2d

        points_to_project = list(set([col.split('.')[0] for col in df_3d.columns if 'visibility' not in col]))

        for point in points_to_project:
            X = df_3d[f'{point}.x'].values - offset_3d_x
            Y = df_3d[f'{point}.y'].values - offset_3d_y
            
            Z = df_3d[f'{point}.z'].values + dynamic_depth

            proj_x = (self.focal_length * (X / Z)) / self.width
            proj_y = (self.focal_length * (Y / Z)) / self.height

            df[f'{point}.x'] = proj_x + anchor_x
            df[f'{point}.y'] = proj_y + anchor_y

        return df