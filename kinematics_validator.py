import numpy as np
import pandas as pd
from constants import BONES, ANATOMICAL_LIMITS, RIGID_CHAINS


class KinematicValidator:
    def __init__(self, df_3D: pd.DataFrame, tolerance: float = 0.15):
        self.df = df_3D.copy()
        self.tolerance = tolerance

    def validate_and_correct_bones(self) -> pd.DataFrame:
        for _, (joint1, joint2) in BONES.items():
            #lengths of raw data along axes
            dx = self.df[f'{joint1}.x'] - self.df[f'{joint2}.x']
            dy = self.df[f'{joint1}.y'] - self.df[f'{joint2}.y']
            dz = self.df[f'{joint1}.z'] - self.df[f'{joint2}.z']

            #lengths of raw data (Actual lengths in 3d)
            lengths = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            true_val = np.median(lengths)

            #tolerance values
            lBound = true_val * (1.0 - self.tolerance)
            uBound = true_val * (1.0 + self.tolerance)

            #finding the lengths to be corrected
            invalid_lengths = (lengths < lBound) | (lengths > uBound)

            #setting invalid lengths to np.nan
            self.df.loc[invalid_lengths, [f'{joint2}.x', f'{joint2}.y', f'{joint2}.z']] = np.nan

        #extracting columns names
        cols = [col for col in self.df.columns if 'visibility' not in col]

        #interpolating the np.nan values
        #the interpolated data will always be in between the tolerance values
        self.df[cols] = self.df[cols].interpolate(method='linear', limit_direction='both').ffill().bfill()

        #going through bones that we have to correct
        #not going through torso and neck because they are not rigid in nature
        #they flex and bend too 
        for joint1, joint2 in RIGID_CHAINS:
            dx = self.df[f'{joint2}.x'] - self.df[f'{joint1}.x']
            dy = self.df[f'{joint2}.y'] - self.df[f'{joint1}.y']
            dz = self.df[f'{joint2}.z'] - self.df[f'{joint1}.z']

            lengths = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            #this is done to prevent division by zero error
            lengths = np.where(lengths == 0, 1e-6, lengths)

            true_val = lengths.median()

            #unit vectors for direction in 3d
            unitX = dx / lengths
            unitY = dy / lengths
            unitZ = dz / lengths

            #setting the second joint keeping the first one fixed along the direction
            self.df[f'{joint2}.x'] = self.df[f'{joint1}.x'] + (unitX * true_val)
            self.df[f'{joint2}.y'] = self.df[f'{joint1}.y'] + (unitY * true_val)
            self.df[f'{joint2}.z'] = self.df[f'{joint1}.z'] + (unitZ * true_val)

        return self.df
    
    def clamp_angles(self, angles_df: pd.DataFrame) -> pd.DataFrame:
        angles = angles_df.copy()

        for joint_name, (min_angle, max_angle) in ANATOMICAL_LIMITS.items():
            if joint_name in angles.columns:
                angles[joint_name] = angles[joint_name].clip(lower=min_angle, upper=max_angle)

        return angles