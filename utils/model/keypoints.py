import numpy as np
from keypoint import Keypoint


class Keypoints:

    def __init_(self, nose: Keypoint, l_eye: Keypoint, r_eye: Keypoint, l_ear: Keypoint, r_ear: Keypoint, l_shoulder: Keypoint,
                r_shoulder: Keypoint, l_elbow: Keypoint, r_elbow: Keypoint, l_wrist: Keypoint, r_wrist: Keypoint, l_hip: Keypoint,
                r_hip: Keypoint, l_knee: Keypoint, r_knee: Keypoint, l_ankle: Keypoint, r_ankle: Keypoint):
        self.nose = nose
        self.l_eye = l_eye
        self.r_eye = r_eye
        self.l_ear = l_ear
        self.r_ear = r_ear
        self.l_shoulder = l_shoulder
        self.r_shoulder = r_shoulder
        self.l_elbow = l_elbow
        self.r_elbow = r_elbow
        self.l_wrist = l_wrist
        self.r_wrist = r_wrist
        self.l_hip = l_hip
        self.r_hip = r_hip
        self.l_knee = l_knee
        self.r_knee = r_knee
        self.l_ankle = l_ankle
        self.r_ankle = r_ankle

    def get_center(self) -> np.array:
        x_center = np.mean(self.l_shoulder.x, self.r_shoulder.x, self.l_wrist.x, self.r_wrist.x)
        y_center = np.mean(self.l_shoulder.y, self.r_shoulder.y, self.l_wrist.y, self.r_wrist.y)
        return np.array(x_center, y_center)

    def get_euclidean_distance(self, body_part: Keypoint) -> float:
        return np.linalg.norm(self.get_center(), body_part.get_coordinates())

