#  Copyright (c) 2019. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.
import numpy as np


class body_model():
    def __init__(self, image_id: str, keypoints):
        self.image_id = image_id
        self.center_mass = self.set_center_mass(keypoints)
        self.keypoints = self.set_relative_keypoints(keypoints)

    @staticmethod
    def set_center_mass(keypoints):
        return (keypoints[11, 0]+keypoints[12, 0])/2, (keypoints[11, 0]+keypoints[12, 0])/2

    def set_relative_keypoints(self, keypoints):

        keypoints[:, 0] = keypoints[:, 0]-self.center_mass[0]
        keypoints[:, 1] = keypoints[:, 1]-self.center_mass[1]

        xmax = np.max((keypoints[:, 0]))
        xmin = np.min((keypoints[:, 0]))
        ymax = np.max((keypoints[:, 1]))
        ymin = np.min((keypoints[:, 1]))
        xscale = xmax-xmin
        yscale = ymax-ymin

        keypoints[:, 0] = keypoints[:, 0] * xscale
        keypoints[:, 1] = keypoints[:, 1] * yscale
        return keypoints


if __name__ == '__main__':


    key = [[21,22,1], [13,12,1], [13,12,1], [13,12,1], [13,12,1],[13,12,1],[13,12,1],[13,12,1],[13,12,1], [13,12,1],[13,12,1],[13,12,1],[13,12,1],[13,12,1],[13,12,1],[13,12,1],[13,12,1],]
    key = np.vstack(key)
    patata = body_model("patata", key)

    print()