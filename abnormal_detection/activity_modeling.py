#  Copyright (c) 2019. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.
import numpy as np
import os
import json


class BodyModel:

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
        xscale = abs(xmax-xmin)
        yscale = abs(ymax-ymin)

        keypoints[:, 0] = keypoints[:, 0] / float(xscale)
        keypoints[:, 1] = keypoints[:, 1] / float(yscale)
        return keypoints


def read_body(in_path):
    body_list = []
    for filename in os.listdir(in_path):
        json_path = os.path.join(in_path, filename)
        with open(json_path) as json_file:
            data = json.load(json_file)
            for anotation in data:
                body_list = BodyModel(anotation['image_id'], anotation['keypoints'])
    return body_list


if __name__ == '__main__':

    read_body()