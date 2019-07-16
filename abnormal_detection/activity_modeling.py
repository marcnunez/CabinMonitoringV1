#  Copyright (c) 2019. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.
import numpy as np
import os
import json

from sklearn import mixture
from sklearn.decomposition import PCA

from utils.eval import parse_keypoints_to_array


class BodyModel:

    def __init__(self, image_id: str, keypoints):
        self.image_id = image_id
        keypoints = parse_keypoints_to_array(keypoints)
        self.center_mass = self.set_center_mass(keypoints)
        self.keypoints = self.set_relative_keypoints(keypoints)

    @staticmethod
    def set_center_mass(keypoints):
        return (keypoints[11, 0]+keypoints[12, 0])/2, (keypoints[11, 1]+keypoints[12, 1])/2

    @staticmethod
    def set_orientation(keypoints):
        pass

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
        return keypoints[:, 0:2]


def fit_model(bodys):
    sklearn_pca = PCA(n_components=2)
    np.random.seed(1)
    g = mixture.GaussianMixture(n_components=34)
    print(np.round(g.weights_, 34))
    return g.fit(bodys)


def predict_model()


def read_body_directory(in_path):
    body_list = []
    for filename in os.listdir(in_path):
        json_path = os.path.join(in_path, filename)
        body_list.append(read_body_json(json_path))
    return body_list


def read_body_json(json_path):
    body_list = []
    with open(json_path) as json_file:
        data = json.load(json_file)
        for anotation in data:
            body_list.append(BodyModel(anotation['image_id'], anotation['keypoints']))
    return body_list


if __name__ == '__main__':
    list_bodies = []
    bodys = read_body_json('../examples/data/activity_modeling/c3-result.json')
    for body in bodys:
        list_bodies.append(body.keypoints.flatten())
    list_bodies = np.vstack(list_bodies)
    fit_model(list_bodies)