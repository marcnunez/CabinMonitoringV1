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
from utils.memory import memory
from opt import opt


class BodyModel:

    def __init__(self, image_id: str, keypoints):
        self.image_id = image_id
        keypoints = parse_keypoints_to_array(keypoints)
        self.center_mass = self.set_center_mass(keypoints)
        self.orientation = self.set_orientation(keypoints)
        self.keypoints = self.set_relative_keypoints(keypoints)
        self.original_keypoints = keypoints[:, 0:2]

    @staticmethod
    def set_center_mass(keypoints):
        return (keypoints[11, 0]+keypoints[12, 0])/2, (keypoints[11, 1]+keypoints[12, 1])/2

    @staticmethod
    def set_orientation(keypoints):
        return np.arctan2(keypoints[11, 0]-keypoints[12, 0], keypoints[11, 1]-keypoints[12, 1])

    def set_relative_keypoints(self, keypoints):
        keypoints = keypoints[:, 0:2]

        """
        INVARIANT TO ROTATION
        rotation_matrix = [[np.cos(self.orientation), -np.sin(self.orientation)], [np.sin(self.orientation),
                                                                                   np.cos(self.orientation)]]
        for i in range(0, len(keypoints)):
            keypoints[i, :] = np.dot(rotation_matrix, keypoints[i,:])
        """

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


def demo_webcam_wraper(results):
    body_list = []
    id_img = results['imgname']
    pca_fit, gmm_fit = fit_model()

    for sk in results['result']:
        keypoints = sk['keypoints'].numpy()
        body_list.append(BodyModel(id_img, keypoints))

        pca_bodys = pca_predict(body_list, pca_fit)

        if predict_model(pca_bodys, gmm_fit) !=0:
            print("There is something Wrong, maybe :P")


def predict_model(bodys_to_predict_gmm, model_fitted):
    list_fitted_bodys = model_fitted.predict(bodys_to_predict_gmm)
    count = 0
    for index in list_fitted_bodys:
        if index == 4:
            count +=1
    return count


@memory.cache()
def fit_model():
    list_bodys = read_body_json('../examples/data/activity_modeling/c1-result.json') + \
                 read_body_json('../examples/data/activity_modeling/c6-result.json') + \
                 read_body_json('../examples/data/activity_modeling/c3-result.json') + \
                 read_body_json('../examples/data/activity_modeling/c5-result.json')
    pca_fit_list = []
    for human in list_bodys:
        pca_fit_list.append(human.keypoints.flatten())
    list_bodies_stacked = np.vstack(pca_fit_list)
    sklearn_pca_fitted = PCA(n_components=opt.pca)
    sklearn_pca_fitted.fit(list_bodies_stacked)

    list_pca_predict = []
    for body in list_bodys:
        list_pca_predict.append(body.keypoints.flatten())
    list_pca_predict = np.vstack(list_pca_predict)
    list_fitted = sklearn_pca_fitted.transform(list_pca_predict)

    g = mixture.GaussianMixture(n_components=opt.clusters, max_iter=100, covariance_type='diag')
    return sklearn_pca_fitted, g.fit(list_fitted)


def pca_predict(bodys, sklearn_pca):
    list_pca_predict = []
    for body in bodys:
        list_pca_predict.append(body.keypoints.flatten())
    list_pca_predict = np.vstack(list_pca_predict)
    return sklearn_pca.transform(list_pca_predict)


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
        pca_fit, gmm_fit = fit_model()

        a = read_body_json('../examples/data/activity_modeling/c1-result.json')
        a = pca_predict(a, pca_fit)
        b = read_body_json('../examples/data/activity_modeling/c3-result.json')
        b = pca_predict(b, pca_fit)
        c = read_body_json('../examples/data/activity_modeling/c5-result.json')
        c = pca_predict(c, pca_fit)
        d = read_body_json('../examples/data/activity_modeling/c6-result.json')
        d = pca_predict(d, pca_fit)
        e = read_body_json('../examples/data/activity_modeling/s4-result.json')
        e = pca_predict(e, pca_fit)

        list_bodys = read_body_json('../examples/data/activity_modeling/c1-result.json') + \
                     read_body_json('../examples/data/activity_modeling/c6-result.json') + \
                     read_body_json('../examples/data/activity_modeling/c3-result.json') + \
                     read_body_json('../examples/data/activity_modeling/c5-result.json')

        list_bodys = pca_predict(list_bodys, pca_fit)
        print(predict_model(e, gmm_fit))
        """
        plot_distribuition(a,b,c,d)

        visualize_2D_gmm(list_bodys, gmm_fit.weights_, gmm_fit.means_.T, np.sqrt(gmm_fit.covariances_).T)
        """