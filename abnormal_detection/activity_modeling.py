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

from abnormal_detection.plot_model import plot_distribuition, visualize_2D_gmm
from utils.eval import parse_keypoints_to_array


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


def fit_model(bodys_to_fit_gmm, number_gaussian_components=4):
    g = mixture.GaussianMixture(n_components=number_gaussian_components, max_iter=100, covariance_type='diag')
    return g.fit(bodys_to_fit_gmm)


def predict_model(bodys_to_predict_gmm, model_fitted):
    list_fitted_bodys = model_fitted.predict_proba(bodys_to_predict_gmm)
    count = 0
    for coinfidence in list_fitted_bodys:
        if max(coinfidence) < 0.60:
            count += 1
    return count


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


def demo_webcam_wraper(results):
    body_list = []
    id_img = results['imgname']
    for sk in results['result']:
        keypoints = sk['keypoints'].numpy()
        body_list.append(BodyModel(id_img, keypoints))
    """
    pca_list = pca_fit(body_list, TODO)
    predict_model(pca_list, TODO)
    """


def pca_predict(bodys, sklearn_pca):
    list_pca_predict = []
    for body in bodys:
        list_pca_predict.append(body.keypoints.flatten())
    list_pca_predict = np.vstack(list_pca_predict)
    return sklearn_pca.transform(list_pca_predict)


def pca_fit(bodys, pca_components):
    pca_fit_list = []
    for body in bodys:
        pca_fit_list.append(body.keypoints.flatten())
    list_bodies_predict = np.vstack(pca_fit_list)
    sklearn_pca = PCA(n_components=pca_components)
    sklearn_pca.fit(list_bodies_predict)
    return sklearn_pca


if __name__ == '__main__':
    gaussian_components = 16
    pca_components = 3
    val1 = 3
    if val1 == 1:

        for i in range (1, 20):
            gaussian_components = i*2
            for pca_components in range (1, 20):

                list_bodies = []
                bodys = read_body_json('../examples/data/activity_modeling/c3-result.json')
                for body in bodys:
                    list_bodies.append(body.keypoints.flatten())
                list_bodies = np.vstack(list_bodies)
                model = fit_model(list_bodies, gaussian_components, pca_components)

                list_bodies_predict = []
                bodys = read_body_json('../examples/data/activity_modeling/i3-result.json')
                for body in bodys:
                    list_bodies_predict.append(body.keypoints.flatten())
                list_bodies_predict = np.vstack(list_bodies_predict)
                incorrects = predict_model(list_bodies_predict, model, pca_components)
                print(incorrects, " : ", len(list_bodies_predict), " : ", gaussian_components, " : ", pca_components)
    elif val1 == 2:
        list_bodies = []
        bodys = read_body_json('../examples/data/activity_modeling/c3-result.json')
        for body in bodys:
            list_bodies.append(body.keypoints.flatten())
        list_bodies = np.vstack(list_bodies)
        model = fit_model(list_bodies, gaussian_components, pca_components)

        list_bodies_predict = []
        bodys = read_body_json('../examples/data/activity_modeling/i3-result.json')
        for body in bodys:
            list_bodies_predict.append(body.keypoints.flatten())
        list_bodies_predict = np.vstack(list_bodies_predict)
        incorrects = predict_model(list_bodies_predict, model, pca_components)
        print(incorrects, " : ", len(list_bodies_predict), " : ", gaussian_components, " : ", pca_components)
    elif val1 == 3:

        bodys = read_body_json('../examples/data/activity_modeling/c1-result.json')
        bodys = bodys + read_body_json('../examples/data/activity_modeling/c3-result.json')
        bodys = bodys + read_body_json('../examples/data/activity_modeling/c5-result.json')
        bodys = bodys + read_body_json('../examples/data/activity_modeling/c6-result.json')
        sklearn_pca = pca_fit(bodys, pca_components)
        bodys_pca = pca_predict(bodys, sklearn_pca)

        a = read_body_json('../examples/data/activity_modeling/c1-result.json')
        a = pca_predict(a, sklearn_pca)
        b = read_body_json('../examples/data/activity_modeling/c3-result.json')
        b = pca_predict(b, sklearn_pca)
        c = read_body_json('../examples/data/activity_modeling/c5-result.json')
        c = pca_predict(c, sklearn_pca)
        d = read_body_json('../examples/data/activity_modeling/c6-result.json')
        d = pca_predict(d, sklearn_pca)

        e = read_body_json('../examples/data/activity_modeling/s4-result.json')
        e = pca_predict(e, sklearn_pca)


        plot_distribuition(a,b,c,d)

        fitted_gmm = fit_model(bodys_pca, 5)
        visualize_2D_gmm(bodys_pca, fitted_gmm.weights_, fitted_gmm.means_.T, np.sqrt(fitted_gmm.covariances_).T)

