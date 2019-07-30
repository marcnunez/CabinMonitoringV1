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
from utils.model.results import Result
from utils.plot_model import plot_distribuition, visualize_2D_gmm


class BodyModel:

    def __init__(self, image_id: str, keypoints):
        self.image_id = image_id
        keypoints = parse_keypoints_to_array(keypoints)
        self.upper_center_mass = self.set_upper_center_mass(keypoints)
        self.lower_center_mass = self.set_lower_center_mass(keypoints)
        self.mid_center_mass = self.set_mid_center_mass()
        self.orientation = self.set_orientation(keypoints)
        self.keypoints = self.set_relative_keypoints(keypoints)
        self.original_keypoints = keypoints[:, 0:2]
        self.is_abnormal = self.mark_abnormal_beheaivour(image_id)

    @staticmethod
    def mark_abnormal_beheaivour(image_id:str) -> bool:
        return image_id.startswith('i')

    @staticmethod
    def set_upper_center_mass(keypoints):
        return (keypoints[5, 0]+keypoints[6, 0])/2, (keypoints[5, 1]+keypoints[6, 1])/2

    @staticmethod
    def set_lower_center_mass(keypoints):
        return (keypoints[11, 0]+keypoints[12, 0])/2, (keypoints[11, 1]+keypoints[12, 1])/2

    def set_mid_center_mass(self):
        return (self.upper_center_mass[0] + self.lower_center_mass[0]) / 2, (self.upper_center_mass[1] + self.lower_center_mass[1]) / 2

    @staticmethod
    def set_orientation(keypoints):
        return np.arctan2(keypoints[11, 0]-keypoints[12, 0], keypoints[11, 1]-keypoints[12, 1])

    def set_relative_keypoints(self, keypoints):
        keypoints = keypoints[:, 0:2]


        """
        #INVARIANT TO ROTATION
        rotation_matrix = [[np.cos(self.orientation), -np.sin(self.orientation)], [np.sin(self.orientation),
                                                                                   np.cos(self.orientation)]]
        for i in range(0, len(keypoints)):
            keypoints[i, :] = np.dot(rotation_matrix, keypoints[i,:])
        """

        keypoints[:, 0] = keypoints[:, 0]-self.mid_center_mass[0]
        keypoints[:, 1] = keypoints[:, 1]-self.mid_center_mass[1]


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
        wrong_beheivours = predict_model(pca_bodys, gmm_fit)
        if wrong_beheivours !=0:
            print("There is something Wrong, maybe :P")


def predict_model(bodys_to_predict_gmm, model_fitted):
    list_fitted_bodys = model_fitted.predict(bodys_to_predict_gmm)
    count = 0
    for index in list_fitted_bodys:
        if index == 4:
            count +=1
    return


@memory.cache()
def fit_model(pca_components=opt.pca, model_components=opt.clusters):
    list_bodys = read_body_json('examples/data/activity_modeling/c1-result.json') + \
                 read_body_json('examples/data/activity_modeling/c3-result.json') + \
                 read_body_json('examples/data/activity_modeling/c5-result.json') + \
                 read_body_json('examples/data/activity_modeling/c6-result.json') + \
                 read_body_json('examples/data/activity_modeling/c7-result.json') + \
                 read_body_json('examples/data/activity_modeling/c8-result.json') + \
                 read_body_json('examples/data/activity_modeling/i1-result.json') + \
                 read_body_json('examples/data/activity_modeling/i2-result.json') + \
                 read_body_json('examples/data/activity_modeling/i3-result.json') + \
                 read_body_json('examples/data/activity_modeling/s2-result.json') + \
                 read_body_json('examples/data/activity_modeling/s3-result.json') + \
                 read_body_json('examples/data/activity_modeling/s4-result.json')

    pca_fit_list = []
    for human in list_bodys:
        pca_fit_list.append(human.keypoints.flatten())
    list_bodies_stacked = np.vstack(pca_fit_list)
    sklearn_pca_fitted = PCA(n_components=pca_components)
    sklearn_pca_fitted.fit(list_bodies_stacked)

    list_pca_predict = []
    for body in list_bodys:
        list_pca_predict.append(body.keypoints.flatten())
    list_pca_predict = np.vstack(list_pca_predict)
    list_fitted = sklearn_pca_fitted.transform(list_pca_predict)

    g = mixture.GaussianMixture(n_components=model_components, max_iter=100, covariance_type='diag')
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


def set_gaussian_beaheivours(fitted_train, bodys):
    dict_abnormal = {}
    out_dict = {}
    dict_total= {}
    for i in range(0, gmm_fit.n_components):
        dict_abnormal[i] = 0
        dict_total[i] = 0
        out_dict[i] = False
    for i in range(len(fitted_train)):
        if bodys[i].is_abnormal:
            dict_abnormal[fitted_train[i]] = dict_abnormal[fitted_train[i]] + 1
        dict_total[fitted_train[i]] = dict_total[fitted_train[i]] + 1
    for i in range(len(dict_abnormal)):
        if dict_abnormal[i] > dict_total[i]/2:
            out_dict[i] = True
    return out_dict


def evaluate_test(fitted_test, bodys2, behaivour_dict) -> Result:
    res = Result()

    for i in range(len(fitted_test)):
        if behaivour_dict[fitted_test[i]] == bodys2[i].is_abnormal:
            if bodys2[i].is_abnormal:
                res.set_tp(res.tp + 1)
            else:
                res.set_tn(res.tn + 1)
        else:
            if bodys2[i].is_abnormal:
                res.set_fn(res.fn + 1)
            else:
                res.set_fp(res.fp + 1)
    return res

if __name__ == '__main__':

    bodys = read_body_json('examples/data/activity_modeling/images/train_processed/full-result.json')
    bodys2 = read_body_json('examples/data/activity_modeling/images/test_processed/full-result.json')
    for pca_dimensions in range(3, 15):
        for gmm_clusters in range(3, 15):
            pca_fit, gmm_fit = fit_model(pca_dimensions, gmm_clusters)

            keypoints_pca = pca_predict(bodys, pca_fit)
            keypoints_pca2 = pca_predict(bodys2, pca_fit)

            fitted_train = gmm_fit.predict(keypoints_pca)
            fitted_test = gmm_fit.predict(keypoints_pca2)

            behaivour_dict = set_gaussian_beaheivours(fitted_train, bodys)

            res = evaluate_test(fitted_test, bodys2, behaivour_dict)

            print("PCA: " + str(pca_dimensions) + " GMM: " + str(gmm_clusters) + " F1: " + str(res.get_f1_score()))