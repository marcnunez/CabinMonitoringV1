#  Copyright (c) 2019. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.
import numpy as np
import os
import json
import pandas as pd

from sklearn import mixture, cluster, neighbors, svm
from sklearn.decomposition import PCA


from utils.eval import parse_keypoints_to_array
from utils.memory import memory
from opt import opt
from utils.model.rectangle import compute_bb
from utils.model.results import Result
from utils.plot_model import plot_distribuition, visualize_2D_gmm, plot_color_gradients, plot_color_3dmap


class BodyModel:

    def __init__(self, image_id: str, keypoints):
        self.image_id = image_id
        keypoints = parse_keypoints_to_array(keypoints)
        self.original_keypoints = keypoints[:, 0:2]
        self.rectangle = compute_bb(keypoints)
        self.upper_center_mass = self.set_upper_center_mass(keypoints)
        self.lower_center_mass = self.set_lower_center_mass(keypoints)
        self.mid_center_mass = self.set_mid_center_mass()
        self.orientation = self.set_orientation(keypoints)
        self.keypoints = self.set_relative_keypoints(keypoints)
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

        keypoints[:, 0] = keypoints[:, 0]-self.upper_center_mass[0]
        keypoints[:, 1] = keypoints[:, 1]-self.upper_center_mass[1]

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
    out = {}
    bad_beheivour_out = []
    good_behaviour_out = []
    body_list = []
    pca_fit, gmm_fit = fit_model()
    dictionary_gaussian = set_gaussian_beaheivours(pca_fit, gmm_fit)

    for sk in results:
        keypoints = sk['keypoints'].numpy()
        body_list.append(BodyModel("", keypoints))
    if body_list:
        pca_bodys = pca_predict(body_list, pca_fit)
        if opt.model_name == "neighbours":
            fitted_demo = gmm_fit.kneighbors(pca_bodys)
        else:
            fitted_demo = gmm_fit.predict(pca_bodys)
        for i in range(len(fitted_demo)):
            if dictionary_gaussian[fitted_demo[i]]:
                print("something Wrong")
                bad_beheivour_out.append(body_list[i].rectangle)
            else:
                good_behaviour_out.append(body_list[i].rectangle)
    out["correct"] = good_behaviour_out
    out["incorrect"] = bad_beheivour_out
    return out


@memory.cache()
def fit_model(pca_components=opt.pca, model_components=opt.clusters, model_name=opt.model_name):
    list_bodys = read_body_json('models/activity_modeling/c1-result.json') + \
                 read_body_json('models/activity_modeling/c3-result.json') + \
                 read_body_json('models/activity_modeling/c5-result.json') + \
                 read_body_json('models/activity_modeling/c6-result.json') + \
                 read_body_json('models/activity_modeling/c7-result.json') + \
                 read_body_json('models/activity_modeling/c8-result.json') + \
                 read_body_json('models/activity_modeling/i1-result.json') + \
                 read_body_json('models/activity_modeling/i2-result.json') + \
                 read_body_json('models/activity_modeling/i3-result.json') + \
                 read_body_json('models/activity_modeling/s2-result.json') + \
                 read_body_json('models/activity_modeling/s3-result.json') + \
                 read_body_json('models/activity_modeling/s4-result.json')

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
    g = []
    if model_name == "gmm":
        g = mixture.GaussianMixture(n_components=model_components, max_iter=100, covariance_type='diag')
    elif model_name == "kmeans":
        g = cluster.KMeans(n_clusters=model_components, n_jobs=-1)
    elif model_name == "neighbours":
        g = neighbors.KNeighborsClassifier(n_neighbors=model_components)
    elif model_name == "meanshift":
        g = cluster.MeanShift(bandwidth=model_components)
    elif model_name == "spectral":
        g = cluster.SpectralClustering(n_clusters=model_components)
    elif model_name == "svm":
        g = svm.SCV()

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


@memory.cache()
def set_gaussian_beaheivours(pca_fit, gmm_fit, clusters = opt.clusters):
    bodys = read_body_json('models/activity_modeling/train_processed/full-result.json')
    keypoints_pca = pca_predict(bodys, pca_fit)
    fitted_train = gmm_fit.predict(keypoints_pca)

    dict_abnormal = {}
    out_dict = {}
    dict_total = {}

    for i in range(0, clusters):
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


def get_best_combination(name_model):
    bodys = read_body_json('models/activity_modeling/train_processed/full-result.json')
    bodys2 = read_body_json('models/activity_modeling/test_processed/full-result.json')
    full_results_mean = np.zeros((32, 32))
    full_results_var = np.zeros((32, 32))
    gmm_index = 0
    for gmm_clusters in range(2, 34):
        pca_index = 0
        for pca_dimensions in range(2, 34):
            list_results = np.zeros((20, 1))
            for iteration in range(0, 20):
                pca_fit, gmm_fit = fit_model(pca_dimensions, gmm_clusters, name_model)

                keypoints_pca = pca_predict(bodys, pca_fit)
                keypoints_pca2 = pca_predict(bodys2, pca_fit)

                fitted_train = gmm_fit.predict(keypoints_pca)
                fitted_test = gmm_fit.predict(keypoints_pca2)

                behaivour_dict = set_gaussian_beaheivours(pca_fit, gmm_fit, gmm_clusters)

                res = evaluate_test(fitted_test, bodys2, behaivour_dict)
                list_results[iteration] = res.get_f1_score()
            full_results_mean[pca_index, gmm_index] = round(list_results.mean(), 4)
            full_results_var[pca_index, gmm_index] = round(list_results.var(), 4)

            print("PCA: " + str(pca_dimensions) + " CLUSTERS: " + str(gmm_clusters) + " F1 AVG: " +
                  str(round(list_results.mean(), 4)) + " F1 VAR: " + str(round(list_results.var(), 4)))
            pca_index +=1
        gmm_index +=1

    plot_color_gradients(full_results_mean, 'mean_F1_' + name_model)
    save_excel(full_results_mean, 'mean_F1_' + name_model)
    plot_color_gradients(full_results_var, 'var_F1_' + name_model, 'white')
    save_excel(full_results_var, 'var_F1_' + name_model)


def save_excel(data, filepath):
    df = pd.DataFrame(data)
    df.to_excel(filepath+".xlsx", index=False)

def plot_excel():
    """
    data = pd.read_excel(open("mean_F1_gmm_diag.xlsx", 'rb')).to_numpy()
    #plot_color_3dmap(data, "3d_mean_F1_gmm")
    plot_color_gradients(data, "mean_F1_gmm")
    data = pd.read_excel(open("mean_F1_kmeans.xlsx", 'rb')).to_numpy()
    #plot_color_3dmap(data, "3d_mean_F1_kmeans")
    plot_color_gradients(data, "mean_F1_kmeans")
    data = pd.read_excel(open("var_F1_gmm_diag.xlsx", 'rb')).to_numpy()
    #plot_color_3dmap(data, "3d_var_F1_gmm")
    plot_color_gradients(data, "var_F1_gmm", "white")
    data = pd.read_excel(open("var_F1_kmeans.xlsx", 'rb')).to_numpy()
    #plot_color_3dmap(data, "3d_var_F1_kmeans")
    plot_color_gradients(data, "var_F1_kmeans", "white")
    """

    data = pd.read_excel(open("var_F1_gmm_tied.xlsx", 'rb')).to_numpy()
    plot_color_gradients(data, "var_F1_gmm_tied", "white")

if __name__ == '__main__':
    get_best_combination("kmeans")
