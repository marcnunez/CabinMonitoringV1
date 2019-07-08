import argparse

import numpy as np
import json
import os
import statistics

from functional import seq
from model.rectangle import Rectangle, compute_bb
from plot_results import plot_boxes
from model.results import Result


parser = argparse.ArgumentParser(description='PyTorch CabinMonitoringV1 Evaluation')

"----------------------------- General options -----------------------------"
parser.add_argument('--anotations', default='../examples/zoox/res/alphapose-results.json', type=str,
                    help='Path to anotations ')
parser.add_argument('--groundTruth', default='../examples/zoox/test/zoox-test2.json', type=str,
                    help='Path to ground turth ')
parser.add_argument('--debug', default=False, type=bool,
                    help='Print the debug information')
parser.add_argument('--oksThreshold', default=0.5, type=float,
                    help='Threshold between 0-1 of mimum OKS')
parser.add_argument('--iouThreshold', default=0.5, type=float,
                    help='Threshold between 0-1 of mimum IOU')

opt = parser.parse_args()


# Evaluate mean OKS of those skeletons that have an IOU over a threshold
def eval_oks_iou():
    compare_OKS = []
    counter = 0
    with open(opt.anotations) as json_file:
        data = json.load(json_file)

        with open(opt.groundTruth) as gt_file:
            gt_data = json.load(gt_file)

            for anotation in data:
                keypoints = parse_keypoints_to_array(anotation['keypoints'])
                rectangle = compute_bb(keypoints)
                list_related_kp = list(filter(lambda gt: gt['image_id'] == anotation['image_id'], gt_data))
                res_iou = 0
                res_gt_keypoints = np.zeros((0))

                for gt_anotated in list_related_kp:
                    keypoints_gt = parse_keypoints_to_array(gt_anotated['keypoints'])
                    rectangle_gt = compute_bb(keypoints_gt)
                    iou = rectangle.iou(rectangle_gt)

                    if opt.debug:
                        path_image = os.path.join("../examples/zoox/res/vis", anotation['image_id'])
                        plot_boxes(path_image, rectangle, rectangle_gt)

                    if iou > res_iou and iou > opt.iouThreshold:
                        res_iou = iou
                        res_gt_keypoints = keypoints_gt

                if res_iou!=0:
                    oks = compute_oks(res_gt_keypoints, keypoints, DELTA)
                    compare_OKS.append(oks)
                else:
                    counter +=1
    return statistics.mean(compare_OKS)


# Obtain FP, TN, FN, TP from all the dataset at the same time
def full_results():
    tp = 0
    with open(opt.anotations) as json_file:
        data = json.load(json_file)
        count_detections = len(data)

        with open(opt.groundTruth) as gt_file:
            gt_data = json.load(gt_file)
            count_gt = len(gt_data)

            for anotation in data:
                keypoints = parse_keypoints_to_array(anotation['keypoints'])
                list_related_kp = list(filter(lambda gt: gt['image_id'] == anotation['image_id'], gt_data))

                for gt_anotated in list_related_kp:
                    keypoints_gt = parse_keypoints_to_array(gt_anotated['keypoints'])
                    oks = compute_oks(keypoints_gt, keypoints, DELTA)
                    if oks > opt.oksThreshold:
                        tp += 1
                        break

    gt_file.close()
    json_file.close()

    fp = count_detections - tp
    fn = count_gt - tp

    return Result(tp, 0, fp, fn)


# Compute
def eval_json_ap():
    res_total = full_results()

    ap = []
    processed_frame = []

    running_tp = 0
    running_total = 0
    with open(opt.groundTruth) as gt_file:
        with open(opt.anotations) as det_file:
            det_data = json.load(det_file)
            gt_data = json.load(gt_file)

            for gt_anotation in gt_data:
                if gt_anotation['image_id'] not in processed_frame:
                    processed_frame.append(gt_anotation['image_id'])
                    list_gt_kp = list(filter(lambda gt: gt['image_id'] == gt_anotation['image_id'], gt_data))
                    list_det_kp = list(filter(lambda det: det['image_id'] == gt_anotation['image_id'], det_data))
                    res = eval_mAP_oks(list_gt_kp, list_det_kp)
                    running_tp += res.tp
                    running_total += res.tp + res.fp
                    if running_total == 0:
                        continue
                    ap.append((running_tp / running_total, running_tp / (res_total.tp + res_total.fn)))

    summation = 0
    max_recall = seq(ap).max_by(lambda p_r: p_r[1])[1]
    for recall_th in np.linspace(0, 1, 11):
        if recall_th <= max_recall:
            summation += seq(ap).filter(lambda p_r: p_r[1] >= recall_th).max_by(lambda p_r: p_r[0])[0]/11

    gt_file.close()
    det_file.close()
    return summation


def eval_mAP_oks(gt, det) -> Result:
    tp = 0
    for ground_truth in gt:
        ground_truth_keypoints = parse_keypoints_to_array(ground_truth['keypoints'])
        for detection in det:
            detection_keypoints = parse_keypoints_to_array(detection['keypoints'])
            if compute_oks(ground_truth_keypoints, detection_keypoints, DELTA) > opt.oksThreshold:
                tp += 1
                break
    fp = len(det) - tp
    fn = len(gt) - tp
    res = Result(tp, 0, fp, fn)
    return res


# Parse Keypoints from 51 by 1 list to 17 by 3 numpy array
def parse_keypoints_to_array(keypoints) -> np.array:
    counter = 1
    index = 0
    out = np.zeros((17, 3))
    for keypoint in keypoints:
        # Get X's keypoinys
        if ((counter+2) % 3) == 0:
            out[index, 0] = keypoint
        # Get Y's keypoinys
        elif ((counter+1) % 3) == 0:
            out[index, 1] = keypoint
        # Get coinfidence
        else:
            out[index, 2] = keypoint
            index += 1
        counter += 1
    return out


# Parse Keypoints from 51 by 1 list to 17 by 3 numpy array
def parse_keypoints_to_array_no_coinf(keypoints) -> np.array:
    counter = 1
    index = 0
    out = np.zeros((17, 2))
    for keypoint in keypoints:
        # Get X's keypoinys
        if ((counter+2) % 3) == 0:
            out[index, 0] = keypoint
        # Get Y's keypoinys
        elif ((counter+1) % 3) == 0:
            out[index, 1] = keypoint
            index +=1

        counter += 1
    return out


# calculate OKS between two single poses
def compute_oks(anno, predict, delta):
    xmax = np.max(np.vstack((anno[:, 0], predict[:, 0])))
    xmin = np.min(np.vstack((anno[:, 0], predict[:, 0])))
    ymax = np.max(np.vstack((anno[:, 1], predict[:, 1])))
    ymin = np.min(np.vstack((anno[:, 1], predict[:, 1])))
    scale = (xmax - xmin) * (ymax - ymin)
    dis = np.sum((anno - predict) ** 2, axis=1)
    oks = np.mean(np.exp(-dis / 2 / delta ** 2 / scale))
    return oks


if __name__ == '__main__':
    for delta in range(1, 10, 1):
        DELTA = delta * 0.1
        summation_map = eval_json_ap()
        print("mAP OKS 0.5 : ", summation_map, " Delta : ", DELTA)
        mean_oks = eval_oks_iou()
        print("mean OKS IoU 0.5 : ", mean_oks, " Delta : ", DELTA)