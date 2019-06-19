import numpy as np
import json
import os
import statistics

from rectangle import Rectangle, compute_bb


THRESOLD_IOU = 0.7
DELTA = 0.2

def eval(out, ground_truth):
    compare_OKS = []
    with open(out) as json_file:
        data = json.load(json_file)
        for anotation in data:
            keypoints = parse_keypoints_to_array(anotation['keypoints'])
            rec = compute_bb(keypoints)
            keypoints_gt = get_ground_truth_associated(ground_truth, anotation['image_id'], rec )
            if len(keypoints_gt) != 0:
                oks = compute_oks(keypoints_gt, keypoints, DELTA)
                compare_OKS.append(oks)
    print(statistics.mean(compare_OKS))


def get_ground_truth_associated(ground_truth, id_frame: str, rec):
    id_frame = id_frame.split(".")[0]+".json"
    in_dir = os.path.join(ground_truth, id_frame)
    res_iou = 0
    res_gt = np.zeros((0))
    with open(in_dir) as json_ground_truth:
        data = json.load(json_ground_truth)
        for anotation in data :
            gt = parse_keypoints_to_array(anotation['keypoints'])
            rec_gt = compute_bb(gt)
            iou = rec_gt.iou(rec)
            if iou>res_iou and iou>THRESOLD_IOU:
                res_iou = iou
                res_gt = gt
    return res_gt


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

out = "../examples/zoox/res/alphapose-results.json"
ground_truth = "../examples/zoox/test/anotations"

eval(out, ground_truth)
