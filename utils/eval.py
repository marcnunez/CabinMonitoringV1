import numpy as np
import json
import os
import statistics

from model.rectangle import Rectangle, compute_bb
from plot_results import plot_boxes
from model.results import Result

THRESOLD_IOU = 0.5
DELTA = 0.1
THRESOLD_OKS = 0.5

def eval_oks_iou(out, ground_truth, debug= False):
    compare_OKS = []
    counter = 0
    with open(out) as json_file:
        data = json.load(json_file)

        with open(ground_truth) as gt_file:
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

                    if debug:
                        path_image = os.path.join("../examples/zoox/res/vis", anotation['image_id'])
                        plot_boxes(path_image, rectangle, rectangle_gt)

                    if iou > res_iou and iou > THRESOLD_IOU:
                        res_iou = iou
                        res_gt_keypoints = keypoints_gt

                if res_iou!=0:
                    oks = compute_oks(res_gt_keypoints, keypoints, DELTA)
                    compare_OKS.append(oks)
                    print(str(oks) + " : " + anotation['image_id'])
                else:
                    counter +=1
                    print(counter)

    print(statistics.mean(compare_OKS))


def eval_oks_ap(out, ground_truth):
    tp = 0
    count_gt =0
    with open(out) as json_file:
        data = json.load(json_file)
        count_detections = len(data)
        with open(ground_truth) as gt_file:
            gt_data = json.load(gt_file)
            count_gt = len(gt_data)
            for anotation in data:
                keypoints = parse_keypoints_to_array(anotation['keypoints'])
                list_related_kp = list(filter(lambda gt: gt['image_id'] == anotation['image_id'], gt_data))

                for gt_anotated in list_related_kp:
                    keypoints_gt = parse_keypoints_to_array(gt_anotated['keypoints'])
                    oks = compute_oks(keypoints_gt, keypoints, DELTA)
                    if oks > THRESOLD_OKS:
                        tp += 1
                        break


    gt_file.close()
    json_file.close()

    fp = count_detections - tp
    fn = count_gt - tp

    return Result(tp, 0, fp, fn)




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
ground_truth = "../examples/zoox/test/zoox-test.json"

#eval_oks_iou(out, ground_truth, False)
r = eval_oks_ap(out, ground_truth)
print()