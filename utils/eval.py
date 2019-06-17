import numpy as np
import json
import os
import statistics
def eval(out, ground_truth):
    compare_OKS = []
    with open(out) as json_file:
        data = json.load(json_file)
        for anotation in data:
            keypoints = get_keypoints_array(anotation['keypoints'])
            compare_OKS.append(related_ground_truth(ground_truth, anotation['image_id'], keypoints))

    print(statistics.mean(compare_OKS))

def related_ground_truth(ground_truth, id_frame: str, out):
    id_frame = id_frame.split(".")[0]+".json"
    in_dir = os.path.join(ground_truth, id_frame)
    res = 0
    with open(in_dir) as json_ground_truth:
        data = json.load(json_ground_truth)
        for anotation in data :
            gt = get_keypoints_array(anotation['keypoints'])
            oks = compute_oks(gt, out, 0.5)
            if res < oks:
                res = oks
    print(str(id_frame)+" : " + str(res))
    return res

# Parse Keypoints from 51 by 1 list to 17 by 3 numpy array
def get_keypoints_array(keypoints) -> np.array:
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

out = "../examples/in/alphapose-results.json"
ground_truth = "../examples/out/datasetsTest/anotations"

eval(out, ground_truth)