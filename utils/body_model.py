import json
import numpy as np
from keypoints import Keypoints
from keypoint import Keypoint

class BodyModel:

    def __init__(self, image_id: int, category_id: int, score: float):
        self.image_id = image_id
        self.category_id = category_id
        self.keypoints = Keypoints
        self.score = score

    def set_keypoints(self, list_keypoints):
        index = 0
        aux_list = []
        length_list = len(list_keypoints)/3
        for i in range(0, int(length_list)):
            keypoint = Keypoint(list_keypoints[0+index], list_keypoints[1+index], list_keypoints[2+index])
            index = index + 3
            aux_list.append(keypoint)
        self.keypoints = Keypoints(aux_list[0], aux_list[1], aux_list[2], aux_list[3], aux_list[4], aux_list[5],
                                   aux_list[6], aux_list[7], aux_list[8], aux_list[9], aux_list[10], aux_list[11],
                                   aux_list[12], aux_list[13], aux_list[14], aux_list[15], aux_list[16])


def set_body_model(path_json):
    with open(path_json) as json_file:
        data = json.load(json_file)
        for anotation in data['annotations']:
            body = BodyModel(anotation['image_id'], anotation['category_id'], 0)
            body.set_keypoints(anotation['keypoints'])
            print()


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

set_body_model("../person_keypoints_val2014.json")

