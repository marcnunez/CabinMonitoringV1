from typing import List

from results import Result
import numpy as np


class Frame:

    def __init__(self, id: int):
        self.id = id
        self.detections = []
        self.ground_truth = []
        self.cached_result = None

    def get_detection_iou(self, ignore_classes=False) -> List[float]:
        ret = []
        for ground_truth in self.ground_truth:
            max_iou = 0
            for detection in self.detections:
                iou = detection.iou(ground_truth)
                if (ignore_classes or detection.label == ground_truth.label) and iou > max_iou:
                    max_iou = iou

            ret.append(max_iou)
        return ret

    def get_detection_iou_mean(self, ignore_classes=False) -> float:
        iou_list = self.get_detection_iou(ignore_classes)
        if len(iou_list) > 0:
            return float(np.mean(iou_list))
        else:
            return 0

    def to_result(self, ignore_classes=False) -> Result:
        if self.cached_result is None:
            tp = 0
            for ground_truth in self.ground_truth:
                for detection in self.detections:
                    if detection.iou(ground_truth) > 0.5 and (ignore_classes or detection.label == ground_truth.label):
                        tp += 1
                        break


            fp = len(self.detections) - tp
            fn = len(self.ground_truth) - tp
            self.cached_result = Result(tp, fp, 0, fn)

        return self.cached_result

    # calculate OKS between two single poses
    def compute_oks(self, delta=1):
        xmax = np.max(np.vstack((self.ground_truth[:, 0], self.detections[:, 0])))
        xmin = np.min(np.vstack((self.ground_truth[:, 0], self.detections[:, 0])))
        ymax = np.max(np.vstack((self.ground_truth[:, 1], self.detections[:, 1])))
        ymin = np.min(np.vstack((self.ground_truth[:, 1], self.detections[:, 1])))
        scale = (xmax - xmin) * (ymax - ymin)
        dis = np.sum((self.ground_truth - self.detections) ** 2, axis=1)
        oks = np.mean(np.exp(-dis / 2 / delta ** 2 / scale))
        return oks