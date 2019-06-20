import cv2
import numpy as np


def plot_boxes(path_image, bb_detect, bb_gt):
    image = cv2.imread(path_image)
    cv2.rectangle(image, bb_detect.parse_int(bb_detect.top_left), bb_detect.parse_int(bb_detect.get_bottom_right()), (0, 255, 0))
    cv2.rectangle(image, bb_detect.parse_int(bb_gt.top_left), bb_detect.parse_int(bb_gt.get_bottom_right()), (255, 0, 0))
    cv2.imshow("Boundig Box (Green Detect, Red GT)", image)
    cv2.waitKey(1)

