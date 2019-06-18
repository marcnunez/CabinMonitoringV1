import os
import json
import h5py as h5
from eval import parse_keypoints_to_array_no_coinf
import numpy as np
from rectangle import Rectangle, compute_bb
import cv2


def create_h5(dir_json):
    count = 0
    box_list = np.zeros((275,1,4))
    name_list = np.zeros((275,16))
    parts_list = np.zeros((275,17,2))
    for partial_path in os.listdir(dir_json):
        json_path = os.path.join(dir_json, partial_path)
        with open(json_path) as json_file:
            data = json.load(json_file)
            for annotation in data:
                keypoints = parse_keypoints_to_array_no_coinf(annotation['keypoints'])
                box_list = set_box_attributes(keypoints, count, box_list)
                parts_list[count, :, :] = keypoints
                renamed_image = rename(annotation['image_id'])
                name_list[count,:] = np.array([ord(c) for c in renamed_image])
                count +=1
    dset = h5.File("annot_cabin.h5", 'w')
    dset.create_dataset("bndbox", (275, 1, 4), data=box_list)
    dset.create_dataset("imgname", (275, 16), data=name_list)
    dset.create_dataset("parts", (275, 17, 2), data=parts_list)
    dset.close()


# change the format of the images in order to adapt it to the h5
def rename_images():
    dir_path = "../examples/out/datasetsTest/frames"
    for partial_path in os.listdir(dir_path):
        img_path = os.path.join(dir_path, partial_path)
        img = cv2.imread(img_path)
        partial_path = rename(partial_path)
        final_path = os.path.join(dir_path, partial_path)
        cv2.imwrite(final_path, img)


# refactor the name of the images
def rename(partial_path):
    partial_path = partial_path.split(".")[0]
    partial_path = partial_path.split("frame")[1]
    while len(partial_path) < 12:
        partial_path = "0" + partial_path
    partial_path = partial_path + ".jpg"
    return partial_path


def set_box_attributes(keypoints, count, box_list) -> np.array:
    rec = compute_bb(keypoints)
    box = np.zeros((1, 4))
    box[:, 0] = rec.top_left[0]
    box[:, 1] = rec.top_left[1]
    box[:, 2] = rec.get_bottom_right()[0]
    box[:, 3] = rec.get_bottom_right()[1]
    box_list[count, :, :] = box
    return box_list


create_h5("../examples/out/datasetsTest/anotations")
