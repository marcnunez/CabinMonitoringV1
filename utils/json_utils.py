import os
import json

import matplotlib.pyplot as plt
from model.anotations import Anotations
import os
import json
import cv2
import numpy as np

def fix_json_format(in_path):
    with open(in_path) as json_file:
        data = json_file.readline()
        data = data.replace("[\"", "[")
        data = data.replace("\"]", "]")
        data = data.replace("\"{", "{")
        data = data.replace("}\"", "}")
        data = data.replace("\\", "")
        print(data)
        parsed = json.loads(data)

        with open(in_path, "w") as out:
            json.dump(parsed, out)
    json_file.close()


def fix_json_format_full_directory(in_path):
    for filename in os.listdir(in_path):
        print(filename)
        path_json = os.path.join(in_path, filename)
        fix_json_format(path_json)


def read_anotations(in_path):
    anotations_list = []
    for filename in os.listdir(in_path):
        json_path = os.path.join(in_path, filename)
        with open(json_path) as json_file:
            data = json.load(json_file)
            for anotation in data:

                sk = Anotations(anotation['image_id'], anotation['category_id'], anotation['score'])
                key = anotation['keypoints']
                sk.set_keypoints(key)
                anotations_list.append(sk.toJSON())
    return anotations_list


def anotate_frame(out_path, in_path):
    for filename in os.listdir(in_path):
        frame_number = filename.split('.')[0]
        json_name = frame_number + ".json"
        json_path = os.path.join(out_path, json_name)

        if not check_processed(json_path):
            print(json_path)

            frame_path = os.path.join(in_path, filename)
            img = plt.imread(frame_path)
            plt.imshow(img)

            plt.pause(1)
            plt.close()
            list_person_anotated = []
            number_person = input("Select Number of Persons...")
            plt.imshow(img)
            for id_person in range(0, int(number_person)):
                person_anotated = Anotations(filename, 1, 1)
                keypoints = plt.ginput(17, timeout=0, show_clicks=True)
                for i in keypoints:
                    person_anotated.add_keypoints(i[0], i[1], 1)

                list_person_anotated.append(person_anotated.toJSON())
                print('Next Person :^)')
            print('Next Frame!')

            plt.close()
            with open(json_path, 'w') as json_file:
                json.dump(list_person_anotated, json_file)


def check_processed(json_name) -> bool:
    return os.path.exists(json_name)


def merge_json(indir, outdir):
    list_anno = read_anotations(indir)
    with open(outdir, 'w') as json_file:
        json.dump(list_anno, json_file)
    fix_json_format(outdir)


def roi_crop_video(in_path):

    for filename in os.listdir(in_path):
        video_name = os.path.join(in_path, filename)
        cap = cv2.VideoCapture(video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = os.path.join(os.path.join(in_path, 'cropped'), filename)
        out = cv2.VideoWriter(out_video, fourcc, 20.0, (1920, 1200))
        count = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                pts = np.array([[170, 980], [1450, 980], [1750, 1200], [0, 1200]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(frame, [pts], (0, 0, 0))
                out.write(frame)
            else:
                count +=1
                print(count)
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    """
    anotate_frame('../examples/zoox/test/anotations2', '../examples/zoox/test/frames2')
    fix_json_format_full_directory('../examples/zoox/test/anotations2')
    merge_json('../examples/zoox/test/anotations2', '../examples/zoox/test/zoox-test2.json')  
   
    anotate_frame('../examples/dummy/anotations', '../examples/dummy/frames')
    """
    roi_crop_video('../examples/data/cabin/videos/p')