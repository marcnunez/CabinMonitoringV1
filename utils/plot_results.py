import cv2
import numpy as np
import os

def plot_boxes(path_image, bb_detect, bb_gt):
    image = cv2.imread(path_image)
    cv2.rectangle(image, bb_detect.parse_int(bb_detect.top_left), bb_detect.parse_int(bb_detect.get_bottom_right()), (0, 255, 0))
    cv2.rectangle(image, bb_detect.parse_int(bb_gt.top_left), bb_detect.parse_int(bb_gt.get_bottom_right()), (255, 0, 0))
    cv2.imshow("Boundig Box (Green Detect, Red GT)", image)
    cv2.waitKey(1)


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
    roi_crop_video('../examples/data/cabin/videos')