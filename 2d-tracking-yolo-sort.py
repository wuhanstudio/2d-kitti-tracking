import argparse

import pandas as pd

import cv2
import numpy as np

from sort.sort import Sort
from utils.box_utils import draw_bounding_boxes

## Part 0: Object Detection model

from what.models.detection.datasets.coco import COCO_CLASS_NAMES
from what.models.detection.yolo.yolov4 import YOLOV4
from what.models.detection.yolo.yolov4_tiny import YOLOV4_TINY

from what.cli.model import *
from what.utils.file import get_file

# Check what_model_list for all supported models
what_yolov4_model_list = what_model_list[4:6]

index = 0 # YOLOv4
# index = 1 # YOLOv4 Tiny

# Download the model first if not exists
WHAT_YOLOV4_MODEL_FILE = what_yolov4_model_list[index][WHAT_MODEL_FILE_INDEX]
WHAT_YOLOV4_MODEL_URL  = what_yolov4_model_list[index][WHAT_MODEL_URL_INDEX]
WHAT_YOLOV4_MODEL_HASH = what_yolov4_model_list[index][WHAT_MODEL_HASH_INDEX]

if not os.path.isfile(os.path.join(WHAT_MODEL_PATH, WHAT_YOLOV4_MODEL_FILE)):
    get_file(WHAT_YOLOV4_MODEL_FILE,
             WHAT_MODEL_PATH,
             WHAT_YOLOV4_MODEL_URL,
             WHAT_YOLOV4_MODEL_HASH)

# Darknet
model = YOLOV4(COCO_CLASS_NAMES, os.path.join(WHAT_MODEL_PATH, WHAT_YOLOV4_MODEL_FILE))
# model = YOLOV4_TINY(COCO_CLASS_NAMES, os.path.join(WHAT_MODEL_PATH, WHAT_YOLOV4_MODEL_FILE))

mot_tracker = Sort( max_age=1, 
                    min_hits=3,
                    iou_threshold=0.3) #create instance of the SORT tracker

GT_FOLDER = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))), 'data/gt/kitti/kitti_2d_box_train/')
TRACKERS_FOLDER = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__))), 'data/trackers/kitti/kitti_2d_box_train/')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D KITTI Detection (Ground Truth)")
    parser.add_argument('--video', type=int, default=0, help='KITTI MOT Video Index: 0-20')

    args = parser.parse_args()

    f_video = './data/video/{0:04d}.mp4'.format(args.video)
    print("Reading KITTI Video:", f_video)

    f_label = os.path.join(GT_FOLDER, 'label_02', '{0:04d}.txt'.format(args.video))
    print("Reading KITTI Label:", f_label)

    gt_labels = pd.read_csv(f_label, header=None, sep=' ')

    vid = cv2.VideoCapture(f_video)

    if (vid.isOpened()== False): 
        print("Error opening the video file")
        exit(1)

    OUT_FILE = os.path.join(TRACKERS_FOLDER, 'YOLO-SORT', 'data', '{0:04d}.txt'.format(args.video))
    try:
        f_tracker = open(OUT_FILE, "w+")
    except OSError:
        print("Could not open file:", OUT_FILE)
        exit(1)

    # Read until video is completed
    i_frame = 0
    while(vid.isOpened()):
        # Capture frame-by-frame
        ret, frame = vid.read()

        if frame is None:
            break;

        # Labels for the current frame
        c_labels = gt_labels[gt_labels[0] == i_frame]
        c_labels = c_labels[c_labels[1] != -1]
        c_labels = c_labels[ (c_labels[2] == 'Van') | (c_labels[2] == 'Car') ]

        if ret == True:
            height, width, _ = frame.shape

            # Image preprocessing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run inference
            images, boxes, labels, probs = model.predict(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Only draw 2: car, 5: bus, 7: truck
            boxes = np.array([box for box, label in zip(boxes, labels) if label in [2, 5, 7]])
            probs = np.array([prob for prob, label in zip(probs, labels) if label in [2, 5, 7]])
            labels = np.array([2 for label in labels if label in [2, 5, 7]])

            # convert [x1, y1, w, h] to [x1, y1, x2, y2]
            if len(boxes) > 0:
                sort_boxes = boxes.copy()

                # (xc, yc, w, h) --> (x1, y1, x2, y2)
                height, width, _ = image.shape

                for box in sort_boxes:
                    box[0] *= width
                    box[1] *= height
                    box[2] *= width 
                    box[3] *= height

                    # From center to top left
                    box[0] -= box[2] / 2
                    box[1] -= box[3] / 2

                    # From width and height to x2 and y2
                    box[2] += box[0]
                    box[3] += box[1]

                labels = ['Car'] * len(boxes)
                dets = np.concatenate((np.array(sort_boxes), np.array(probs).reshape((len(probs), -1))), axis=1)

                # Update tracker
                trackers = mot_tracker.update(dets)

                # convert [x1, y1, x2, y2] to [xc, yc, w, h ]
                for track in trackers:
                    f_tracker.write(f'{i_frame} {int(track[4])} Car -1.000000 -1 -1 {track[0]} {track[1]} {track[2]} {track[3]} -1 -1 -1 -1 -1 -1 -1 -1 1 \n')
                    f_tracker.flush()

                    # From x2 and y2 to width and height
                    track[2] -= track[0]
                    track[3] -= track[1]

                    # From top left to center
                    track[0] += track[2] / 2
                    track[1] += track[3] / 2

                    track[0] /= width
                    track[1] /= height
                    track[2] /= width
                    track[3] /= height

                # Draw bounding boxes onto the image
                draw_bounding_boxes(frame, trackers[:, 0:4], labels, trackers[:, 4])

            # Display the resulting frame
            cv2.imshow('Frame', frame)
            i_frame = i_frame + 1

            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else: 
            break

    f_tracker.close()
    vid.release()
    cv2.destroyAllWindows()
