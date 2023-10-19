import argparse

import pandas as pd

import cv2
import numpy as np

from sort.sort import Sort
from utils.box_utils import draw_bounding_boxes

mot_tracker = Sort( max_age=1, 
                    min_hits=3,
                    iou_threshold=0.3) #create instance of the SORT tracker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D KITTI Detection (Ground Truth)")
    parser.add_argument('--video', type=int, default=0, help='KITTI MOT Video Index: 0-20')

    args = parser.parse_args()

    f_video = './data/{0:04d}.mp4'.format(args.video)
    print("Reading KITTI Video:", f_video)

    f_label = './data/gt/kitti/kitti_2d_box_train/label_02/{0:04d}.txt'.format(args.video)
    print("Reading KITTI Label:", f_label)

    gt_labels = pd.read_csv(f_label, header=None, sep=' ')

    vid = cv2.VideoCapture(f_video)

    if (vid.isOpened()== False): 
        print("Error opening the video file")
        exit(1)

    # Read until video is completed
    i_frame = 0
    while(vid.isOpened()):
        # Capture frame-by-frame
        ret, frame = vid.read()

        # Labels for the current frame
        c_labels = gt_labels[gt_labels[0] == i_frame]
        c_labels = c_labels[c_labels[1] != -1]
        c_labels = c_labels[ (c_labels[2] == 'Van') | (c_labels[2] == 'Car') ]

        if ret == True:
            i_frame = i_frame + 1
            height, width, _ = frame.shape

            # Draw Bounding Boxes
            ids = []
            boxes = []
            probs = []
            for _, c_label in c_labels.iterrows():
                x1, y1, x2, y2 = c_label[6], c_label[7], c_label[8], c_label[9]
                boxes.append(np.array([x1, y1, x2, y2]))
                ids.append(c_label[1])
                probs.append(1.0)

            if len(boxes) > 0:
                labels = ['Car'] * len(boxes)
                dets = np.concatenate((np.array(boxes), np.array(probs).reshape((len(probs), -1))), axis=1)

                # Update tracker
                trackers = mot_tracker.update(dets)

                # convert [x1, y1, x2, y2] to [x, y, w, h ]
                for track in trackers:
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

            # Press Q on keyboard to  exit
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else: 
            continue

    vid.release()
    cv2.destroyAllWindows()
