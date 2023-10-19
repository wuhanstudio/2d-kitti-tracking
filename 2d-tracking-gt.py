import argparse

import pandas as pd

import cv2
import numpy as np

from utils.box_utils import draw_bounding_boxes

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

        if ret == True:
            i_frame = i_frame + 1

            # Draw Bounding Boxes
            labels = []
            ids = []
            boxes = []
            for _, c_label in c_labels.iterrows():
                height, width, _ = frame.shape

                # (x1, y1) (x2, y2) --> (xc, yc), w, h
                x1, y1, x2, y2 = c_label[6], c_label[7], c_label[8], c_label[9]

                w = x2 - x1
                h = y2 - y1
                xc = x1 + w / 2
                yc = y1 + h / 2

                xc = xc / width
                yc = yc / height
                w = w / width
                h = h / height
                
                boxes.append(np.array([xc, yc, w, h]))
                labels.append(c_label[2])
                ids.append(c_label[1])

            draw_bounding_boxes(frame, np.array(boxes), labels, ids);

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else: 
            continue
        
    vid.release()
    cv2.destroyAllWindows()
