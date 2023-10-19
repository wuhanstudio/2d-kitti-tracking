import argparse

import pandas as pd

import cv2
import numpy as np

colors = np.random.randint(0, 256, size=(32, 3))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D KITTI Detection (Ground Truth)")
    parser.add_argument('--video', type=int, default=0, help='KITTI MOT Video Index: 0-20')

    args = parser.parse_args()

    f_video = './data/{0:04d}.mp4'.format(args.video)
    print("Reading KITTI Video:", f_video)

    f_label = './data/gt/kitti/kitti_2d_box_train/label_02/{0:04d}.txt'.format(args.video)
    print("Reading KITTI Label:", f_label)

    labels = pd.read_csv(f_label, header=None, sep=' ')

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
        c_labels = labels[labels[0] == i_frame]
        c_labels = c_labels[c_labels[1] != -1]

        if ret == True:
            i_frame = i_frame + 1

            # Draw Bounding Boxes
            for _, c_label in c_labels.iterrows():
                x1, y1, x2, y2 = int(c_label[6]), int(c_label[7]), int(c_label[8]), int(c_label[9])
                cv2.rectangle(  frame, 
                                (x1, y1),
                                (x2, y2),
                                tuple([int(c) for c in colors[int(c_label[1]) % 32, :]]),
                                2)

                cv2.putText(    frame, 
                                c_label[2] + ' ' + str(c_label[1]), 
                                (x2 + 10, y1), 
                                0, 
                                1, 
                                tuple([int(c) for c in colors[int(c_label[1]) % 32, :]]),
                                2)

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else: 
            break
        
    vid.release()
    cv2.destroyAllWindows()
