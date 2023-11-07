import os
import argparse

import cv2
import numpy as np
import pandas as pd

from oc_sort.ocsort import OCSort

from utils.box_utils import draw_bounding_boxes

SHOW_IMAGE = True

mot_tracker = OCSort(det_thresh=0.6, iou_threshold=0.3, use_byte=False)


def is_not_empty_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Detection (OC SORT)")
    parser.add_argument('--video', type=int, default=0,
                        help='Video Index: 0-20')
    parser.add_argument('--dataset',
                        default='kitti',
                        const='kitti',
                        nargs='?',
                        choices=['kitti', 'carla'],
                        help='Evaluation Dataset (default: %(default)s)')

    args = parser.parse_args()

    DATASET = args.dataset
    GT_FOLDER = os.path.join(os.path.abspath(os.path.join(
        os.path.dirname(__file__))), f'data/gt/{DATASET}/{DATASET}_2d_box_train/')
    TRACKERS_FOLDER = os.path.join(os.path.abspath(os.path.join(
        os.path.dirname(__file__))), f'data/trackers/{DATASET}/{DATASET}_2d_box_train/')

    f_video = f'./data/video/{DATASET}/{args.video:04d}.mp4'
    print(f"Reading {DATASET} Video:", f_video)

    f_label = os.path.join(GT_FOLDER, 'label_02', f'{args.video:04d}.txt')
    print(f"Reading {DATASET} Label:", f_label)

    gt_labels = None
    if is_not_empty_file(f_label):
        gt_labels = pd.read_csv(f_label, header=None, sep=' ')
    else:
        print("Empty label file:", f_label)

    vid = cv2.VideoCapture(f_video)

    if (vid.isOpened() == False):
        print("Error opening the video file:", f_video)
        exit(1)

    OUT_FILE = os.path.join(TRACKERS_FOLDER, 'GT-OC-SORT',
                            'data', '{0:04d}.txt'.format(args.video))
    if not os.path.exists(os.path.dirname(OUT_FILE)):
        # Create a new directory if it does not exist
        os.makedirs(os.path.dirname(OUT_FILE))
    try:
        f_tracker = open(OUT_FILE, "w+")
    except OSError:
        print("Could not open file:", OUT_FILE)
        exit(1)

    # Read until video is completed
    i_frame = 0
    while (vid.isOpened()):
        # Capture frame-by-frame
        ret, frame = vid.read()

        # Labels for the current frame
        if gt_labels is not None:
            c_labels = gt_labels[gt_labels[0] == i_frame]
            c_labels = c_labels[c_labels[1] != -1]
            c_labels = c_labels[(c_labels[2] == 'Van') |
                                (c_labels[2] == 'Car')]
        else:
            c_labels = pd.DataFrame([])

        if ret == True:
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
                dets = np.concatenate((np.array(boxes), np.array(
                    probs).reshape((len(probs), -1))), axis=1)

                # Update tracker
                trackers = mot_tracker.update(dets, [height, width], (height, width))

                # convert [x1, y1, x2, y2] to [x, y, w, h ]
                for track in trackers:
                    f_tracker.write(
                        f'{i_frame} {int(track[4])} Car -1.000000 -1 -1 {track[0]} {track[1]} {track[2]} {track[3]} -1 -1 -1 -1 -1 -1 -1 -1 1 \n')
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

                    # vertical = tlwh[2] / tlwh[3] > 1.6
                    # if tlwh[2] * tlwh[3] > 20 and not vertical:
                    #     online_tlwhs.append(tlwh)
                    #     online_ids.append(tid)

                # Draw bounding boxes onto the image
                draw_bounding_boxes(
                    frame, trackers[:, 0:4], labels, trackers[:, 4])

            i_frame = i_frame + 1

            if SHOW_IMAGE:
                # Display the resulting frame
                cv2.imshow('Frame', frame)

                # Press Q on keyboard to  exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break

    vid.release()
    cv2.destroyAllWindows()
