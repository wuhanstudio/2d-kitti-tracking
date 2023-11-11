import os
import argparse

import cv2
import numpy as np
import pandas as pd

from utils.box_utils import *
from utils.encorder import *

import time
import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import postprocess, vis

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


SHOW_IMAGE = True

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

    def inference(self, img):
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


exp = get_exp(None, "yolox-x")

ckpt = torch.load("yolox_x.pth", map_location="cpu")
model = exp.get_model()
model.cuda()
model.eval()

# load the model state dict
model.load_state_dict(ckpt["model"])
predictor = Predictor(
        model, exp, COCO_CLASSES, None, None,
        "gpu", False, False,
    )

# Deep SORT
encoder = create_box_encoder("mars-small128.pb", batch_size=32)

metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
tracker = Tracker(metric)

def is_not_empty_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Detection (Deep-SORT)")
    parser.add_argument('--video', type=int, default=0,
                        help='MOT Video Index: 0-20')
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

    if (vid.isOpened()== False): 
        print("Error opening the video file")
        exit(1)

    OUT_FILE = os.path.join(TRACKERS_FOLDER, 'YOLOX-DEEP-SORT',
                            'data', f'{args.video:04d}.txt')
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
    while(vid.isOpened()):
        # Capture frame-by-frame
        ret, frame = vid.read()

        if frame is None:
            break;

        # Labels for the current frame
        if gt_labels is not None:
            c_labels = gt_labels[gt_labels[0] == i_frame]
            c_labels = c_labels[c_labels[1] != -1]
            c_labels = c_labels[(c_labels[2] == 'Van') |
                                (c_labels[2] == 'Car')]
        else:
            c_labels = pd.DataFrame([])

        if ret == True:
            origin = frame.copy()
            height, width, _ = frame.shape

            # Draw bounding boxes onto the original image
            labels = []
            ids = []
            boxes = []
            for _, c_label in c_labels.iterrows():
                height, width, _ = frame.shape

                x1, y1, x2, y2 = c_label[6], c_label[7], c_label[8], c_label[9]

                boxes.append(np.array([x1, y1, x2, y2]))
                labels.append(c_label[2])
                ids.append(c_label[1])

            draw_bounding_boxes(origin, np.array(boxes), labels, ids)

            # Image preprocessing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run inference
            outputs, img_info = predictor.inference(image)
            if outputs[0] is None:
                continue
        
            boxes  = outputs[0][:, 0:4].cpu().numpy()
            labels = outputs[0][:, -1].cpu().numpy()
            probs = (outputs[0][:, 4] * outputs[0][:, 5]).cpu().numpy()

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Only draw 2: car, 5: bus, 7: truck
            boxes = np.array([box for box, label in zip(boxes, labels) if int(label) in [2, 5, 7]])
            probs = np.array([prob for prob, label in zip(probs, labels) if label in [2, 5, 7]])
            labels = np.array([2 for label in labels if label in [2, 5, 7]])

            if len(boxes) > 0:
                sort_boxes = boxes.copy()

                detections = []
                scale = min(640 / float(height), 640 / float(width))

                # (x1, y1, x2, y2) --> (x1, y1, w, h)
                for i, box in enumerate(sort_boxes):
                    if probs[i] < 0.5:
                        continue

                    box /= scale

                    # From x2 and y2 to width and height
                    box[2] -= box[0]
                    box[3] -= box[1]

                    # [x1, y1, w, h]
                    feature = encoder(image, box.reshape(1, -1).copy())

                    detections.append(Detection(box, probs[i], feature[0]))

                # Update tracker
                tracker.predict()
                tracker.update(detections)

                bboxes = []
                ids = []

                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    bbox = track.to_tlbr()

                    f_tracker.write(
                        f'{i_frame} {int(track.track_id)} Car -1.000000 -1 -1 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} -1 -1 -1 -1 -1 -1 -1 -1 1 \n')
                    f_tracker.flush()

                    bboxes.append(bbox)
                    ids.append(track.track_id)

                # Draw bounding boxes onto the predicted image
                labels = ['Car'] * len(bboxes)
                draw_bounding_boxes(frame, np.array(bboxes), labels, ids)

            i_frame = i_frame + 1

            if SHOW_IMAGE:
                # Display the resulting frame
                cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("Frame", cv2.WND_PROP_FULLSCREEN , cv2.WINDOW_FULLSCREEN)

                if args.dataset == "kitti":
                    cv2.imshow('Frame', draw_gt_pred_image(origin, frame, orientation="vertical"))
                else:
                    cv2.imshow('Frame', draw_gt_pred_image(origin, frame, orientation="horizontal"))

                # Press Q on keyboard to  exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else: 
            break

    f_tracker.close()
    vid.release()
    cv2.destroyAllWindows()
