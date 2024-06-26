# 2D Object Tracking

You can find the dataset here (KITTI and CARLA): https://github.com/wuhanstudio/2d-kitti-tracking/releases

![](demo.png)

## Quick Start

    VIDEO=0 # 0-20
      
    # Ground Truth
    python 2d-tracking-gt.py --video ${VIDEO} --dataset kitti
    
    # Ground Truth as Detector
    python 2d-tracking-gt-sort.py --video ${VIDEO} --dataset kitti
    python 2d-tracking-gt-deep-sort.py --video ${VIDEO} --dataset kitti
    python 2d-tracking-gt-oc-sort.py --video ${VIDEO} --dataset kitti
    python 2d-tracking-gt-strong-sort.py --video ${VIDEO} --dataset kitti
    
    # Yolov3 as Detector
    python 2d-tracking-yolov3-sort.py --video ${VIDEO} --dataset kitti
    python 2d-tracking-yolov3-deep-sort.py --video ${VIDEO} --dataset kitti
    python 2d-tracking-yolov3-oc-sort.py --video ${VIDEO} --dataset kitti
    python 2d-tracking-yolov3-strong-sort.py --video ${VIDEO} --dataset kitti
    
    # Yolov4 as Detector
    python 2d-tracking-yolov4-sort.py --video ${VIDEO} --dataset kitti
    python 2d-tracking-yolov4-deep-sort.py --video ${VIDEO} --dataset kitti
    python 2d-tracking-yolov4-oc-sort.py --video ${VIDEO} --dataset kitti
    python 2d-tracking-yolov4-strong-sort.py --video ${VIDEO} --dataset kitti
    
    # YoloX as Detector
    python 2d-tracking-yolox-sort.py --video ${VIDEO} --dataset kitti
    python 2d-tracking-yolox-deep-sort.py --video ${VIDEO} --dataset kitti
    python 2d-tracking-yolox-oc-sort.py --video ${VIDEO} --dataset kitti
    python 2d-tracking-yolox-strong-sort.py --video ${VIDEO} --dataset kitti

## Evaluation Results

| HOTA                       | KITTI  | CARLA  |
| :------------------------- | :----: | :----: |
| Ground Truth               | 100.00 | 100.00 |
| Ground Truth (OC SORT)     | 93.84  | 92.93  |
| Ground Truth (Strong SORT) | 93.72  | 91.17  |
| Ground Truth (Deep SORT)   | 85.55  | 82.41  |
| Ground Truth (SORT)        | 84.62  | 82.11  |
| **3D Lidar**               |        |        |
| PermaTrack (OC-SORT)       | 76.50  |   -    |
| **Stereo Camera**          |        |        |
| CWIT                       | 66.31  |   -    |
| **Single Camera**          |        |        |
| YOLOX (OC SORT)            | 53.78  | 60.40  |
| YOLOX (Strong SORT)        | 54.30  | 59.60  |
| YOLOX (Deep SORT)          | 52.82  | 56.97  |
| YOLOX (SORT)               | 51.32  | 57.01  |
| &nbsp;                     |        |        |
| YOLOv4 (OC SORT)           | 53.38  | 57.99  |
| YOLOv4 (Strong SORT)       | 55.17  | 59.49  |
| YOLOv4 (Deep SORT)         | 51.99  | 56.65  |
| YOLOv4 (SORT)              | 50.63  | 56.31  |
| &nbsp;                     |        |        |
| YOLOv3 (OC SORT)           | 40.60  | 55.90  |
| YOLOv3 (Strong SORT)       | 43.25  | 56.19  |
| YOLOv3 (Deep SORT)         | 41.69  | 53.80  |
| YOLOv3 (SORT)              | 39.27  | 53.45  |

