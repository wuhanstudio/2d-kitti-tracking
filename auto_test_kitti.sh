for VIDEO in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
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
done

python eval_kitti.py
