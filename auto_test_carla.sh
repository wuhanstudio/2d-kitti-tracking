# for VIDEO in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
for VIDEO in 3 4 5
do
    # python 2d-tracking-gt.py --video ${VIDEO} --dataset carla

    # python 2d-tracking-gt-sort.py --video ${VIDEO} --dataset carla
    # python 2d-tracking-gt-deep-sort.py --video ${VIDEO} --dataset carla
    python 2d-tracking-gt-oc-sort.py --video ${VIDEO} --dataset carla
    # python 2d-tracking-gt-strong-sort.py --video ${VIDEO} --dataset carla

    # python 2d-tracking-yolo-sort.py --video ${VIDEO} --dataset carla
    # python 2d-tracking-yolo-deep-sort.py --video ${VIDEO} --dataset carla
    python 2d-tracking-yolo-oc-sort.py --video ${VIDEO} --dataset carla
    # python 2d-tracking-yolo-strong-sort.py --video ${VIDEO} --dataset carla

    # python 2d-tracking-yolox-sort.py --video ${VIDEO} --dataset carla
    # python 2d-tracking-yolox-deep-sort.py --video ${VIDEO} --dataset carla
    python 2d-tracking-yolox-oc-sort.py --video ${VIDEO} --dataset carla
    # python 2d-tracking-yolox-strong-sort.py --video ${VIDEO} --dataset carla
done

python eval_carla.py
