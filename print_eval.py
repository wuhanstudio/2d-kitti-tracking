import pandas as pd

folder = "YOLOv3-SORT"
dataset = "carla"
data = pd.read_csv(f"data/trackers/{dataset}/{dataset}_2d_box_train/{folder}/car_summary.txt", sep=" ")

print("HOTA", data["HOTA"])
print("MOTA", data["MOTA"])
print("IDF1", data["IDF1"])
print("MT", data["MT"])
print("ML", data["ML"])
