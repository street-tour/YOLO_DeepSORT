import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 모델 로드
yolo_model = YOLO('../yolo/best.pt')
tracker = DeepSort()

# 데이터 저장 리스트
track_data = []

# 비디오 로드
cap = cv2.VideoCapture("../data/forklift_2.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = yolo_model(frame)[0]

    detections = []
    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        detections.append(([x1, y1, x2, y2], conf, int(cls)))

    tracks = tracker.update_tracks(detections, frame=frame)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for track in tracks:
        if not track.is_confirmed(): continue
        track_id = track.track_id
        bbox = track.to_ltrb()
        cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
        track_data.append([timestamp, track_id, cx, cy])

cap.release()

# 데이터 저장
df = pd.DataFrame(track_data, columns=["timestamp", "track_id", "cx", "cy"])
df.to_csv("../data/object_tracks.csv", index=False)
print("object_tracks.csv 저장 완료")