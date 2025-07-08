import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tensorflow.keras.models import load_model
import joblib
from collections import defaultdict, deque

yolo_model = YOLO("../yolo/best.pt")
lstm_model = load_model("../data/lstm_direction_model.h5", compile=False)
scaler = joblib.load("../data/scaler.pkl")
tracker = DeepSort()

sequence_length = 10
track_buffers = defaultdict(lambda: deque(maxlen=sequence_length))
SPEED_THRESHOLD = 0.2
COOLDOWN_TIME = 2.0

DISTANCE_THRESHOLD_REALTIME = 150.0
DISTANCE_THRESHOLD_PREDICTED = 200.0
PERSIST_TIME = 3.0  # 예측 위험 이후 실시간 위험 감시 기간

realtime_risk_counter = 0
predict_risk_counter = 0
predicted_risk_pairs = {}  # (forklift_id, person_pos): time

cap = cv2.VideoCapture("../data/forklift_test.mp4")

def calc_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2

def calc_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    current_time = time.time()

    results = yolo_model(frame)[0]
    detections, person_bboxes, forklift_bboxes = [], [], []

    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        class_id = int(cls)
        detections.append(([x1, y1, x2, y2], conf, class_id))
        if class_id == 0:
            person_bboxes.append([x1, y1, x2, y2])
        elif class_id == 1:
            forklift_bboxes.append([x1, y1, x2, y2])

    tracks = tracker.update_tracks(detections, frame=frame)

    positions, classes = {}, {}
    for track in tracks:
        if not track.is_confirmed(): continue
        track_id, det_class = track.track_id, track.det_class
        bbox = track.to_ltrb()
        cx, cy = calc_center(bbox)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        prev = track_buffers[track_id][-1] if track_buffers[track_id] else None
        speed = np.linalg.norm([cx - prev[0], cy - prev[1]]) if prev else 0.0
        angle = np.arctan2(cy - prev[1], cx - prev[0]) if prev else 0.0

        track_buffers[track_id].append([cx, cy, speed, angle, w, h])
        positions[track_id] = [cx, cy]
        classes[track_id] = det_class

    moving_forklifts = [(tid, positions[tid]) for tid, buf in track_buffers.items() if classes.get(tid) == 1 and len(buf) == sequence_length]

    for person_bbox in person_bboxes:
        px, py = calc_center(person_bbox)
        person_pos = np.array([px, py])
        person_scaled = scaler.transform([[px, py, 0, 0, 0, 0]])[:, :2][0]

        for fid, forklift_pos in moving_forklifts:
            buf = track_buffers[fid]
            forklift_input = np.array(buf).reshape(1, sequence_length, 6)
            pred_scaled = lstm_model.predict(forklift_input)[0]
            pred_full = np.concatenate([pred_scaled, [0, 0, 0, 0]])
            pred_real = scaler.inverse_transform(pred_full.reshape(1, -1))[0][:2]

            if calc_distance(person_scaled, pred_real) < DISTANCE_THRESHOLD_PREDICTED:
                predicted_risk_pairs[(fid, tuple(person_scaled))] = current_time
                predict_risk_counter += 1

    to_remove = []
    for (fid, person_pos), t in predicted_risk_pairs.items():
        if current_time - t > PERSIST_TIME:
            to_remove.append((fid, person_pos))
        else:
            for person_bbox in person_bboxes:
                px, py = calc_center(person_bbox)
                if calc_distance([px, py], person_pos) < 10:
                    if fid in positions:
                        forklift_pos = positions[fid]
                        if calc_distance([px, py], forklift_pos) < DISTANCE_THRESHOLD_REALTIME:
                            realtime_risk_counter += 1
                            cv2.line(frame, (int(px), int(py)), (int(forklift_pos[0]), int(forklift_pos[1])), (0, 0, 255), 2)

    for k in to_remove:
        predicted_risk_pairs.pop(k)

    for track in tracks:
        if not track.is_confirmed(): continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        color = (0, 255, 0) if track.det_class == 0 else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.putText(frame, f'Realtime: {realtime_risk_counter}  Predict: {predict_risk_counter}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Risk Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
