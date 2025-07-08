import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, deque
from tensorflow.keras.models import load_model
import joblib
import time

# 모델 불러오기
yolo_model = YOLO("../yolo/best.pt")
lstm_model = load_model("../data/lstm_direction_model.h5", compile=False)
scaler = joblib.load("../data/scaler.pkl")
tracker = DeepSort()

# 설정
sequence_length = 10
track_buffers = defaultdict(lambda: deque(maxlen=sequence_length))
# DISTANCE_THRESHOLD_REALTIME = 230
# DISTANCE_THRESHOLD_PREDICTED = 850
SPEED_THRESHOLD = 0.2
COOLDOWN_TIME = 2.0

CLASS_PERSON = 0
CLASS_FORKLIFT = 1

realtime_risk_counter = 0
predict_risk_counter = 0
last_warning_time_realtime = {}
last_warning_time_predict = {}

def calc_adjusted_center(bbox, adjustment_ratio=0.8):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = y1 + (y2 - y1) * adjustment_ratio
    return cx, cy

def shrink_bbox_width(bbox, ratio=0.7):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    shrink_w = w * ratio
    center_x = (x1 + x2) / 2
    new_x1 = center_x - shrink_w / 2
    new_x2 = center_x + shrink_w / 2
    return [new_x1, y1, new_x2, y2]

def calc_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1+x2)/2, (y1+y2)/2

def calc_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# def is_inside(person_bbox, forklift_bbox, tolerance=0.05):
#     px1, py1, px2, py2 = person_bbox
#     fx1, fy1, fx2, fy2 = forklift_bbox
#     fx1 -= tolerance * (fx2 - fx1)
#     fx2 += tolerance * (fx2 - fx1)
#     fy1 -= tolerance * (fy2 - fy1)
#     fy2 += tolerance * (fy2 - fy1)
#     return (px1 > fx1 and px2 < fx2 and py1 > fy1 and py2 < fy2)

def is_moving(buffer, threshold=SPEED_THRESHOLD):
    speeds = []
    buffer_list = list(buffer)
    for i in range(1, len(buffer_list)):
        prev = np.array(buffer_list[i-1])
        curr = np.array(buffer_list[i])
        dist = np.linalg.norm(curr - prev)
        speeds.append(dist)
    if not speeds:
        return False
    avg_speed = np.mean(speeds)
    return avg_speed > threshold

cap = cv2.VideoCapture("../data/forklift_2.mp4")

DISTANCE_REALTIME_SAMPLES = []
DISTANCE_PREDICTED_SAMPLES = []

# 초기값, 추천값으로 덮어씀
DISTANCE_THRESHOLD_REALTIME = 150
DISTANCE_THRESHOLD_PREDICTED = 200

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = yolo_model(frame)[0]

    detections, person_bboxes, forklift_bboxes = [], [], []
    class_ids = []

    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        class_id = int(cls)
        class_ids.append(class_id)
        detections.append(([x1, y1, x2, y2], conf, class_id))
        if class_id == CLASS_PERSON:
            person_bboxes.append([x1, y1, x2, y2])
        elif class_id == CLASS_FORKLIFT:
            forklift_bboxes.append([x1, y1, x2, y2])

    print(f"감지된 클래스 ID 목록: {class_ids}")
    print(f"감지된 사람 수: {len(person_bboxes)} / 감지된 지게차 수: {len(forklift_bboxes)}")

    tracks = tracker.update_tracks(detections, frame=frame)

    positions, classes = {}, {}
    for track in tracks:
        if not track.is_confirmed(): continue
        track_id, det_class = track.track_id, track.det_class
        bbox = track.to_ltrb()

        if det_class == CLASS_FORKLIFT:
            adjusted_bbox = shrink_bbox_width(bbox, ratio=0.7)
            cx = (adjusted_bbox[0] + adjusted_bbox[2]) / 2 
            cy = adjusted_bbox[1] + (adjusted_bbox[3] - adjusted_bbox[1]) * 0.8
        else:
            cx, cy = calc_center(bbox)

        prev = track_buffers[track_id][-1] if len(track_buffers[track_id]) > 0 else None
        if prev is not None:
            dx, dy = cx - prev[0], cy - prev[1]
            speed = np.hypot(dx, dy)
            angle = np.arctan2(dy, dx)
        else:
            speed = 0.0
            angle = 0.0

        track_buffers[track_id].append([cx, cy, speed, angle])
        positions[track_id] = [cx, cy]
        classes[track_id] = det_class
    
    # 탑승자 제거 대신 모든 사람을 위험 감지 대상으로 포함
    external_persons = person_bboxes

    print(f"[프레임] 사람: {len(person_bboxes)} 외부: {len(external_persons)} 지게차: {len(forklift_bboxes)}")
    
    # 이동중 지게차 필터
    moving_forklifts = []
    for track_id, buffer in track_buffers.items():
        if classes.get(track_id) == CLASS_FORKLIFT and len(buffer) == sequence_length:
            if is_moving(buffer):
                avg_speed = np.mean([np.linalg.norm(np.array(buffer[i][:2]) - np.array(buffer[i-1][:2])) for i in range(1, len(buffer))])
                print(f"지게차 {track_id} 이동중 -> 평균 속도: {avg_speed:.2f}")
                moving_forklifts.append((track_id, positions[track_id]))

    print(f"[디버그] 이동중 지게차 목록: {moving_forklifts}")

    current_time = time.time()   

    #### 실시간 위험 감지
    for person_bbox in external_persons:
        px, py = calc_center(person_bbox)
        person_pos = np.array([px, py])
        for forklift_id, forklift_pos in moving_forklifts:
            forklift_pos = np.array(forklift_pos)
            distance = calc_distance(person_pos, forklift_pos)
            DISTANCE_REALTIME_SAMPLES.append(distance)
            print(f"[실시간 거리] {distance:.1f}px 기준: {DISTANCE_THRESHOLD_REALTIME:.1f}px")
            if DISTANCE_THRESHOLD_REALTIME and distance < DISTANCE_THRESHOLD_REALTIME:
                realtime_risk_counter += 1
                cv2.line(frame, (int(px), int(py)), (int(forklift_pos[0]), int(forklift_pos[1])), (0, 0, 255), 2)
            # pair_key = (forklift_id, tuple(person_pos))
            # last_time = last_warning_time_realtime.get(pair_key, 0)
            # if distance < DISTANCE_THRESHOLD_REALTIME and (current_time - last_time > COOLDOWN_TIME):
            #     print(f"실시간 위험: {distance:.1f}px")
            #     realtime_risk_counter += 1
            #     last_warning_time_realtime[pair_key] = current_time
            # if DISTANCE_THRESHOLD_REALTIME and distance < DISTANCE_THRESHOLD_REALTIME:
            #     realtime_risk_counter += 1


    #### 예측 위험 감지
    for person_bbox in external_persons:
        px, py = calc_center(person_bbox)
        person_input = np.array([px, py, 0, 0]).reshape(1, -1)
        person_scaled = scaler.transform(person_input)[:, :2]

        for forklift_id, buffer in track_buffers.items():
            if classes.get(forklift_id) != CLASS_FORKLIFT or len(buffer) < sequence_length:
                continue
            forklift_input = np.array(buffer).reshape(1, sequence_length, 4)
            forklift_pred_scaled = lstm_model.predict(forklift_input)[0]
            forklift_pred_full = np.concatenate([forklift_pred_scaled, [0, 0]])
            forklift_pred_real = scaler.inverse_transform(forklift_pred_full.reshape(1, -1))[0][:2]

            distance = calc_distance(person_scaled[0], forklift_pred_real)
            DISTANCE_PREDICTED_SAMPLES.append(distance)
            print(f"[예측 거리] {distance:.1f}px 기준: {DISTANCE_THRESHOLD_PREDICTED:.1f}px")
            if DISTANCE_THRESHOLD_PREDICTED and distance < DISTANCE_THRESHOLD_PREDICTED:
                predict_risk_counter += 1
                cv2.line(frame, (int(px), int(py)), (int(forklift_pred_real[0]), int(forklift_pred_real[1])), (0, 255, 255), 2)
            # pair_key = (forklift_id, tuple(person_scaled[0]))
            # last_time = last_warning_time_predict.get(pair_key, 0)
            # if distance < DISTANCE_THRESHOLD_PREDICTED and (current_time - last_time > COOLDOWN_TIME):
            #     print(f"예측 위험: {distance:.1f}px")
            #     predict_risk_counter += 1
            #     last_warning_time_predict[pair_key] = current_time
            # if DISTANCE_THRESHOLD_PREDICTED and distance < DISTANCE_THRESHOLD_PREDICTED:
            #     predict_risk_counter += 1

    #### 실시간 시각화
    for track in tracks:
        if not track.is_confirmed(): continue
        track_id, det_class = track.track_id, track.det_class
        bbox = track.to_ltrb()
        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 255, 0) if det_class == CLASS_PERSON else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("실시간 위험 감지 시스템", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

if DISTANCE_REALTIME_SAMPLES:
    recommended_realtime = np.percentile(DISTANCE_REALTIME_SAMPLES, 20)
    DISTANCE_THRESHOLD_REALTIME = recommended_realtime
    print(f"[추천 실시간 거리 임계값] 약 {recommended_realtime:.1f}px")

if DISTANCE_PREDICTED_SAMPLES:
    recommended_predicted = np.percentile(DISTANCE_PREDICTED_SAMPLES, 20)
    DISTANCE_THRESHOLD_PREDICTED = recommended_predicted
    print(f"[추천 예측 거리 임계값] 약 {recommended_predicted:.1f}px")

print(f"실시간 위험 감지: {realtime_risk_counter}회")
print(f"예측 위험 감지: {predict_risk_counter}회")

"""
TODO: 
실시간 위험 감지 횟수: 0회
예측 위험 감지 횟수: 0회

문제 해결하기, yolo 자체에서 탐지를 잘 하고 있지 못함
inverse_transform 적용을 하고 있지 않음, scaler를 저장

탑승자는 위험감지 대상에서 제외
이동중인 지게차와 주변 사람만 위험 감지
실시간 + 예측 위험 감지 병합
중복 억제 포함
시각화 포함
"""