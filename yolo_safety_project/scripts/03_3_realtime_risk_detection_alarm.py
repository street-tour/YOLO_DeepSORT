import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
import torch
import warnings
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict, deque
from tensorflow.keras.models import load_model
from alarm import play_alarm

import joblib
import time

warnings.filterwarnings("ignore", category=UserWarning)

# 모델 불러오기
yolo_model = YOLO("../yolo/best.pt")

lstm_model = load_model("../data/lstm_direction_model.h5", compile=False)
scaler = joblib.load("../data/scaler.pkl")
tracker = DeepSort()
"""
Sort vs DeepSort 차이
Sort(Simple Online and Realtime Tracking):

간단한 칼만필터 + 바운딩 박스 기반 IOU 매칭으로 객체 추적

속도 빠르고 가볍지만, appearance(외형) 정보 사용 안함

동일 객체가 프레임 중간에 가려졌다 다시 등장하면 ID 변동 위험 있음

DeepSort:

Sort의 한계 보완

바운딩 박스 + 딥러닝 기반 외형(appearance) feature 사용

객체가 잠깐 사라져도 같은 ID 유지 확률 높음

상대적으로 속도 느리고 복잡

현재 코드가 공장 내부 실시간 처리에 초점을 두고 있어 빠른 추적을 위해 Sort()를 쓴 것으로 보입니다. 
만약 ID 안정성이 더 중요하면 DeepSort로 교체 가능.

LSTM 모델을 사용하려면 객체 ID가 고정이 되어야하지않나?
"""
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

# 화면에 표시할 알람 상태
show_alarm = False
alarm_display_end_time = 0
ALARM_DISPLAY_DURATION = 1.0 # 화면에 알람 텍스트 표시 지속시간 (초)

# 초기값, 추천값으로 덮어씀
DISTANCE_THRESHOLD_REALTIME = 150.0
DISTANCE_THRESHOLD_PREDICTED = 200.0

APPLY_THRESHOLD_FRAME = 200
frame_count = 0

DISTANCE_REALTIME_SAMPLES = []
DISTANCE_PREDICTED_SAMPLES = []

cap = cv2.VideoCapture("../data/forklift_2.mp4")


# ----------보조 함수------------
def calc_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1+x2)/2, (y1+y2)/2

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



while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    frame_count += 1
    current_time = time.time()

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

    # print(f"감지된 클래스 ID 목록: {class_ids}")
    # print(f"감지된 사람 수: {len(person_bboxes)} / 감지된 지게차 수: {len(forklift_bboxes)}")

    tracks = tracker.update_tracks(detections, frame=frame)

    positions, classes = {}, {}
    for track in tracks:
        if not track.is_confirmed(): 
            continue
        track_id, det_class = track.track_id, track.det_class
        bbox = track.to_ltrb()

        if det_class == CLASS_FORKLIFT:
            adjusted_bbox = shrink_bbox_width(bbox, ratio=0.7)
            cx, cy = calc_adjusted_center(adjusted_bbox)
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
    
    # 이동중 지게차 필터
    moving_forklifts = []
    for track_id, buffer in track_buffers.items():
        if classes.get(track_id) == CLASS_FORKLIFT and len(buffer) == sequence_length:
            moving_forklifts.append((track_id, positions[track_id]))
    
    # 탑승자 제거 대신 모든 사람을 위험 감지 대상으로 포함
    external_persons = person_bboxes
    current_time = time.time()   


    #### 실시간 위험 감지
    for person_bbox in external_persons:
        px, py = calc_center(person_bbox)
        person_pos = np.array([px, py])
        for forklift_id, forklift_pos in moving_forklifts:
            forklift_pos = np.array(forklift_pos)
            distance = calc_distance(person_pos, forklift_pos)
            DISTANCE_REALTIME_SAMPLES.append(distance)
            # print(f"[실시간 거리] {distance:.1f}px 기준: {DISTANCE_THRESHOLD_REALTIME:.1f}px")
            if distance < DISTANCE_THRESHOLD_REALTIME:
                pair_key = (forklift_id, tuple(person_pos))
                last_time = last_warning_time_realtime.get(pair_key, 0)
                if current_time - last_time > COOLDOWN_TIME:
                    realtime_risk_counter += 1
                    last_warning_time_realtime[pair_key] = current_time
                    cv2.line(frame, (int(px), int(py)), (int(forklift_pos[0]), int(forklift_pos[1])), (0, 0, 255), 2)

                    play_alarm()
                    show_alarm = True
                    alarm_display_end_time = current_time + ALARM_DISPLAY_DURATION

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
            # print(f"[예측 거리] {distance:.1f}px 기준: {DISTANCE_THRESHOLD_PREDICTED:.1f}px")
            if distance < DISTANCE_THRESHOLD_PREDICTED:
                pair_key = (forklift_id, tuple(person_scaled[0]))
                last_time = last_warning_time_predict.get(pair_key, 0)
                if current_time - last_time > COOLDOWN_TIME:
                    predict_risk_counter += 1
                    last_warning_time_predict[pair_key] =  current_time
                    cv2.line(frame, (int(px), int(py)), (int(forklift_pred_real[0]), int(forklift_pred_real[1])), (0, 255, 255), 2)

                    play_alarm()
                    show_alarm = True
                    alarm_display_end_time = current_time + ALARM_DISPLAY_DURATION

            # 예측 위치 점 시각화 (청록색 원) - LSTM 예측 지게차 미래 위치
            cv2.circle(frame, (int(forklift_pred_real[0]), int(forklift_pred_real[1])), 5, (255, 255, 0), -1)

    #### 실시간 시각화
    for track in tracks:
        if not track.is_confirmed(): 
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        color = (0, 255, 0) if track.det_class == CLASS_PERSON else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{track.track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #### 추천 임계값 자동 적용
    if frame_count >= APPLY_THRESHOLD_FRAME:
        if DISTANCE_REALTIME_SAMPLES:
            recommended_realtime = np.percentile(DISTANCE_REALTIME_SAMPLES, 20)
            DISTANCE_THRESHOLD_REALTIME = recommended_realtime
        if DISTANCE_PREDICTED_SAMPLES:
            recommended_predicted = np.percentile(DISTANCE_PREDICTED_SAMPLES, 20)
            DISTANCE_THRESHOLD_PREDICTED = recommended_predicted

    #### 알람 텍스트 표시
    if show_alarm and current_time < alarm_display_end_time:
        cv2.putText(frame, "WARNING!", (int(frame.shape[1]/2) - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    elif current_time >= alarm_display_end_time:
        show_alarm = False

    print(f"[디버그] 실시간 카운트: {realtime_risk_counter}, 예측 카우트: {predict_risk_counter}")

    #### 알람 횟수 표시
    cv2.putText(frame, f'Realtime Alarms: {realtime_risk_counter}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.putText(frame, f'Predict Alarms: {predict_risk_counter}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    cv2.imshow("실시간 위험 감지 시스템", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()

print(f"[최종 실시간 거리 임계값] {DISTANCE_THRESHOLD_REALTIME:.1f}px")
print(f"[최종 예측 거리 임계값] {DISTANCE_THRESHOLD_PREDICTED:.1f}px")

print(f"실시간 위험 감지: {realtime_risk_counter}회")
print(f"예측 위험 감지: {predict_risk_counter}회")

"""
TODO: 
탑승자는 위험감지 대상에서 제외

실시간 + 예측 위험 감지 병합
중복 억제 포함
"""