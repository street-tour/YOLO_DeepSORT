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
import json
import matplotlib.pyplot as plt

# LED 연동 준비 (라즈베리파이 환경 고려)
try:
    import Jetson.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    print("Jetson 환경이 아니므로 LED 제어 비활성화")
    GPIO_AVAILABLE = False

LED_PIN = 17 # Jetson Orin NX GPIO 핀 번호 (실제 핀맵에 맞게 조정 필요)

if GPIO_AVAILABLE:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_PIN, GPIO.OUT)


warnings.filterwarnings("ignore", category=UserWarning)


# 절대 경로 기반 로그 파일 설정
ALARM_LOG_PATH = os.path.join(os.path.dirname(__file__), "alarm_log.csv")
# 로그 파일 초기화 (헤더 작성)
with open(ALARM_LOG_PATH, "w") as f:
    f.write("time,type,distance,person_x,person_y,forklift_x,forklift_y\n")

# 위험 위치 맵핑용 로그 저장
RISK_ZONE_LOG = os.path.join(os.path.dirname(__file__), "risk_zone_log.csv")
if not os.path.exists(RISK_ZONE_LOG):
    with open(RISK_ZONE_LOG, "w") as f:
        f.write("time,type,x,y\n")

# 모델 불러오기
yolo_model = YOLO("../yolo/best.pt")
lstm_model = load_model("../data/lstm_direction_model.h5", compile=False)
scaler = joblib.load("../data/scaler.pkl")
tracker = DeepSort()


# 설정

FEATURE_DIM = 6
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

# 임계값 자동 저장 경로 및 초기화
THRESHOLD_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'recommended_thresholds.json')
DISTANCE_THRESHOLD_REALTIME = 150.0
DISTANCE_THRESHOLD_PREDICTED = 200.0
if os.path.exists(THRESHOLD_SAVE_PATH):
    with open(THRESHOLD_SAVE_PATH, 'r') as f:
        thresholds = json.load(f)
        DISTANCE_THRESHOLD_REALTIME = thresholds.get('DISTANCE_THRESHOLD_REALTIME', DISTANCE_THRESHOLD_REALTIME)
        DISTANCE_THRESHOLD_PREDICTED = thresholds.get('DISTANCE_THRESHOLD_PREDICTED', DISTANCE_THRESHOLD_PREDICTED)

def save_thresholds(realtime, predicted):
    with open(THRESHOLD_SAVE_PATH, 'w') as f:
        json.dump({
            'DISTANCE_THRESHOLD_REALTIME': realtime,
            'DISTANCE_THRESHOLD_PREDICTED': predicted
        }, f)

# 위험구역 위치 저장 함수
def log_risk_zone(event_type, x, y):
    with open(RISK_ZONE_LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}, {event_type},{x:.2f},{y:.2f}\n")

# 위험구역 시각화 함수 (실행 후 수동 호출)
def show_risk_zone_map():
    import pandas as pd

    if not os.path.exists(RISK_ZONE_LOG):
        print("[INFO] 위험 위치 로그 파일이 없습니다.")
        return
    
    df = pd.read_csv(RISK_ZONE_LOG, skipinitialspace=True)
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)

    # 좌표별 발생 횟수 집계
    df_count = df.groupby(['type', 'x', 'y']).size().reset_index(name='count')

    plt.figure(figsize=(10, 8))
    for label, color in zip(['realtime', 'predict'], ['red', 'orange']):
        subset = df_count[df_count['type'] == label]
        # subset = df[df['type'] == label].drop_duplicates(subset=['x', 'y'])
        if not subset.empty:
            # plt.scatter(subset['x'], subset['y'], c=color, label=label, alpha=0.5, s=30)
            plt.scatter(
                subset['x'], subset['y'], s=subset['count'] * 5, # 발생 횟수에 따라 점 크기 확대
                c = color, label = label, alpha = 0.5, edgecolors='black', linewidths=0.5
            )

    plt.title("risk location frequency mapping")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis() # 영상 좌표 기준
    plt.tight_layout()
    plt.savefig("risk_zone_freq_map.png")
    plt.show()

APPLY_THRESHOLD_FRAME = 200
frame_count = 0

DISTANCE_REALTIME_SAMPLES = []
DISTANCE_PREDICTED_SAMPLES = []

# 그래프용 버퍼
DISTANCE_GRAPH_BUFFER = deque(maxlen=100)


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

def led_on(duration=1.0):
    if GPIO_AVAILABLE:
        GPIO.output(LED_PIN, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(LED_PIN, GPIO.LOW)
    else:
        print('[가상 LED] LED ON (Jetson 외 환경)')


# ----------메인 루프----------
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

        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        prev = track_buffers[track_id][-1] if len(track_buffers[track_id]) > 0 else None
        if prev is not None:
            dx, dy = cx - prev[0], cy - prev[1]
            speed = np.hypot(dx, dy)
            angle = np.arctan2(dy, dx)
        else:
            speed = 0.0
            angle = 0.0

        track_buffers[track_id].append([cx, cy, speed, angle, w, h])
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
            DISTANCE_GRAPH_BUFFER.append(distance)
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
                    log_risk_zone("realtime", px, py)
                    alarm_display_end_time = current_time + ALARM_DISPLAY_DURATION

                    with open(ALARM_LOG_PATH, "a") as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},realtime,{distance:.2f},{px:.2f},{py:.2f},{forklift_pos[0]:.2f},{forklift_pos[1]:.2f}\n")

    #### 예측 위험 감지
    for person_bbox in external_persons:
        px, py = calc_center(person_bbox)
        person_input = np.array([px, py, 0, 0, 0, 0]).reshape(1, -1)
        person_scaled = scaler.transform(person_input)[:, :2]

        for forklift_id, buffer in track_buffers.items():
            if classes.get(forklift_id) != CLASS_FORKLIFT or len(buffer) < sequence_length:
                continue
            forklift_input = np.array(buffer).reshape(1, sequence_length, FEATURE_DIM)
            forklift_pred_scaled = lstm_model.predict(forklift_input)[0]
            forklift_pred_full = np.concatenate([forklift_pred_scaled, [0, 0, 0, 0]])
            forklift_pred_real = scaler.inverse_transform(forklift_pred_full.reshape(1, -1))[0][:2]

            distance = calc_distance(person_scaled[0], forklift_pred_real)
            DISTANCE_PREDICTED_SAMPLES.append(distance)
            DISTANCE_GRAPH_BUFFER.append(distance)
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
                    log_risk_zone("predict", px, py)
                    alarm_display_end_time = current_time + ALARM_DISPLAY_DURATION

                    with open(ALARM_LOG_PATH, "a") as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},predict,{distance:.2f},{px:.2f},{py:.2f},{forklift_pos[0]:.2f},{forklift_pos[1]:.2f}\n")

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
        save_thresholds(DISTANCE_THRESHOLD_REALTIME, DISTANCE_THRESHOLD_PREDICTED)
        
    #### 알람 텍스트 표시
    if show_alarm and current_time < alarm_display_end_time:
        cv2.putText(frame, "WARNING!", (int(frame.shape[1]/2) - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    elif current_time >= alarm_display_end_time:
        show_alarm = False

    print(f"[디버그] 실시간 카운트: {realtime_risk_counter}, 예측 카운트: {predict_risk_counter}")

    #### 알람 횟수 표시
    cv2.putText(frame, f'Realtime Alarms: {realtime_risk_counter}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.putText(frame, f'Predict Alarms: {predict_risk_counter}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    #### 위험 거리 그래프 표시
    graph_height = 100
    graph_width = 300
    graph_x = 10
    graph_y = frame.shape[0] - graph_height - 10
    cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height), (50, 50, 50), -1)

    if len(DISTANCE_GRAPH_BUFFER) >= 2:
        max_distance = max(max(DISTANCE_GRAPH_BUFFER), 300)
        for i in range(1, len(DISTANCE_GRAPH_BUFFER)):
            x1 = graph_x + int((i - 1) / len(DISTANCE_GRAPH_BUFFER) * graph_width)
            y1 = graph_y + graph_height - int(DISTANCE_GRAPH_BUFFER[i - 1] / max_distance * graph_height)
            x2 = graph_x + int(i / len(DISTANCE_GRAPH_BUFFER) * graph_width)
            y2 = graph_y + graph_height - int(DISTANCE_GRAPH_BUFFER[i] / max_distance * graph_height)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.line(frame, (graph_x, graph_y + graph_height - int(DISTANCE_THRESHOLD_REALTIME / max_distance * graph_height)),
                 (graph_x + graph_width, graph_y + graph_height - int(DISTANCE_THRESHOLD_REALTIME / max_distance * graph_height)),
                 (0, 0, 255), 1)
        
    cv2.putText(frame, "Distance Graph", (graph_x, graph_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    #### 화면 출력
    cv2.imshow("실시간 위험 감지 시스템", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()

if __name__ == '__main__':
    show_risk_zone_map()

print(f"[최종 실시간 거리 임계값] {DISTANCE_THRESHOLD_REALTIME:.1f}px")
print(f"[최종 예측 거리 임계값] {DISTANCE_THRESHOLD_PREDICTED:.1f}px")

print(f"실시간 위험 감지: {realtime_risk_counter}회")
print(f"예측 위험 감지: {predict_risk_counter}회")

"""
TODO: 
탑승자는 위험감지 대상에서 제외

알람 발생 시 LED 또는 하드웨어 연동 (진행 중)
위험 거리 실시간 그래프 표시 (진행 중)
알람 발생 이력 로깅 (진행 중)

"""