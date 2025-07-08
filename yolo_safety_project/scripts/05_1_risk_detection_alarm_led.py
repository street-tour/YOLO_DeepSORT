import os
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
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# 상수 선언
FEATURE_DIM = 6
SEQUENCE_LENGTH = 10
SPEED_THRESHOLD = 0.2
COOLDOWN_TIME = 2.0
DISTANCE_THRESHOLD_REALTIME = 150.0
DISTANCE_THRESHOLD_PREDICTED = 200.0
ALARM_DISPLAY_DURATION = 1.0
APPLY_THRESHOLD_FRAME = 200

# 모델 불러오기
yolo_model = YOLO("../yolo/best.pt")
lstm_model = load_model("../data/lstm_direction_model.h5", compile=False)
scaler = joblib.load("../data/scaler.pkl")
tracker = DeepSort()

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

def led_on(duration=1.0):
    if GPIO_AVAILABLE:
        GPIO.output(LED_PIN, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(LED_PIN, GPIO.LOW)
    else:
        print('[가상 LED] LED ON (Jetson 외 환경)')

# 설정
track_buffers = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))
realtime_risk_counter = 0
predict_risk_counter = 0
last_warning_time_realtime = {}
last_warning_time_predict = {}
show_alarm = False
alarm_display_end_time = 0
frame_count = 0
DISTANCE_REALTIME_SAMPLES = []
DISTANCE_PREDICTED_SAMPLES = []

# 절대 경로 기반 로그 파일 설정
ALARM_LOG_PATH = os.path.join(os.path.dirname(__file__), "alarm_log.csv")
# 로그 파일 초기화 (헤더 작성)
with open(ALARM_LOG_PATH, "w") as f:
    f.write("time,type,distance,person_x,person_y,forklift_x,forklift_y\n")

cap = cv2.VideoCapture("../data/forklift_test.mp4")

# # 초기값, 추천값으로 덮어씀
# APPLY_THRESHOLD_FRAME = 200
# frame_count = 0

# DISTANCE_REALTIME_SAMPLES = []
# DISTANCE_PREDICTED_SAMPLES = []

# # 그래프용 버퍼
# DISTANCE_GRAPH_BUFFER = deque(maxlen=100)

# ----------보조 함수------------
def calc_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1+x2)/2, (y1+y2)/2

# def calc_adjusted_center(bbox, adjustment_ratio=0.8):
#     x1, y1, x2, y2 = bbox
#     cx = (x1 + x2) / 2
#     cy = y1 + (y2 - y1) * adjustment_ratio
#     return cx, cy

# def shrink_bbox_width(bbox, ratio=0.7):
#     x1, y1, x2, y2 = bbox
#     w = x2 - x1
#     shrink_w = w * ratio
#     center_x = (x1 + x2) / 2
#     new_x1 = center_x - shrink_w / 2
#     new_x2 = center_x + shrink_w / 2
#     return [new_x1, y1, new_x2, y2]

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

# def is_moving(buffer, threshold=SPEED_THRESHOLD):
#     speeds = []
#     buffer_list = list(buffer)
#     for i in range(1, len(buffer_list)):
#         prev = np.array(buffer_list[i-1])
#         curr = np.array(buffer_list[i])
#         dist = np.linalg.norm(curr - prev)
#         speeds.append(dist)
#     if not speeds:
#         return False
#     avg_speed = np.mean(speeds)
#     return avg_speed > threshold


# ----------메인 루프----------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    frame_count += 1
    current_time = time.time()
    results = yolo_model(frame)[0]
    detections, person_bboxes, forklift_bboxes = [], [], []
    # class_ids = []

    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        class_id = int(cls)
        detections.append(([x1, y1, x2, y2], conf, class_id))
        if class_id == 0:
            person_bboxes.append([x1, y1, x2, y2])
        elif class_id == 1:
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
        cx, cy = calc_center(bbox)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        prev = track_buffers[track_id][-1] if len(track_buffers[track_id]) > 0 else None
        speed = np.linalg.norm([cx - prev[0], cy - prev[1]]) if prev else 0.0
        angle = np.arctan2(cy - prev[1], cx - prev[0]) if prev else 0.0

        track_buffers[track_id].append([cx, cy, speed, angle, w, h])
        positions[track_id] = [cx, cy]
        classes[track_id] = det_class
    
    # 이동중 지게차 필터
    moving_forklifts = [(tid, positions[tid]) for tid, buf in track_buffers.items() if classes.get(tid) == 1 and len(buf) == SEQUENCE_LENGTH]
    
    # 탑승자 제거 대신 모든 사람을 위험 감지 대상으로 포함
    # external_persons = person_bboxes
    # current_time = time.time()   


    #### 실시간 위험 감지
    for person_bbox in person_bboxes:
        px, py = calc_center(person_bbox)
        person_input = np.zeros((1, FEATURE_DIM))
        person_input[0, 0] = px
        person_input[0, 1] = py
        person_scaled = scaler.transform(person_input)[0][:2]

        for fid, forklift_pos in moving_forklifts:
            buf = track_buffers[fid]
            forklift_input = np.array(buf).reshape(1, SEQUENCE_LENGTH, FEATURE_DIM)
            forklift_pred_scaled = lstm_model.predict(forklift_input)[0]

            forklift_pred_full = np.zeros((1, FEATURE_DIM))
            forklift_pred_full[0, 0] = forklift_pred_scaled[0]
            forklift_pred_full[0, 1] = forklift_pred_scaled[1]
            forklift_pred_real = scaler.inverse_transform(forklift_pred_full)[0][:2]

            distance_pred = calc_distance(person_scaled, forklift_pred_real)
            distance_real = calc_distance([px, py], forklift_pos)

            DISTANCE_REALTIME_SAMPLES.append(distance_real)
            DISTANCE_PREDICTED_SAMPLES.append(distance_pred)

            if distance_pred < DISTANCE_THRESHOLD_PREDICTED:
                predict_risk_counter += 1
                cv2.line(frame, (int(px), int(py)), (int(forklift_pred_real[0]), int(forklift_pred_real[1])), (0, 255, 255), 2)
                play_alarm()
                led_on()

            if distance_real < DISTANCE_THRESHOLD_REALTIME:
                realtime_risk_counter += 1
                cv2.line(frame, (int(px), int(py)), (int(forklift_pos[0]), int(forklift_pos[1])), (0, 0, 255), 2)
                play_alarm()
                led_on()

    if frame_count >= APPLY_THRESHOLD_FRAME:
        if DISTANCE_REALTIME_SAMPLES:
            DISTANCE_THRESHOLD_REALTIME = np.percentile(DISTANCE_REALTIME_SAMPLES, 20)            
        if DISTANCE_PREDICTED_SAMPLES:
            DISTANCE_THRESHOLD_PREDICTED = np.percentile(DISTANCE_PREDICTED_SAMPLES, 20)

    #### 실시간 시각화
    for track in tracks:
        if not track.is_confirmed(): 
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        color = (0, 255, 0) if track.det_class == 0 else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # cv2.putText(frame, f"ID:{track.track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
       
    cv2.putText(frame, f'Realtime: {realtime_risk_counter} Predict: {predict_risk_counter}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    #### 화면 출력
    cv2.imshow("실시간 위험 감지 시스템", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()

print(f'최종 실시간 감지: {realtime_risk_counter}회, 예측 감지: {predict_risk_counter}회')

"""
TODO: 
탑승자는 위험감지 대상에서 제외

알람 발생 시 LED 또는 하드웨어 연동 (진행 중)
위험 거리 실시간 그래프 표시 (진행 중)
알람 발생 이력 로깅 (진행 중)

"""