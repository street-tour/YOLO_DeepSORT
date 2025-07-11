- yolo_lstm_safety_project/
   처음 CCTV 영상을 가지고 작업한 작업물
- yolo_lstm_safety_project_copy/
   처음 CCTV 영상을 가지고 작업한 정리된 python files
- yolo_safety_project
   다른 CCTV 영상을 가지고 작업한 python files
사전작업 : 기존 YOLO모델에는 Forklift(지게차)의 항목이 없기 때문에 파인튜닝 진행해서 새로운 best모델을 만들어야함
        labelimg 이용 -> 영상을 프레임 단위로 분할한 뒤 라벨링 작업 수행
        YOLO 모델링 작업 이후, best.pt 모델로 라벨이 감지된 영상 추출

        YOLO best.pt 모델을 yolo 폴더 안에 넣기, 영상 데이터를 data안에 넣기

01 : 기존 yolo best 모델을 불러와서 LSTM 모델을 만들기 위한 작업
    DeepSORT로 track_id들의 이동 경로를 저장

02 : 정규화 작업 및 LSTM 모델링 작업 
    모델 학습 및 정규화 복원 작업 

03 : 실시간 위험 감지 및 예측 위험 감지를 하는 모델
    추천 임계값을 자동 적용하여, 실시간 위험과 예측 위험 횟수를 탐지 

04 : 위험이 탐지 되면 알람이 울리고, 알람 횟수를 표시
    그래프에 위험 수치 표시
    알람이 울리는 로그 저장 

05 : 위험 알람이 울리면 LED로 안내해주는 코드 
    (추론을 Jetson으로 진행했기 때문에 Window환경에선 가상 LED 사용)

06 : CCTV 스트림 연결해서 돌리는 모델코드 (현재는 CCTV 스트림 정보가 없음)

07 : 위험위치 맵핑용 로그 저장

08 : YOLO DeepSORT 진행 후 bbox가 객체의 크기보다 큰 현상 수정
    객체 인식은 문제 없음