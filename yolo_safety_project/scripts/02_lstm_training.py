import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping


# 데이터 로드
df = pd.read_csv("../data/object_tracks.csv")

# 이동량, 속도, 각도 계산
df = df.sort_values(by=['track_id', 'timestamp'])
df['delta_cx'] = df.groupby('track_id')['cx'].diff()
df['delta_cy'] = df.groupby('track_id')['cy'].diff()
df = df.dropna(subset=['delta_cx', 'delta_cy'])
df['speed'] = np.sqrt(df['delta_cx']**2 + df['delta_cy']**2)
df['angle'] = np.arctan2(df['delta_cy'], df['delta_cx'])

# 움직임 적은 track 제거
total_movement = df.groupby('track_id')['speed'].sum()
valid_tracks = total_movement[total_movement > 20].index
df_filtered = df[df['track_id'].isin(valid_tracks)]

# 정규화
features = ['cx', 'cy', 'speed', 'angle', 'w', 'h']
scaler = MinMaxScaler()
df_filtered[features] = scaler.fit_transform(df_filtered[features])

# scaler 저장
joblib.dump(scaler, "../data/scaler.pkl")

# 시퀀스 생성 함수
def create_sequences(df, sequence_length=10):
    sequences, targets = [], []
    for track_id, group in df.groupby("track_id"):
        values = group[features].values
        if len(values) < sequence_length + 1:
            continue
        for i in range(len(values) - sequence_length):
            seq = values[i:i+sequence_length]
            target = values[i+sequence_length][:2]
            sequences.append(seq)
            targets.append(target)
    return np.array(sequences), np.array(targets)

# 시퀀스 생성
sequence_length = 10
X, y = create_sequences(df_filtered, sequence_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM 모델 정의
def build_lstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_lstm_model(X_train.shape[1:])

# EarlyStopping 적용
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

# 모델 학습
history = model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# 모델 저장
model.save("../data/lstm_direction_model.h5")
print("모델 저장 완료")

# 평가 (MSE, MAE, R2)
model = tf.keras.models.load_model("../data/lstm_direction_model.h5", compile=False)
y_pred = model.predict(X_test)

# 정규화 복원 (inverse_transform)
y_test_full = np.concatenate([y_test, np.zeros((y_test.shape[0], 4))], axis=1)
y_pred_full = np.concatenate([y_pred, np.zeros((y_pred.shape[0], 4))], axis=1)

y_test_inv = scaler.inverse_transform(y_test_full)[:, :2]
y_pred_inv = scaler.inverse_transform(y_pred_full)[:, :2]

mse = np.mean((y_test_inv - y_pred_inv) ** 2)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")