import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score #회기분석
from sklearn.preprocessing import StandardScaler, minmax_scale
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import datetime

# 1.데이터
path = './data/boston/'
x_train = pd.read_csv(path+'train-data.csv')
x_test = pd.read_csv(path + 'test-data.csv')
y_train = pd.read_csv(path+'train-target.csv')
y_test = pd.read_csv(path + 'test-target.csv')

print(x_train.shape, x_test.shape) # (333, 11) (173, 11)
print(y_train.shape, y_test.shape) # (333, 1) (173, 1)
print(x_train.columns)
print(y_train.columns)
print(x_train.describe())
print(x_train.info())

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(50, input_dim=11))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일 훈련
# 발리데이션이 없기때문에 안된다
# earlyStopping = EarlyStopping (
#     monitor='val_loss',
#     patience=50,
#     mode='min',
#     verbose=1,
#     restore_best_weights=True 
# )

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=128)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# r2 score(결정 계수)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 : ', r2)