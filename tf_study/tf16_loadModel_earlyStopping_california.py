import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
                                #변수
print(x.shape, y.shape) # (20640, 8) (20640,)

print(datasets.feature_names)
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.6,
    test_size= 0.2,
    random_state=123,
    shuffle=True
)

print(x_train.shape) #(14447, 8)
print(y_train.shape) #(14447,)
print(x_test.shape)  #(6193, 8)
print(y_test.shape)  #(6193,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델구성
# 모델 불러오기
model = load_model('./_save/tf16_california.h5')

# 3. 컴파일, 훈련
# earluStopping 
from keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping (
    monitor='val_loss',
    patience=50,
    mode='min',
    verbose=1,
    restore_best_weights=True   # Default = false
)
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, validation_split=0.2,callbacks=[earlyStopping], epochs=100, batch_size=128)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# r2 score(결정 계수)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2', r2) # 1에 가까울수록 좋다
