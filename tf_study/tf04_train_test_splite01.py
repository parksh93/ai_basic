import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

print(x.shape)  # (20,)
print(y.shape)  # (20,)

# 훈련(70%)
x_train = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
y_train = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])

# 테스트(30%)
x_test = np.array([15,16,17,18,19,20])
y_test = np.array([15,16,17,18,19,20])

# 2. 모델 구성
model = Sequential()
# train이 14이므로 14이하로 인풋해줘야 한다
model.add(Dense(1,input_dim=1))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss="mse", optimizer='adam')
model.fit(x_train,y_train, epochs=100, batch_size=32)

# 4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

result = model.predict([21])
print('result : ', result)