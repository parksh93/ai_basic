# 1. 데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=1))   # 입력층(input layer)
model.add(Dense(5))     # hidden layer 1(node)
model.add(Dense(7))     # hidden layer 2
model.add(Dense(8))     # hidden layer 3
model.add(Dense(3))     # hidden layer 4
model.add(Dense(1))     # 출력층

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') # mse : 평균 제곱 오차((4-1)^2 + (4-2)^2 + (4-3)^2 / 3) / mae : 평균 절대값 오차 <- 음수값이 나올경우 효율이 더 좋다
model.fit(x, y, epochs=100)

# 4. 평가, 예측
loss = model.evaluate(x,y) # 0.0007333324174396694
print("loss : ",loss)

result = model.predict([4]) # [[3.9402738]]
print('result : ',result)