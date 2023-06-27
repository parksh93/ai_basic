import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 2.1, 3.1, 4.1, 5.1, 6,7, 8, 9.3, 10.5]])

y = np.array([11,12,13,14,15,16,17,18,19,20])

print(x.shape)
print(y.shape)

x = x.transpose()  # 동일한 코드 (x = x.T)
print(x.shape)

# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer="adam")
model.fit(x, y, epochs=100, batch_size=2)

# 평가, 예측
loss = model.evaluate(x, y)
print('loss : ',loss)
# loss :  0.0018647933611646295

result = model.predict([[10, 10.5]]) # 20이 나와야한다
print('result : ', result)
# result :  [[20.101242]]