from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x_data = [[0,0],[0,1],[1,0], [1,1]]
y_data = [0, 1, 1, 0]

# 2. 모델
# model = Perceptron()
model = Sequential()
model.add(Dense(1, input_dim= 2, activation='sigmoid'))
### sklean의 Perceptron 모델과 동일

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_data, y_data, epochs=100, batch_size=32)

# 4. 평가, 예측
loss, acc = model.evaluate(x_data, y_data)

print('loss : ', loss)
print('acc : ', acc)
'''
loss :  0.7478232979774475
acc :  0.25
'''

y_predict = model.predict(x_data)
print(x_data,'의 예측결과 : ', y_predict)

'''
[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 :  [[0.52372456]
                                                [0.31026813]
                                                [0.45676193]
                                                [0.25593194]]

'''