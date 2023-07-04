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
model.add(Dense(32, input_dim= 2))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
### MLT 모델


# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_data, y_data, epochs=100, batch_size=32)

# 4. 평가, 예측
loss, acc = model.evaluate(x_data, y_data)

print('loss : ', loss)
print('acc : ', acc)
'''
loss :  0.21140369772911072
acc :  1.0
'''

y_predict = model.predict(x_data)
print(x_data,'의 예측결과 : ', y_predict)

'''
[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 :  [[0.11107316]
                                                [0.7775311 ]
                                                [0.7909515 ]
                                                [0.21472721]]

'''