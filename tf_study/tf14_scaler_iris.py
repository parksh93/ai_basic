# 다중분류
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

print(x.shape)  #(150, 4)
print(y.shape)  #(150,)
print(datasets.feature_names) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'] <- 특성
print(datasets.DESCR)

# 원핫인코딩(one-hot encoding)
'''
 - class:
        - Iris-Setosa
        - Iris-Versicolour
        - Iris-Virginica
'''
from keras.utils import to_categorical
y = to_categorical(y)
print(y.shape) # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=72, shuffle=True)

print(x_train.shape, x_test.shape) # (105, 4) (45, 4) -> input : 4
print(y_train.shape, y_test.shape) # (105, 3) (45, 3) -> output : 3

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#  2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=4))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(3, activation='softmax'))

#  3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['mse','accuracy'])    #회기분석은 mse와 r2 score / 분류분석은 mse, accuracy score
start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=32)
end_time = time.time() - start_time
print('걸린 시간 : ',end_time)

# 4. 평가, 예측
loss, mse, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mse : ', mse)
print('accuracy : ', acc)
'''
loss :  0.012086344882845879
mse :  0.0008110394701361656
accuracy :  1.0

scaler 적용
loss :  0.08522330224514008
mse :  0.02134021744132042
accuracy :  0.9333333373069763
'''

