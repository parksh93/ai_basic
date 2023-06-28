# 이진분류
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


# 1. data
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape)  #(569, 30)
print(y.shape)  #(569,)
print(datasets.feature_names)
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, random_state=1234,shuffle=True)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# Scaler 적용
# scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#  2. model
model = Sequential()
model.add(Dense(100, input_dim=30))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(1, activation='sigmoid'))   # 2진분류는 마지막 outlayer에 sigmoid를 넣어줘야한다.

# 3. compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse','accuracy'])
model.fit(x_train,y_train,epochs=100,batch_size=128)

# 4. 평가, 예측
# metrics가 있기 때문에 3가지 항목이 나온다 그렇기 때문에 3개로 받아야 한다
loss, mse, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mse : ', mse)
print('accuracy : ', accuracy)
'''
1. scaler 적용 전
loss :  0.8626043200492859
mse :  0.1649285852909088
accuracy :  0.7953216433525085

2. StandardScaler 적용
loss :  0.251194030046463
mse :  0.030446138232946396
accuracy :  0.9649122953414917

3. MinMaxScaler 적용
loss :  0.2737239897251129
mse :  0.056328076869249344
accuracy :  0.9356725215911865

4. MaxAbsScaler 적용
loss :  0.16019317507743835
mse :  0.04080752283334732
accuracy :  0.9473684430122375

5. RobustScaler 적용
loss :  0.27140119671821594
mse :  0.039330486208200455
accuracy :  0.9532163739204407
'''




