# 이진분류
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score, accuracy_score


# 1. data
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape)  #(569, 30)
print(y.shape)  #(569,)
print(datasets.feature_names)
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.6,test_size=0.2, random_state=1234,shuffle=True)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

#  2. model
model = Sequential()
model.add(Dense(100, input_dim=30))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(1, activation='sigmoid'))   # 2진분류는 마지막 outlayer에 sigmoid를 넣어줘야한다.

# 3. compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse','accuracy'])
model.fit(x_train,y_train,epochs=100,batch_size=128, validation_split=0.2)

# 4. 평가, 예측
# metrics가 있기 때문에 3가지 항목이 나온다 그렇기 때문에 3개로 받아야 한다
loss, mse, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('mse : ', mse)
print('accuracy : ', accuracy)
'''
loss :  0.8626043200492859
mse :  0.1649285852909088
accuracy :  0.7953216433525085
'''
y_predict = model.predict(x_test)
print(y_predict)
y_predict = np.where(y_predict > 0.5, 1, 0)
# y_predict = np.round(y_predict)
print(y_predict)
accuracy = accuracy_score(y_test, y_predict)
print('acc',accuracy)





