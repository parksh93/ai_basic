import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

print(x.shape)  # (20,)
print(y.shape)  # (20,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, batch_size=1, epochs=100)

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predic = model.predict(x_test)

r2 = r2_score(y_test, y_predic)
print(r2)
