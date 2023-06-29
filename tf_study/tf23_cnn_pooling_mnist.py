from keras.datasets import mnist
import numpy as np
from keras.layers import Dense, Conv2D, \
    Flatten, MaxPooling2D
from keras.models import Sequential
import time
from keras.callbacks import EarlyStopping

# 1. 데이터
(x_train, y_train),(x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

print(x_train[0])
print(y_train[0])

# 시각화
# import matplotlib.pyplot as plt
# plt.imshow(x_train[0], 'gray')
# plt.show()

#  reshape
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape)
print(x_test.shape)

#  2. 모델구성
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4,4), padding='same',input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(16,(2, 2), activation='relu', padding='valid'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
earlyStopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    verbose=1,
    patience=5,
    restore_best_weights=True
)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time = time.time()

model.fit(x_train, y_train, validation_split=0.2, callbacks=[earlyStopping], epochs=100, batch_size=32, verbose=1)

end_time = time.time() - start_time

# 4. 평가 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)
print('time : ', end_time)
'''
loss :  0.08847439289093018
acc :  0.9758999943733215
time :  123.74410676956177
'''