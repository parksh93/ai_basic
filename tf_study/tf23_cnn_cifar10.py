from keras.models import Sequential
from keras.datasets import cifar10
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time

# 1.데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

# plt.imshow(x_train[10])
# plt.show()

# scaling(이미지 0 ~ 255 -> 0 ~ 1 범위로 만들어줌)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255
x_test = x_test / 255

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu',input_shape=(32, 32 , 3), padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(Conv2D(62,(4,4),padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# 3.컴파일, 훈련
earlyStopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    verbose=1,
    patience=5,
    restore_best_weights=True
)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start_time = time.time()
model.fit(x_train, y_train, epochs=100, callbacks=[earlyStopping], verbose=1, validation_split=0.2, batch_size=14)
end_time = time.time() - start_time

# 4. 평가 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('acc : ', acc)
print('time : ', end_time)
'''
loss :  1.078698992729187
acc :  0.6297000050544739

scaling 후
loss :  0.8888710141181946
acc :  0.6948000192642212
time :  347.6795744895935   

loss :  0.8476828336715698
acc :  0.7164000272750854
time :  1373.0056462287903
'''