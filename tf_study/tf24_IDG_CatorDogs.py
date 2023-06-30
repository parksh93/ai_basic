import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import DirectoryIterator
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import datetime

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,         # 스케일 조정
    horizontal_flip=True,   # 수평으로 반전
    vertical_flip=True,     # 수직으로 반전
    width_shift_range=0.1,  # 수평 이동 범위
    height_shift_range=0.1, # 수직 이동 범위
    rotation_range=5,       # 회전 범위
    zoom_range=1.2,         # 확대 범위
    shear_range=0.5,        # 기울이기 범위
    fill_mode='nearest'     # 채우기 범위    
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(
    './data/cat_dog/training_set/',
    target_size=(150, 150),
    batch_size=8005,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

print(xy_train)
# print(xy_train[0][0].shape) # x_train (8005, 150, 150, 3)
# print(xy_train[0][1].shape) # y_train (8005,)

xy_test = test_datagen.flow_from_directory(
    './data/cat_dog/test_set/',
    target_size=(150, 150),
    batch_size=2023,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

print(xy_test)
print(xy_test[0][0].shape)  # x_test (2023, 150, 150, 3)
print(xy_test[0][1].shape)  # y_test (2023,)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(150,150,3), padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(xy_train[0][0], xy_train[0][1], epochs=30, batch_size=128, validation_split=0.2, verbose=1)

# 4. 평가, 예측
loss, acc = model.evaluate(xy_test[0][0], xy_test[0][1])
print('loss : ', loss)
print('acc : ', acc)
'''
loss :  3.3993377685546875
acc :  0.6188119053840637
'''