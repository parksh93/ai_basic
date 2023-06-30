from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import time
import datetime

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    width_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.5,
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(
    './data/rps/',
    target_size=(150,150),
    batch_size=2016,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
    subset='training'
)

print(xy_train[0][0].shape) # (2016, 150, 150, 3)
print(xy_train[0][1].shape) # (2016, 4)

xy_test = train_datagen.flow_from_directory(
    './data/rps/',
    target_size=(150,150),
    batch_size=504,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
    subset='validation'
)

print(xy_test[0][0].shape) # (128, 150, 150, 3)
print(xy_test[0][1].shape) # (128, 4)

model = Sequential()
model.add(Conv2D(64, (4, 4), input_shape=(150, 150, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))
model.add(Conv2D(31, (2, 2), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
                    #rps 데이터는 이미 원핫 인코딩이 되어있다
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

earlyStopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=20,
    restore_best_weights=True,
    verbose=1
)
filepath = './_mcp/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    verbose=1,
    save_best_only=True,
    filepath="".join([filepath, 'rps', date, '_',filename])
)
start_time = time.time()
model.fit(xy_train[0][0], xy_train[0][1], validation_split=0.2, epochs=30, batch_size=128, verbose=1, callbacks=[earlyStopping, mcp])
end_time = time.time() - start_time

loss, acc = model.evaluate(xy_test[0][0], xy_test[0][1])

print('loss : ', loss)
print('acc : ', acc)