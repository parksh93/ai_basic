import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, minmax_scale
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import datetime

# 1. 데이터
path = './data/boston/'
dataset = pd.read_csv(path+'Boston_house.csv')
print(dataset.shape) # (506, 14)
print(dataset.columns)
'''
Index(['AGE', 'B', 'RM', 'CRIM', 'DIS', 'INDUS', 'LSTAT', 'NOX', 'PTRATIO',
       'RAD', 'ZN', 'TAX', 'CHAS', 'Target'],
      dtype='object')
'''
                            # 한줄 전부를 삭제
x = dataset.drop(['Target'], axis=1)
y = dataset.Target

print(x.shape) #(506, 13)
print(y.shape) #(506, )

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size= 0.6,
    random_state=72,
    shuffle=True
)

print(x_train.shape, x_test.shape) # (303, 13) (203, 13)
print(y_train.shape, y_test.shape) # (303,) (203,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=13))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일, 훈련
earlyStopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    verbose=1,
    patience=50,
    restore_best_weights=True
)
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath = './mcp/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    verbose=1,
    filepath="".join([filepath,'tf20_boston',date,'_',filename])
)

model.compile(loss='mse', optimizer='adam')
model.fit(
    x_train, y_train,
    epochs=5000,
    validation_split=0.2,
    callbacks = [earlyStopping, mcp],
    verbose=1,
    batch_size=32
)

#  4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print(r2)
