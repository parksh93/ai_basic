import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
                                #변수
print(x.shape, y.shape) # (20640, 8) (20640,)

print(datasets.feature_names) # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)
# DataFrame 변환
df = pd.DataFrame(x, columns=[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']])
df['Target(y)'] = y

# 시각화, 상관관계
import matplotlib.pyplot as plt
import seaborn as sns   # pip install seaborn / 안될시 : pip install -U seaborn

sns.set(font_scale = 1.2)
sns.set(rc={'figure.figsize':(9, 6)})   # 가로 세로 사이즈 세팅
sns.heatmap(
    data=df.corr(),    # corr : 상관관계
    square=True,            # 정사각형으로 view
    annot=True,             # 각 cell값 표기
    cbar=True               # color bar 표기
)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.6,
    test_size= 0.2,
    random_state=72,
    shuffle=True
)

print(x_train.shape) #(14447, 8)
print(y_train.shape) #(14447,)
print(x_test.shape)  #(6193, 8)
print(y_test.shape)  #(6193,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 2. 모델구성
model = Sequential()
                    # 변수가 들어가야됨
model.add(Dense(50, input_dim=8))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

# 3. 컴파일, 훈련
# earluStopping 
from keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping (
    monitor='val_loss',
    patience=50,
    mode='min',
    verbose=1,
    restore_best_weights=True   # Default = false
)
# ModelCheckpoint : 훈련중 중간중간 성능이 좋은 구간을 찾아 저장한다.

# 파일명 생성
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M')
filepath = './_mcp/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    # filepath='./_mcp/tf20_california.hdf5'
    filepath="".join([filepath,'tf20_california', date, '_', filename])
)

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, validation_split=0.2,callbacks=[earlyStopping, mcp], epochs=100, batch_size=32)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

# r2 score(결정 계수)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2', r2) # 1에 가까울수록 좋다

#  시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.',c='blue',label='val_loss')
plt.title('Loss & Val_Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()