import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# 데이터 불러오기
path = './data/credit_card_prediction/'
data = pd.read_csv(path + 'train.csv')

# 데이터 크기 줄이기
data = data.sample(frac=0.1, random_state=72)

# 특성과 타겟 변수 분리
x = data[['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
          'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']]
y = data['Is_Lead']

# 필요없는 열 제거
x = x.drop(['ID'], axis=1)

# One-Hot Encoding
x = pd.get_dummies(x, columns=['Region_Code'])

# LabelEncoder
ob_col = list(x.dtypes[x.dtypes == 'object'].index)  # object 컬럼 리스트
for col in ob_col:
    x[col] = LabelEncoder().fit_transform(x[col].values)

# NaN 값 처리
# 'Credit_Product' 컬럼의 NaN 값을 'Unknown'으로 대체
x['Credit_Product'] = x['Credit_Product'].fillna('Unknown')

# scaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# train 데이터와 test 데이터 분할
train_data_ratio = 0.8
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y,
    test_size=1 - train_data_ratio,
    random_state=72
)

# 모델 정의
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=x_train.shape[1]))  # 뉴런 수 늘림
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))  # 뉴런 수 늘림
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])  # 학습률 조정

# 모델 훈련
model.fit(
    x_train, y_train, 
    epochs=100, 
    batch_size=128, 
    validation_data=(x_test, y_test))  # epochs, batchsize 조정

# 모델 평가
_, accuracy = model.evaluate(x_test, y_test)
print("Accuracy:", accuracy)