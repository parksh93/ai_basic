from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y,
#     train_size= 0.7,
#     random_state=72,
#     shuffle=True
# )

scaler = StandardScaler()
x = scaler.fit_transform(x)

kFold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=72
)

# 2. 모델
model = SVR()

# 3. 훈련, 평가
score = cross_val_score(
    model,
    x, y,
    cv=kFold
)

print('r2 : ', score, '\n cross_val_score : ', round(np.mean(score), 4))
'''
r2 :  [0.74977437 0.72822127 0.73631372 0.75289926 0.72902466] 
 cross_val_score :  0.7392
'''
