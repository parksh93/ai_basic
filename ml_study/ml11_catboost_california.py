from catboost import CatBoostRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# scaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

# kfold
n_splits= 7
random_state = 72
kfold = KFold(
    n_splits=n_splits,
    random_state=random_state,
    shuffle=True
)

# 2. 모델
model = CatBoostRegressor()

# 3. 훈련, 평가
score = cross_val_score(
    model,
    x, y,
    cv=kfold
)

print('r2 : ', score,'\ncross_val_score', round(np.mean(score), 4))
'''
r2 :  [0.85782438 0.86030451 0.83861767 0.85623304 0.84729392 0.85983063
 0.8504464 ]
cross_val_score 0.8529
'''