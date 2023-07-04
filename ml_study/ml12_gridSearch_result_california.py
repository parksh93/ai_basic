import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#  1. 데이터
datasets = fetch_california_housing()
# x = datasets.data
x = datasets['data']
y = datasets.target

scaler = StandardScaler()
x = scaler.fit_transform(x)


kFold = KFold(      # KFold : 회기모델 / StratifiedKFold : 분류모델
    n_splits=5,
    random_state=72,
    shuffle=True
)

param = [
    {'n_estimators' : [100, 500], 'max_depth':[6, 8, 10, 12], 'n_jobs' : [-1, 2, 4]},  
    {'max_depth' : [6, 8, 10, 12], 'min_samples_split' : [2, 3, 5, 10]},
    {'n_estimators' : [100, 200], 'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5, 10], 'n_jobs' : [-1, 2, 4]}, 
    {'n_estimators' : [100, 200],'n_jobs' : [-1, 2, 4]}
]

# 2. 모델
model = RandomForestRegressor(n_estimators=200, n_jobs=1)

# 3. 훈련
import time;
start_time = time.time()
model.fit(x, y)
end_time = time.time() - start_time

print('model_score : ', model.score(x, y))
print('걸린시간 : ', end_time)
'''
model_score :  0.974736169916633
걸린시간 :  17.727771520614624
'''

# 4. 평가
score = cross_val_score(
   model, x, y, cv=kFold, verbose=1
)

print('r2 : ', score,'\ncross_val_score : ', round(np.mean(score), 4))
'''
r2 :  [0.81375484 0.80496793 0.80870841 0.81442021 0.80700277]
cross_val_score :  0.8098
'''