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

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=72,
    shuffle=True
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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
rf_model = RandomForestRegressor()
model = GridSearchCV(
    rf_model,       # 모델
    param,          # 하이퍼파라미터
    cv=kFold,
    verbose=1,
    refit=True,
    n_jobs=-1       # 자원(-1은 모든걸 다쓰겠다는 의미)
)

# 3. 훈련
import time;
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time

print('최적의 파라미터 : ',model.best_params_)
print('최적의 매개변수 : ', model.best_estimator_)
print('best_score : ', model.best_score_)
print('model_score : ', model.score(x_test, y_test))
print('걸린시간 : ', end_time)
'''
최적의 파라미터 :  {'n_estimators': 200, 'n_jobs': 2}
최적의 매개변수 :  RandomForestRegressor(n_estimators=200, n_jobs=2)
best_score :  0.7988694808890517
model_score :  0.8118263699549024
걸린시간 :  561.9208946228027
'''