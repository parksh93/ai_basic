import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

#  1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=72,
    shuffle=True
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

kFold = StratifiedKFold(
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
rf_model = RandomForestClassifier()
model = GridSearchCV(
    rf_model,       # 모델
    param,          # 하이퍼파라미터
    cv=kFold,
    verbose=1,
    refit=True,
    n_jobs=-1
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
최적의 파라미터 :  {'max_depth': 8, 'n_estimators': 100, 'n_jobs': -1}
최적의 매개변수 :  RandomForestClassifier(max_depth=8, n_jobs=-1)
best_score :  0.9619047619047618
model_score :  0.9333333333333333
걸린시간 :  12.125959396362305
'''