from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np

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
    n_splits=7,
    random_state=72,
    shuffle=True
)

dt_model = DecisionTreeClassifier()
bagging = BaggingClassifier(
    dt_model,
    n_estimators=100,
    n_jobs=-1,
    random_state=72
)

param = {
    'n_estimators': [100],
    'random_state': [42, 52, 62, 72],
    'max_features': [3, 4, 7]
}

model = GridSearchCV(
    bagging,
    param,
    refit=True,
    verbose=1,
    cv=kFold,
    n_jobs=-1,  # -1 : 모든 cpu를 사용/ 1 : 하나만 사용(catboost는 1)
    # pre_dispatch=2  #catboost 모델은 병렬처리해야 함
)


start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time

result = model.score(x_test, y_test)
score = cross_val_score(
    model,
    x, y,
    cv=kFold
)
print('acc score : ', result,'\ncross_val_score : ', round(np.mean(score), 4))
'''
acc score :  0.9555555555555556 
cross_val_score :  0.9397
'''