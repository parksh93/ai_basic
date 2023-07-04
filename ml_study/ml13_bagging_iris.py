from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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
model = BaggingClassifier(
    dt_model,
    n_estimators=100,
    n_jobs=-1,
    random_state=72
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
acc score :  0.9333333333333333 
scross_val_score :  0.9397
'''