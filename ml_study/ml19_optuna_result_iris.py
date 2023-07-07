import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostClassifier

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
    random_state=1733,
    shuffle=True
)

model = CatBoostClassifier(
    n_estimators=916,
    depth=1,
    fold_permutation_block=250,
    learning_rate= 0.08920740294945129,
    od_pval=0.4600378418166219,
    l2_leaf_reg=1.4296809112239943,
    random_state=1733
)

# 3. 훈련, 평가
score = cross_val_score(
    model,
    x, y,
    cv=kFold
)

print('acc : ', score,'\ncross_val_score', round(np.mean(score), 4))