from lightgbm import LGBMClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

# scaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# kfold
n_splits= 7
random_state = 72
kfold = StratifiedKFold(
    n_splits=n_splits,
    random_state=random_state,
    shuffle=True
)

# 2. 모델
model = LGBMClassifier()

# 3. 훈련, 평가
score = cross_val_score(
    model,
    x, y,
    cv=kfold
)

print('acc : ', score,'\ncross_val_score', round(np.mean(score), 4))
'''
acc :  [0.96341463 0.96341463 0.9382716  1.         0.96296296 0.97530864
 0.98765432]
cross_val_score 0.9701
'''