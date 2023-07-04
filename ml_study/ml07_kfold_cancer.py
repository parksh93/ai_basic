import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

datasets = load_breast_cancer()

x = datasets.data
y = datasets.target
print(x.shape) # (569, 30)


# drop feature
x = np.delete(x, [3, 4], axis=1)
print(x.shape) # (569, 28)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=72,
    shuffle=True
)

# scaler
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# kfold
n_splits = 7
random_state = 72
kfold = StratifiedKFold(
    n_splits=n_splits,
    random_state=random_state,
    shuffle=True
)

# 2. 모델
model = RandomForestClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가
score = cross_val_score(
    model,
    x, y,
    cv=kfold
)
print('score : ', score,'\ncross_val_score : ', round(np.mean(score), 4))
'''
n_split = 7
score :  [0.97560976 0.97560976 0.96296296 0.98765432 0.98765432 0.97530864 0.96296296]
cross_val_score :  0.9754

n_split = 5
score :  [0.97368421 0.96491228 0.97368421 0.99122807 0.94690265] 
cross_val_score :  0.9701

drop feature 후
cross_val_score :  0.9667
'''

# feature importance 시각화 - drop feature 했기 때문에 에러
# print(model, " : ", model.feature_importances_)
# import matplotlib.pyplot as plt

# n_features = datasets.data.shape[1]
# plt.barh(range(n_features), model.feature_importances_, align='center')
# plt.yticks(np.arange(n_features), datasets.feature_names)
# plt.title('Cancer Feature Importances')
# plt.ylabel('Feature')
# plt.xlabel('Importances')
# plt.ylim(-1, n_features)
# plt.show()