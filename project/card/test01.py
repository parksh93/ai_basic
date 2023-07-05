import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer

# 1. 데이터
path = './data/credit_card_prediction/'
datasets = pd.read_csv(path + 'train.csv')

print(datasets.columns)
print(datasets.head(11))

#  NaN 값 처리
imputer = SimpleImputer() 
imputer.fit(datasets)
data_rsult = imputer.transform(datasets)
print(data_rsult)

x = datasets[['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
       'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']]
y = datasets[['Is_Lead']]
print(x.shape) # (245725, 10)
print(y.shape) # (245725, 1)



# 상관계수 히트맵(heatmap)
# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set(font_scale=1.2)
# sns.set(rc={'figure.figsize':(12, 9)})
# sns.heatmap(
#     data = datasets.corr(),
#     square = True,
#     annot = True,
#     cbar = True
# )
# plt.show()

# x = x.drop(['Age','Vintage'], axis=1)

# LabelEncoder
ob_col = list(x.dtypes[x.dtypes=='object'].index) # object 컬럼 리스트
for col in ob_col:
    x[col] = LabelEncoder().fit_transform(x[col].values)

print(x.info())

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size= 0.7,
    shuffle=True,
    random_state=72
)

 # y_train을 1차원 배열로 변환
y_train = np.ravel(y_train) 

# Scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# KFold
kFold = KFold(
    n_splits=7,
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
    rf_model,      
    param,         
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


