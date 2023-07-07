import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sns
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 1. 데이터
path = './data/credit_card_prediction/'
datasets = pd.read_csv(path + 'train.csv')

print(datasets.columns)

print('******Labeling 전 데이터*****')
print(datasets.head(11))

# 문자를 숫자로 변경 (LabelEncoder)
df = pd.DataFrame(datasets)

ob_col = list(df.dtypes[df.dtypes=='object'].index) # object 컬럼 리스트

# NaN 처리 (NaN값을 Unknown으로 변경)
df['Credit_Product'].fillna('Unknown', inplace=True)

# NaN값 처리 후 라벨링
for col in ob_col:
    df[col] = LabelEncoder().fit_transform(df[col].values)
    
print('******Labeling 후 데이터*****')
print(datasets.head(11))
    
# 상관계수 히트맵(heatmap)
# sns.set(font_scale=1.2)
# sns.set(rc={'figure.figsize':(12, 9)})
# sns.heatmap(
#     data = datasets.corr(),
#     square = True,
#     annot = True,
#     cbar = True
# )
# plt.show()   

x = datasets[['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
       'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']]
y = datasets[['Is_Lead']]

print(x.shape) # (245725, 10)
print(y.shape) # (245725, 1)

# 필요없는 칼럼 제거 : Age와 Vintage의 상관계수 0.63으로 가장 높음
x = x.drop(['Age', 'Vintage'], axis=1)

# train 데이터와 test 데이터 분할
train_data_ratio = 0.7
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-train_data_ratio, random_state=123)

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# 결과를 출력하여 분할이 성공적으로 완료되었는지 확인합니다.
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# scaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

'''
scaler 미적용
    Accuracy: 0.840573812188422

StandardScaler
    Accuracy: 0.840573812188422

MinMaxScaler
    Accuracy: 0.840573812188422

MaxAbsScaler
    Accuracy: 0.840573812188422

RobustScaler
    Accuracy: 0.840573812188422
'''

# KFold
kFold = KFold(   
    n_splits=7,
    random_state=72,
    shuffle=True
)

# # DecisionTree
# model = RandomForestClassifier(random_state=42)


# gridSearch
# WARNING: Parameters: { "min_samples_split" } are not used. -> min_smaples_split 삭제
param = [
    {'n_estimators' : [100, 500], 'max_depth':[6, 8, 10, 12], 'n_jobs' : [-1, 2, 4]},  
    {'max_depth' : [6, 8, 10, 12]},
    {'n_estimators' : [100, 200], 'min_samples_leaf' : [3, 5, 7, 10]},
    {'n_estimators' : [100, 200],'n_jobs' : [-1, 2, 4]}
]

# 2. 모델
'''
lgbm_model = LGBMClassifier(max_depth=10)
model = GridSearchCV(
    lgbm_model,      
    param,          
    cv=kFold,
    verbose=1,
    refit=True,
    n_jobs=-1       
)

cat = CatBoostClassifier()
lgbm = LGBMClassifier()
xgb = XGBClassifier()
# model = VotingClassifier(
#     estimators=[('cat', cat), ('lgbm', lgbm), ('xgb', xgb)],
#     voting='soft',
#     # voting='hard',
#     n_jobs=-1
# )

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가
# from sklearn.metrics import accuracy_score
# classifiers = [cat, lgbm, xgb]
# for model in classifiers:
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
#     score = accuracy_score(y_pred, y_test)
#     class_names = model.__class__.__name__
#     print('{0} 정확도 : {1: .4f}'.format(class_names, score))

# result = model.score(x_test, y_test)
# print('voting 결과 : ', result)
'''

'''
CatBoostClassifier 정확도 :  0.8534
LGBMClassifier 정확도 :  0.8545 ***
XGBClassifier 정확도 :  0.8541
voting 결과 :  0.8540790580319596
voting결과 정화도가 높은 Classfier는 LGBMClassfier이다
'''

model = LGBMClassifier(max_depth=10)
# 훈련
model.fit(x_train, y_train)

# print('최적의 파라미터 : ',model.best_params_)
# print('최적의 매개변수 : ', model.best_estimator_)
# print('best_score : ', model.best_score_)
# print('model_score : ', model.score(x_test, y_test))

'''
LGBMClassifier 
    최적의 파라미터 :  {'max_depth': 10, 'n_estimators': 100, 'n_jobs': -1}
    최적의 매개변수 :  LGBMClassifier(max_depth=10)
    best_score :  0.8523548279826999
    model_score :  0.8544181882308256
'''


# 모델을 사용하여 테스트 세트를 예측하고 정확도를 출력합니다.
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# DecisionTreeRegressor = Accuracy: 0.7840879031437583
# RandomForestClassifier = Accuracy: 0.8413266863363517
# LGBMClassifier = Accuracy: 0.8544181882308256
