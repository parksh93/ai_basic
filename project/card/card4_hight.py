import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# 데이터 로드 및 전처리
path = './data/credit_card_prediction/'
datasets = pd.read_csv(path + 'train.csv')
data = datasets.sample(frac=0.1, random_state=123)

x = data[['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
        'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']]
y = data[['Is_Lead']]

ob_col = list(x.dtypes[x.dtypes=='object'].index) # object 컬럼 리스트
for col in ob_col:
    x[col] = LabelEncoder().fit_transform(x[col].values)

x = x.fillna(x.mode().iloc[0])

# 특성 삭제
x = x.drop(['ID'], axis=1)

# 데이터 분할
train_data_ratio = 0.7
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=1-train_data_ratio,
    random_state=674)

# 파이프라인 정의
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', VotingClassifier(
        estimators=[('cat', CatBoostClassifier(
                n_estimators=1442,
                depth=4,
                fold_permutation_block=222,
                learning_rate= 0.559790864275736,
                od_pval=0.5985119942285987,
                l2_leaf_reg=0.5082927466740941,
                random_state=674
        )),
                    ('lgbm', LGBMClassifier(max_depth=10)),
                    ('xgb', XGBClassifier())],
        voting='soft',
        n_jobs=-1
    ))
])

# 파이프라인을 사용하여 모델 훈련
pipeline.fit(x_train, y_train)

# 테스트 세트 예측 및 정확도 평가
accuracy = pipeline.score(x_test, y_test)
print("Accuracy:", accuracy)

# Accuracy: 0.8712105798575789(drop X)