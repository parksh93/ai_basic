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

# KFold
kFold = KFold(   
    n_splits=7,
    random_state=72,
    shuffle=True
)

# optuna 적용
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error

def objectiveCAT(trial: Trial, x_train, y_train, x_test):
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 500, 4000),
        'depth' : trial.suggest_int('depth', 1, 16),
        'fold_permutation_block' : trial.suggest_int('fold_permutation_block', 1, 256),
        'learning_rate' : trial.suggest_float('learning_rate', 0, 1),
        'od_pval' : trial.suggest_float('od_pval', 0, 1),
        'l2_leaf_reg' : trial.suggest_float('l2_leaf_reg', 0, 4),
        'random_state' :trial.suggest_int('random_state', 1, 2000)
    }

    # 학습 모델 생성
    model = CatBoostClassifier(**param)
    CAT_model = model.fit(x_train, y_train, verbose=True) # 학습 진행
    # 모델 성능 확인
    score = accuracy_score(CAT_model.predict(x_test), y_test)
    return score

# MAE가 최소가 되는 방향으로 학습을 진행
# TPESampler : Sampler using TPE (Tree-structured Parzen Estimator) algorithm.
study = optuna.create_study(direction='maximize', sampler=TPESampler())
# n_trials 지정해주지 않으면, 무한 반복
study.optimize(lambda trial : objectiveCAT(trial, x, y, x_test), n_trials = 5)
print('Best trial : score {}, /nparams {}'.format(study.best_trial.value, 
study.best_trial.params))
'''
[I 2023-07-06 12:08:52,027] Trial 4 finished with value: 0.49751756694430127 and parameters: {'n_estimators': 1442, 'depth': 4, 'fold_permutation_block': 222, 'learning_rate': 0.559790864275736, 'od_pval': 0.5985119942285987, 'l2_leaf_reg': 0.5082927466740941, 'random_state': 674}. Best is trial 4 with value: 0.49751756694430127.
Best trial : score 0.49751756694430127, /nparams {'n_estimators': 1442, 'depth': 4, 'fold_permutation_block': 222, 'learning_rate': 0.559790864275736, 'od_pval': 0.5985119942285987, 'l2_leaf_reg': 0.5082927466740941, 'random_state': 674}
'''