import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

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
Trial 4 finished with value: 0.37777777777777777 and parameters: {'n_estimators': 916, 'depth': 11, 'fold_permutation_block': 250, 'learning_rate': 0.08920740294945129, 'od_pval': 0.4600378418166219, 'l2_leaf_reg': 1.4296809112239943, 'random_state': 1733}. Best is trial 1 with value: 0.6.
Best trial : score 0.6, /nparams {'n_estimators': 3944, 'depth': 16, 'fold_permutation_block': 9, 'learning_rate': 0.36830000188731193, 'od_pval': 0.3460800077340924, 'l2_leaf_reg': 2.0488288593341157, 'random_state': 1445}
'''