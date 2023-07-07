from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# 1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

feature_name = datasets.feature_names
print(feature_name)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    shuffle=True,
    random_state=72
)

# scaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# kfold
n_splits = 5
random_state = 62
kfold = StratifiedKFold(
    n_splits=n_splits,
    random_state=random_state,
    shuffle=True,
)

# 2. 모델
model = XGBClassifier()

# 3. 훈련
model.fit(
    x_train, y_train,
    early_stopping_rounds=20,
    eval_set = [(x_train, y_train), (x_test, y_test)],
    eval_metric = 'merror'
)

# 4. 평가
score = cross_val_score(
    model,
    x_train, y_train,
    cv=kfold
)
y_pred = cross_val_predict(
    model,
    x_test, y_test,
    cv=kfold
)

acc = accuracy_score(y_test, y_pred)

print('score : ', score)
print('acc : ', acc)

# score :  [0.95238095 0.9047619  0.95238095 0.9047619  1.        ]
# acc :  0.9333333333333333

# Select_From_Model
thresholds = model.feature_importances_

for thresh in thresholds:
    selction = SelectFromModel(
        model, threshold=thresh, prefit=True
    )
    select_x_train = selction.transform(x_train)
    select_x_test = selction.transform(x_test)

    selection_model = XGBClassifier()
    selection_model.fit(select_x_train, y_train)
    y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, ACC:%.2f%%"%(thresh, select_x_train.shape[1], score*100))

    # 컬럼명 출력
    selected_feature_indices = selction.get_support(indices=True)
    selected_feature_names = [feature_name[i] for i in selected_feature_indices]
    print(selected_feature_names)