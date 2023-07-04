from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size= 0.7,
    random_state=72,
    shuffle=True
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = RandomForestClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가
result = model.score(x_test, y_test)   
print('모델 score : ', result)   
'''
LinearSVC 적용
모델 score :  0.8947368421052632

SVC 적용
 모델 score :  0.9064327485380117

RobustScaler & DecisionTreeClassifier
모델 score :  0.9122807017543859

RobustScaler & RandomForestClassifier
모델 score :  0.9590643274853801
'''       

