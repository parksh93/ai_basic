from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test,  y_train, y_test  = train_test_split(x, y, train_size=0.7, random_state=72, shuffle=True)

print(x_train.shape, y_train.shape) # (14447, 8) (14447,)
print(x_test.shape, y_test.shape)   # (6193, 8) (6193,)

# Scaler 적용
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = RandomForestRegressor()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가
result = model.score(x_test, y_test)
print('r2 :', result)   # 회기모델의 model.score값은 r2 score
'''
LinearSVR 모델
r2 : -4.76774649302304

SVC 모델
r2 : -0.015599163337713495

StandardScaler
r2 : 0.7403367455387826

RobustScaler
r2 : 0.673360862057929

MinMaxScaler
r2 : 0.6669097115577558

MaxAbsScaler
r2 : 0.5802961600616365

DecisionTreeRegressor & StandardScaler
r2 : 0.6249593745471063

RandomForestRegressor & StandardScaler
r2 : 0.8092003092428421
'''