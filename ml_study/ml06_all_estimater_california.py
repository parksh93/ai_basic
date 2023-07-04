from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import all_estimators

import warnings
warnings.filterwarnings('ignore')   # 에러 무시 : all_estimators에 cancer와 맞지 않는게 있어 에러가 날 수 있기 때문에

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
allAlgorithms = all_estimators(type_filter='regressor')

print('allAlgorithms : ', allAlgorithms)
print('몇 개? ', len(allAlgorithms)) # 55

# 3. 출력
for (name, allAlgorithms) in allAlgorithms:
    try:
        model = allAlgorithms()
        model.fit(x_train, y_train)
        result = model.score(x_test, y_test)
        print(name, '의 정답률 : ', result)
    except:
        print(name, '안 나옴')
 
'''
ARDRegression 의 정답률 :  0.6108130075965925
AdaBoostRegressor 의 정답률 :  0.4166701048182263
BaggingRegressor 의 정답률 :  0.7891805692387118
BayesianRidge 의 정답률 :  0.6108437669862823
CCA 안 나옴
DecisionTreeRegressor 의 정답률 :  0.6219105109379982
DummyRegressor 의 정답률 :  -0.0008501506355156341
ElasticNet 의 정답률 :  0.20563678465143442
ElasticNetCV 의 정답률 :  0.6102077511250081
ExtraTreeRegressor 의 정답률 :  0.536746174093497
ExtraTreesRegressor 의 정답률 :  0.8116370901919496
GammaRegressor 의 정답률 :  0.34782679532340843
GaussianProcessRegressor 의 정답률 :  -2927.6057652514055
GradientBoostingRegressor 의 정답률 :  0.7889490978700421
HistGradientBoostingRegressor 의 정답률 :  0.8420168579332129 ******
HuberRegressor 의 정답률 :  0.6022375158384744
IsotonicRegression 안 나옴
KNeighborsRegressor 의 정답률 :  0.6895229511790029
KernelRidge 의 정답률 :  -2.5900490719436555
Lars 의 정답률 :  0.6108470879666938
LarsCV 의 정답률 :  0.6101906782344111
Lasso 의 정답률 :  -0.0008501506355156341
LassoCV 의 정답률 :  0.610194477338796
LassoLars 의 정답률 :  -0.0008501506355156341
LassoLarsCV 의 정답률 :  0.6101906782344111
LassoLarsIC 의 정답률 :  0.6108470879666938
LinearRegression 의 정답률 :  0.6108470879666937
LinearSVR 의 정답률 :  0.584165244613227
MLPRegressor 의 정답률 :  0.7868668803383436
MultiOutputRegressor 안 나옴
MultiTaskElasticNet 안 나옴
MultiTaskElasticNetCV 안 나옴
MultiTaskLasso 안 나옴
MultiTaskLassoCV 안 나옴
NuSVR 의 정답률 :  0.7416799042521707
OrthogonalMatchingPursuit 의 정답률 :  0.47589541487537634
OrthogonalMatchingPursuitCV 의 정답률 :  0.6040786008046694
PLSCanonical 안 나옴
PLSRegression 의 정답률 :  0.5326530769860746
PassiveAggressiveRegressor 의 정답률 :  0.26693642303856524
PoissonRegressor 의 정답률 :  0.44862844419529746
'''