
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils import all_estimators

import warnings
warnings.filterwarnings('ignore')   # 에러 무시 : all_estimators에 cancer와 맞지 않는게 있어 에러가 날 수 있기 때문에

# 1. 데이터
datasets = load_iris()
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
allAlgorithms = all_estimators(type_filter='classifier')

print('allAlgorithms : ', allAlgorithms)
print('몇 개? ', len(allAlgorithms)) # 41

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
AdaBoostClassifier 의 정답률 :  0.9333333333333333
BaggingClassifier 의 정답률 :  0.9333333333333333
BernoulliNB 의 정답률 :  0.7333333333333333
CalibratedClassifierCV 의 정답률 :  0.8888888888888888
CategoricalNB 안 나옴
ClassifierChain 안 나옴
ComplementNB 안 나옴
DecisionTreeClassifier 의 정답률 :  0.9333333333333333
DummyClassifier 의 정답률 :  0.28888888888888886
ExtraTreeClassifier 의 정답률 :  0.8888888888888888
ExtraTreesClassifier 의 정답률 :  0.9333333333333333
GaussianNB 의 정답률 :  0.9333333333333333
GaussianProcessClassifier 의 정답률 :  0.9111111111111111
GradientBoostingClassifier 의 정답률 :  0.9555555555555556
HistGradientBoostingClassifier 의 정답률 :  0.9555555555555556
KNeighborsClassifier 의 정답률 :  0.9333333333333333
LabelPropagation 의 정답률 :  0.9333333333333333
LabelSpreading 의 정답률 :  0.9333333333333333
LinearDiscriminantAnalysis 의 정답률 :  1.0
LinearSVC 의 정답률 :  0.9333333333333333
LogisticRegression 의 정답률 :  0.9333333333333333
LogisticRegressionCV 의 정답률 :  0.9333333333333333
MLPClassifier 의 정답률 :  0.9333333333333333
MultiOutputClassifier 안 나옴
MultinomialNB 안 나옴
NearestCentroid 의 정답률 :  0.8666666666666667
NuSVC 의 정답률 :  0.9333333333333333
OneVsOneClassifier 안 나옴
OneVsRestClassifier 안 나옴
OutputCodeClassifier 안 나옴
PassiveAggressiveClassifier 의 정답률 :  0.9333333333333333
Perceptron 의 정답률 :  0.9333333333333333
QuadraticDiscriminantAnalysis 의 정답률 :  1.0
RadiusNeighborsClassifier 의 정답률 :  0.8444444444444444
RandomForestClassifier 의 정답률 :  0.9333333333333333
RidgeClassifier 의 정답률 :  0.8444444444444444
RidgeClassifierCV 의 정답률 :  0.8444444444444444
SGDClassifier 의 정답률 :  0.8888888888888888
SVC 의 정답률 :  0.9333333333333333
StackingClassifier 안 나옴
VotingClassifier 안 나옴
'''

