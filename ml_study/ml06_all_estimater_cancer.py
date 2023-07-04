
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils import all_estimators

import warnings
warnings.filterwarnings('ignore')   # 에러 무시 : all_estimators에 cancer와 맞지 않는게 있어 에러가 날 수 있기 때문에

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
AdaBoostClassifier 의 정답률 :  0.9766081871345029
BaggingClassifier 의 정답률 :  0.9181286549707602
BernoulliNB 의 정답률 :  0.8888888888888888
CalibratedClassifierCV 의 정답률 :  0.9532163742690059
CategoricalNB 안 나옴
ClassifierChain 안 나옴
ComplementNB 안 나옴
DecisionTreeClassifier 의 정답률 :  0.8947368421052632
DummyClassifier 의 정답률 :  0.6140350877192983
ExtraTreeClassifier 의 정답률 :  0.8713450292397661
ExtraTreesClassifier 의 정답률 :  0.9649122807017544
GaussianNB 의 정답률 :  0.9122807017543859
GaussianProcessClassifier 의 정답률 :  0.9649122807017544
GradientBoostingClassifier 의 정답률 :  0.9473684210526315
HistGradientBoostingClassifier 의 정답률 :  0.9532163742690059
KNeighborsClassifier 의 정답률 :  0.9590643274853801
LabelPropagation 의 정답률 :  0.9122807017543859
LabelSpreading 의 정답률 :  0.9122807017543859
LinearDiscriminantAnalysis 의 정답률 :  0.935672514619883
LinearSVC 의 정답률 :  0.9590643274853801
LogisticRegression 의 정답률 :  0.9707602339181286
LogisticRegressionCV 의 정답률 :  0.9766081871345029
MLPClassifier 의 정답률 :  0.9590643274853801
MultiOutputClassifier 안 나옴
MultinomialNB 안 나옴
NearestCentroid 의 정답률 :  0.9122807017543859
NuSVC 의 정답률 :  0.9298245614035088
OneVsOneClassifier 안 나옴
OneVsRestClassifier 안 나옴
OutputCodeClassifier 안 나옴
PassiveAggressiveClassifier 의 정답률 :  0.9473684210526315
Perceptron 의 정답률 :  0.9415204678362573
QuadraticDiscriminantAnalysis 의 정답률 :  0.9532163742690059
RadiusNeighborsClassifier 안 나옴
RandomForestClassifier 의 정답률 :  0.9590643274853801
RidgeClassifier 의 정답률 :  0.9415204678362573
RidgeClassifierCV 의 정답률 :  0.9415204678362573
SGDClassifier 의 정답률 :  0.9415204678362573
SVC 의 정답률 :  0.9766081871345029 ****
StackingClassifier 안 나옴
VotingClassifier 안 나옴
'''

