from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.metrics import accuracy_score

# 1. 데이터
x_data = [[0,0],[0,1],[1,0], [1,1]]
y_data = [0, 1, 1, 1]

# 2. 모델
model = Perceptron()

# 3. 훈련 (컴파일X)
model.fit(x_data, y_data)

# 4. 평가, 예측
result = model.score(x_data, y_data)    # 딥러닝에서는 model.evaluate
print('모델 score : ', result)          

y_predict = model.predict(x_data)
print(x_data,'의 예측결과 : ', y_predict)

acc = accuracy_score(y_data, y_predict)
print('acc : ', acc)
'''
[[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과 :  [0 1 1 1]
acc :  1.0
'''