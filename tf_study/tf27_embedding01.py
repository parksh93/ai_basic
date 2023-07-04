import numpy as np
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

# 1. 데이터
docs = ['재밌어요', '재미없다', '돈 아깝다', '최고에요', '배우가 잘생겼어요', '추천해요', '글쎄요', '감동이다', '최악', '후회된다', '보다 나왔다', '발연기에요', '꼭봐라', '세번봐라', '또 보고싶다',
        '돈 버렸다', '다른거 볼걸', 'n회차 관람', '다음편 나왔으면 좋겠다', '연기가 어색해요', '줄거리가 이상해요', '숙면 했어요', '망작이다', '차라리 집에서 잘걸', '즐거운 시간보냈어요']

#  긍정 : 1 / 부정 : 0
labels = np.array([
    1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1
])

#  Tokenizer
token = Tokenizer()
token.fit_on_texts(docs)    # index화
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)

# pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=3)
print(pad_x)
print(pad_x.shape) # (25, 3)

# word_size => input_dim의 개수
word_size = len(token.word_index)
print('word_size : ', word_size) # 39

# 2. 모델 구성
model = Sequential()
model.add(Embedding(
    input_dim=40,   #input_dim = word_size + 1
    output_dim=16,  # output_dim = node 수
    input_length=3  # input_length = 문장의 길이(가장 긴 문장)
))
model.add(LSTM(32)) # 문장은 시간의 순서가 중요하므로 LSTM 모델 사용
model.add(Dense(1, activation='sigmoid'))   # 긍정과 부정의 이진분류

#  3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(pad_x, labels, epochs=100, batch_size=32)

# 4. 평가, 예측
loss, acc = model.evaluate(pad_x, labels)
print('loss : ', loss)
print('acc : ', acc)
'''
loss :  0.19358551502227783
acc :  1.0
'''