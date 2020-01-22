# 1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# print(x.shape)  # 1차원 임을 확인(input_dim=1)
# print(y.shape)

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(30, input_dim=1, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(1))  # 0 또는 1의 값이 아니기 때문에 모델의 마지막 출력 계층에서 Activation Function를 지정할 필요가 없다.
          
# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  # MSE(target과 predict의 차이값의 제곱의 평균), RMSE(MSE에 root취한 값),  MAE(target과 predict의 차이 절대값), RMAE(MAE에 root취한 값)
model.fit(x,y, epochs=500, batch_size=10)   # batch_size=1 : 10개의 데이터를 1개씩 잘라서 학습 -> 낮을 수록 정확도 올라감

# 4. 평가예측
loss, mse = model.evaluate(x, y, batch_size=10)
print('acc : ', mse)

x_pred = np.array([11,12,13])
p = model.predict(x_pred, batch_size=10)
print(p)

# pp = model.predict(x, batch_size=1)
# print(pp)



