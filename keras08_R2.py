# 1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(1,101))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.6, random_state = 0, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = 0.5, random_state = 0, shuffle=False)

print(x_train)
print(x_val)
print(x_test)


# print(x.shape)  # 1차원 임을 확인(input_dim=1)
# print(y.shape)

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

#model.add(Dense(5, input_dim=1))
model.add(Dense(5, input_shape=(1, )))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))  

model.summary()
          
# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
model.fit(x_train,y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))  

# 4. 평가예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('acc : ', mse)

x_pred = np.array([101,102,103])
p = model.predict(x_pred, batch_size=1)
print(p)

y_predict = model.predict(x_test, batch_size=1)

# 회기 분석의 최종 결과 확인 : RMSE, R2로 확인 
# RMSE 구하기 
# 낮을 수록 좋다.
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print('RMSE :', RMSE(y_test,y_predict))

# R2 구하기
# R2의 max는 1(1에 가까울 수록 좋다.)
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test,y_predict)
print("R2 : ",r2_y_predict)
