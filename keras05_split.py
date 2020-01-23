# 1. 데이터
import numpy as np
x = np.array(range(1,101))
y = np.array(range(1,101))

x_train = x[:60]
y_train = y[:60]
x_test = x[60:80]
y_test = y[60:80]
x_val = x[80:]
y_val = y[80:]


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
