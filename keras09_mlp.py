# 1. 데이터
import numpy as np
x = np.array([range(1,101), range(101,201)])    # (2,100)  # vector가 2개, 즉. 열이 2개
y = np.array([range(1,101), range(101,201)])    # (2,100)
x = np.transpose(x)
y = np.transpose(y)

print(x.shape)  
print(y.shape)  


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.6, random_state = 0, shuffle=False)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = 0.5, random_state = 0, shuffle=False)

# print(x_train)
# print(x_val)
# print(x_test)

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

#model.add(Dense(5, input_dim=2))
model.add(Dense(5, input_shape=(2, )))    # input = 2
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(2))                       # output = 2

model.summary()
          
# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
model.fit(x_train,y_train, epochs=500, batch_size=1, validation_data=(x_val, y_val))  

# 4. 평가예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('acc : ', mse)

x_pred = np.array([[201,202,203],[204,205,206]])  
x_pred = np.transpose(x_pred)
p = model.predict(x_pred, batch_size=1)
print(p)

y_predict = model.predict(x_test, batch_size=1)


# RMSE 구하기 
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print('RMSE :', RMSE(y_test,y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test,y_predict)
print("R2 : ",r2_y_predict)
