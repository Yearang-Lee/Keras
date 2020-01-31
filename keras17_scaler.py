from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20000,30000,40000], [30000,40000,50000], [40000,50000,60000], [100,200,300]])
y = array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000,400])


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler

# # StandardScaler(x) : 평균이 0과 표준편차가 1이 되도록 변환
# scaler = StandardScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# # print(x)

# # MinMaxScaler(X): 최대값이 각각 1, 최소값이 0이 되도록 변환
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# print(x)

# train은 10개, 나머지는 test
# Dense모델로 구현
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, random_state = 0, shuffle=False)

scaler = StandardScaler()

scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)

x_test_scaled = scaler.transform(x_test)




# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

#model.add(Dense(5, input_dim=3))
model.add(Dense(5, input_shape=(3, )))    
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))                     
model.summary()
          
# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
model.fit(x_train_scaled,y_train, epochs=500, batch_size=1)  

# 4. 평가예측
loss, mse = model.evaluate(x_test_scaled, y_test, batch_size=1)
print('acc : ', mse)

x_pred = np.array([[201,202,203]])  
x_pred_scaled = scaler.transform(x_pred)
#x_pred = np.transpose(x_pred)
p = model.predict(x_pred_scaled, batch_size=1)
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