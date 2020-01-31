from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM
import numpy as np

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20000,30000,40000], [30000,40000,50000], [40000,50000,60000], [100,200,300]])
y = array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000,400])


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler



# 실습
# train은 10개, 나머지는 test
# RNN 모델 구현
# R2 지표
# [250,260, 270] 으로 predict


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, random_state = 0, shuffle=False)




scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)



# 2. 모델 구성
model = Sequential()
model.add(LSTM(10, activation = 'relu',input_shape = (3,1), return_sequences = True))   # return_sequences = True : LSTM의 아웃풋을 다음 input layer 차원에 맞춰주는 거(defalut : False)
model.add(LSTM(2, activation = 'relu', return_sequences = True))  
model.add(LSTM(3, activation = 'relu', return_sequences = True)) 
model.add(LSTM(4, activation = 'relu', return_sequences = True)) 
model.add(LSTM(5, activation = 'relu', return_sequences = True)) 
model.add(LSTM(6, activation = 'relu', return_sequences = True)) 
model.add(LSTM(7, activation = 'relu', return_sequences = True)) 
model.add(LSTM(8, activation = 'relu', return_sequences = True)) 
model.add(LSTM(9, activation = 'relu', return_sequences = True)) 
model.add(LSTM(10, activation = 'relu', return_sequences = False)) 

model.add(Dense(5, activation = 'relu'))
model.add(Dense(1))

model.summary()
          
# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
model.fit(x_train,y_train, epochs=500, batch_size=1)  

# 4. 평가예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('acc : ', mse)

x_input = np.array([[250,260,270]])
x_input = x_input.reshape(1,3,1)
x_pred = model.predict(x_input, batch_size=1)
print(x_pred)

y_predict = model.predict(x_test, batch_size=1)


# RMSE 구하기 
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print('RMSE :', RMSE(y_test,y_predict))

# # R2 구하기
# from sklearn.metrics import r2_score
# r2_y_predict = r2_score(y_test,y_predict)
# print("R2 : ",r2_y_predict)