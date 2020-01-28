# 1. 데이터
import numpy as np
x1 = np.array([range(1,101), range(101,201), range(301,401)])  
x2 = np.array([range(1001,1101), range(1101,1201), range(1301,1401)]) 

y1 = np.array([range(101,201)])          
                      

# print(x1.shape)  # (3, 100)  
# print(x2.shape)  # (3, 100)  
# print(y1.shape)  # (1, 100)


x1 = np.transpose(x1) 
x2 = np.transpose(x2) 
y1 = np.transpose(y1)   

# print(x1.shape)   # (100, 3)
# print(x2.shape)   # (100, 3)  
# print(y1.shape)   # (100, 1) 


from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1, train_size = 0.6, random_state = 66, shuffle=False)
x1_val, x1_test, x2_val, x2_test, y1_val, y1_test = train_test_split(x1_test, x2_test, y1_test, test_size = 0.5, random_state = 66, shuffle=False)

# print(x1_train)
# print(x1_val)
# print(x1_test)

# 2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()

# 함수형 모델 : 먼저 모델을 구성하고 이 모델이 함수형 모델이라는 것을 제일 나중에 명시한다.
input1 = Input(shape=(3,))
dense1 = Dense(5)(input1)
dense2 = Dense(2)(dense1)
dense3 = Dense(3)(dense2)
output1 = Dense(1)(dense3)

input2= Input(shape=(3,))
dense21 = Dense(7)(input2)
dense22 = Dense(4)(dense21)
output2 = Dense(5)(dense22)


from keras.layers.merge import concatenate
merge1 = concatenate([output1,output2])

middle1 = Dense(4)(merge1)
middle2 = Dense(7)(middle1)
output = Dense(1)(middle2)


model = Model(inputs = [input1,input2], outputs = output)

model.summary()


#model.add(Dense(5, input_dim=3))
# model.add(Dense(5, input_shape=(3, )))    
# model.add(Dense(2))
# model.add(Dense(3))
# model.add(Dense(1))                     

          
# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
model.fit([x1_train,x2_train], y1_train, epochs=500, batch_size=1, validation_data=([x1_val,x2_val], y1_val))  

# 4. 평가예측
loss, mse = model.evaluate([x1_test,x2_test], y1_test, batch_size=1)
print('acc : ', mse)

x1_pred = np.array([[201,202,203],[204,205,206],[207,208,209]])
x2_pred = np.array([[201,202,203],[204,205,206],[207,208,209]]) 
x1_pred = np.transpose(x1_pred)
x2_pred = np.transpose(x2_pred)

p = model.predict([x1_pred, x2_pred], batch_size=1)
print(p)

y1_predict = model.predict( [x1_test, x2_test], batch_size=1 )


# RMSE 구하기 
from sklearn.metrics import mean_squared_error
def RMSE(y1_test, y1_predict):
    return np.sqrt(mean_squared_error(y1_test,y1_predict))
print('RMSE :', RMSE(y1_test,y1_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y1_test,y1_predict)
print("R2 : ",r2_y_predict)
