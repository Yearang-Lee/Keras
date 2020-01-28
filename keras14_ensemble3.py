# 1. 데이터
import numpy as np
x1 = np.array([range(1,101), range(101,201), range(301,401)])   # (3, 100) 
#x2 = np.array([range(1001,1101), range(1101,1201), range(1301,1401)])    # (3, 100)  

# y1 = np.array([range(101,201)])        # (1, 100)  
                      
y1 = np.array([range(1,101), range(101,201), range(301,401)])  
y2 = np.array([range(1001,1101), range(1101,1201), range(1301,1401)]) 
y3 = np.array([range(1,101), range(101,201), range(301,401)])     

# print(x1.shape)  
# print(x2.shape)  
# print(y1.shape)  


x1 = np.transpose(x1) 
#x2 = np.transpose(x2) 
y1 = np.transpose(y1) 
y2 = np.transpose(y2) 
y3 = np.transpose(y3) 
  

# print(x1.shape)   # (100, 3)
# print(x2.shape)   # (100, 3)  
# print(y1.shape)   # (100, 1) 


from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size = 0.6, random_state = 66, shuffle=False)
x1_val, x1_test, y1_val, y1_test = train_test_split(x1_test, y1_test, test_size = 0.5, random_state = 66, shuffle=False)

y2_train, y2_test, y3_train, y3_test = train_test_split(y2, y3, train_size = 0.6, random_state = 66, shuffle=False)
y2_val, y2_test, y3_val, y3_test = train_test_split(y2_test, y3_test, test_size = 0.5, random_state = 66, shuffle=False)

print(y3_train.shape)   # (60, 3)
print(y3_test.shape)    # (20, 3)
print(y3_val.shape)     # (20, 3)

# 2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()

# 함수형 모델 : 먼저 모델을 구성하고 이 모델이 함수형 모델이라는 것을 제일 나중에 명시한다.
input1 = Input(shape=(3,))
dense1 = Dense(5)(input1)
dense2 = Dense(3)(dense1)
output1 = Dense(4)(dense2)

# input2= Input(shape=(3,))
# dense21 = Dense(7)(input2)
# dense22 = Dense(4)(dense21)
# output2 = Dense(5)(dense22)


# from keras.layers.merge import concatenate
# merge1 = concatenate([output1,output2])

# middle1 = Dense(4)(merge1)
# middle2 = Dense(7)(middle1)
# middle3 = Dense(1)(middle2)      # 현재 merge된 마지막 레이어

output_1 = Dense(2)(output1)    # 1번째 아웃풋 모델
output_1 = Dense(3)(output_1)

output_2 = Dense(4)(output1)    # 2번째 아웃풋 모델
output_2 = Dense(4)(output_2)
output_2 = Dense(3)(output_2)


output_3 = Dense(5)(output1)    # 3번째 아웃풋 모델
output_3 = Dense(3)(output_3)


model = Model(inputs = input1 , outputs = [output_1, output_2, output_3])

model.summary()


#model.add(Dense(5, input_dim=3))
# model.add(Dense(5, input_shape=(3, )))    
# model.add(Dense(2))
# model.add(Dense(3))
# model.add(Dense(1))                     

          
# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
model.fit(x1_train, [y1_train,y2_train,y3_train], epochs=100, batch_size=1, validation_data=(x1_val, [y1_val,y2_val,y3_val]))  


# 4. 평가예측
aaa = model.evaluate(x1_test, [y1_test,y2_test,y3_test], batch_size=1)
print('aaa', aaa)  # => 결과 :[전체loss, 첫 번째 모델의 loss, 두 번째 모델의 loss, 세 번째 모델의 loss, 첫 번째 mse, 두 번째 mse, 세 번째 mse]

#loss, mse = model.evaluate([x1_test,x2_test], [y1_test,y2_test,y3_test], batch_size=1)
#print('acc : ', mse)
## 1) 변수를 1개
## 2) 변수를 mse 개수별로


x1_pred = np.array([[201,202,203],[204,205,206],[207,208,209]])
#x2_pred = np.array([[201,202,203],[204,205,206],[207,208,209]]) 

x1_pred = np.transpose(x1_pred)
#x2_pred = np.transpose(x2_pred)

p = model.predict(x1_pred, batch_size=1)
print(p)

y1_predict = model.predict( x1_test, batch_size=1 )
#print(y1_predict)   # (20,3) * 3 리스트
#print(y1_predict[0])

# RMSE 구하기 
from sklearn.metrics import mean_squared_error
def RMSE(y1_test, y1_predict):
    return np.sqrt(mean_squared_error(y1_test,y1_predict))
#print('RMSE :', RMSE(y1_test,y1_predict))

rmse1 = RMSE(y1_predict[0], y1_test)
rmse2 = RMSE(y1_predict[1], y2_test)
rmse3 = RMSE(y1_predict[2], y3_test)

rmse = (rmse1 + rmse2 + rmse3) / 3
print("RMSE : ", rmse)


# R2 구하기
from sklearn.metrics import r2_score
#r2_y_predict = r2_score(y1_test,y1_predict)
#print("R2 : ",r2_y_predict)

r2_y_predict1 = r2_score(y1_test, y1_predict[0])
r2_y_predict2 = r2_score(y1_test, y1_predict[1])
r2_y_predict3 = r2_score(y1_test, y1_predict[2])
r2_y_predict = ( r2_y_predict1 + r2_y_predict2 + r2_y_predict3 ) / 3
print("R2 Score : ", r2_y_predict)
