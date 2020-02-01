# 1. 데이터
import numpy as np
x = np.array([range(1,101), range(101,201), range(301,401)])   # (3, 100)
y = np.array([range(101,201)])                                 # (1, 100)
# y2 = np.array(range(101,201))   # (100, )

# print(x.shape)  
# print(y.shape)  
# print(y2.shape)

x = np.transpose(x)   # (100, 3)
y = np.transpose(y)   # (100, 1)

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

#model.add(Dense(5, input_dim=3))
model.add(Dense(5, input_shape=(3, )))    
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))                     
model.summary()
          
# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  

########## tensorboard ##########
from keras.callbacks import EarlyStopping, TensorBoard
th_hist = TensorBoard(log_dir='./graph', 
                      histogram_freq=0,
                      write_graph=True,
                      write_images=True)

early_stopping = EarlyStopping(monitor= 'loss', patience= 100, mode = 'auto') 
model.fit(x_train, y_train, epochs=1000, batch_size=1,validation_data=(x_val, y_val) , callbacks=[early_stopping, th_hist])

# 4. 평가예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('acc : ', mse)

x_pred = np.array([[201,202,203],[204,205,206],[207,208,209]])  
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
