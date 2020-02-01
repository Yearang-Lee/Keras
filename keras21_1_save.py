# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

#model.add(Dense(5, input_dim=3))
model.add(Dense(5, input_shape=(3, )))    
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))  
                   
#model.summary()
          
model.save('./save/save_test01.h5')
print("저장 완료")