############ DNN_Dense ############

from numpy import array
import numpy as np


def split_sequence2(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)- 1 :
            break
        seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix-1, :]
        X.append(seq_x)
        y.append(seq_y)
        
    return array(X), array(y)

in_seq1 = np.arange(10, 110, 10)
in_seq2 = np.arange(15, 110, 10)


out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

print(in_seq1)
print(in_seq1.shape) # (10, )
print(in_seq2)
print(in_seq2.shape) # (10, )

print(out_seq)
print(out_seq.shape) # (10, )


in_seq1 = in_seq1.reshape(len(in_seq1), 1)
in_seq2 = in_seq2.reshape(len(in_seq2), 1)
out_seq = out_seq.reshape(len(out_seq), 1)


print(in_seq1.shape)  # (10, 1)
print(in_seq2.shape)  # (10, 1)
print(out_seq.shape)  # (10, 1)


dataset = np.hstack((in_seq1, in_seq2, out_seq)) # (10, 3)
n_steps = 3

x, y = split_sequence2(dataset, n_steps)


print(dataset)
for i in range(len(x)):
    print(x[i], y[i])
    

# 실습
# 1. 함수분석
# 2. DNN 모델 만들것
# 3. 지표는 loss
# 4. x_prd = [[90, 95], [100, 105], [110, 115]]


print(x.shape)  # (7, 3, 3)

# print(x)
x= x.reshape(7, -1)

# print(x)
# 2. 모델
from keras.models import Sequential
from keras.layers import Dense



model = Sequential()
model.add(Dense(200, activation = 'relu', input_shape = (9,)))
model.add(Dense(128))
model.add(Dense(3)) # output 개수

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['acc']) # mse, mae 사용

model.fit(x,y, 
          epochs=150, batch_size = 1) 


# 평가예측
loss = model.evaluate(x, y, batch_size = 1)
print('loss', loss)


x_input = array([[90, 95, 105], [100, 105, 110], [110, 115, 120]])
x_input = x_input.reshape(1,9)

print(x_input)

y_predict = model.predict(x_input)
print(y_predict)
