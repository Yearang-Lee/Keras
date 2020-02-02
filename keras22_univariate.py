from numpy import array

# 10개 데이터를 4개씩 자를지            # sequence:전테 데이터, n_steps:몇 개씩 자를지
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):   # 10
        end_ix = i + n_steps         # 0 + 4 = 4
        if end_ix > len(sequence)-1:   # 4 > 10-1
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)    # 0,1,2,3
        y.append(seq_y)    # 4
    return array(X), array(y)

dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
n_steps = 3

X, y = split_sequence(dataset, n_steps)

for i in range(len(X)):
    print(X[i],y[i])

# print(X)
# print(y)
        