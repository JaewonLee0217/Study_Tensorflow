import tensorflow as tf
import numpy as np
from tensorflow.contrib import  rnn

sentence = ("if you want to build a ship, don't drum up people together to "
"collect wood and don't assign them tasks and work, but rather "
"teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char_dic = {w:i for i,w in enumerate(char_set)}

hidden_size = 50
num_classes = len(char_set)
sequence_length = 10 # 10글자 짤라내서 돌리고
learning_rate = 0.1

#################
dataX = []
dataY = []
for i in range(0,len(sentence)- sequence_length):
    x_str = sentence[i:i+sequence_length]
    y_str = sentence[i+1:i+sequence_length+1]
    print(i,x_str,"->",y_str) #10글자씩 시퀀스로 짤라서 그다음 시퀀스가 나온다라고 학습을 진행할 것임.


    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]

    dataX.append(x);
    dataY.append(y);

batch_size = len(dataX)

X = tf.placeholder(tf.int32, [None,sequence_length])
Y = tf.placeholder(tf.int32, [None,sequence_length])

#원 핫 인코딩시켜줘야 됌

X_one_hot = tf.one_hot(X,num_classes)
print(X_one_hot)

def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple =True)
    return cell

multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(2)],state_is_tuple=True)

#결과 는 unfolding size x 히든 사이즈,
outputs, _states = tf.nn.dynamic_rnn(multi_cells,X_one_hot,dtype=tf.float32)

#fully connected layer FC
X_for_fc = tf.reshape(outputs,[-1,hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc,num_classes,activation_fn=None)
# softmax_w = tf.get_variable("softmax_w",[hidden_size, num_classes])
# softmax_b = tf.get_variable("softmax_b",[num_classes])
# outputs = tf.matmul(X_for_fc, softmax_w) + softmax_b
