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
X_for_fc = tf.reshape(outputs,[-1,hidden_size]) # -1은 보고 자동으로 설정해라는 의미.
outputs = tf.contrib.layers.fully_connected(X_for_fc,num_classes,activation_fn=None)
# softmax_w = tf.get_variable("softmax_w",[hidden_size, num_classes])
# softmax_b = tf.get_variable("softmax_b",[num_classes])
# outputs = tf.matmul(X_for_fc, softmax_w) + softmax_b ##이 과정이 fc함수에 들어있다.

outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes]) # batch 사이즈를 원래대로 복구를 시켜준다.

#웨이트는 로스들을 잘 합칠 때 가중치를 줘가주고 더하겠다
weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(
logits= outputs, targets=Y, weights=weights)
mean_loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(500): #500번을 돌려서 학습을 시킨다음에
    _, l, results = sess.run(
        [train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        print(i, j, ''.join([char_set[t] for t in index]), l)


# 마지막 글자 결과 값을 예측을 통해 출력해 보면,
results = sess.run(outputs,feed_dict={X:dataX})
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j is 0: # print all for the first result to make a sentence
        print(''.join([char_set[t] for t in index]), end='')
    else:
        print(char_set[index[-1]], end='')

