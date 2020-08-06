#X라는 input이 모델에 들어오면, W,b에서 곱하기, 더하기 연산이 일어나고  softymax를 거쳐서 확률
#을 출력하게 되는데, 이 사이 과정을 Affine과 예를 들어서 Relu activation함수가 담당을 한다.

import tensorflow as tf
####데이터 준비하는 part
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

####여기서부터가 그래프 그리는 part

X = tf.placeholder(tf.float32,[None,784]) # input image 28*28
Y = tf.placeholder(tf.float32, [None,10]) # label 0~9(10개)

#지금부터는 모델을 만드는 부분.
W1 = tf.Variable(tf.random_normal([784,256]))
#input 784개 받아서 output256개 만들게.
#256이 hidden layer부분이며 너무 많으면 overfitting이 발생을 하는 점에 주의.
b1 = tf.Variable(tf.random_normal([256]))
#bias는 perceptron이 256개이니까 256개 들어가야 되고
L1 = tf.nn.relu(tf.matmul(X,W1)+b1)
#최종 output은 X와 W1의 matrix multiply를 해가주고, 바이어스까지 더해서
#relu actication을 돌린 값을 output= l1으로 설정.

#이제 Layer2부분
W2 = tf.Variable(tf.random_normal([256,256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)

#이제 Layer3에서 마지막 묶어준다.
W3 = tf.Variable(tf.random_normal([256,10]))
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2,W3)+b3

#지금까지 한 것이 ,MNIST data를 처리하는 모델, 그래프의 구축 완료.
#다음으로 softmax처리하는 부분.
softmax_result = tf.nn.softmax(hypothesis)

###학습
#꿀팁.softmax -> crossentropy계산해서 loss한번에 알려주는거
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))

##그 다음은 gradient descent구하는 거
learning_rate = 0.001
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9).minimize(cost)
#loss값 최소화하도록 최적화한다는 것.
#optimizer설정을 지금은 SGD사용했지만 Adagrad,RMSProp,Momentum,Adam

####세션
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#학습 설정
training_epochs = 15#트레이닝 에포치는 전체 데이터셋을 15번 반복하겠다

batch_size = 100 #mini_batch

max = 0
early_stopped_time = 0


for epoch in range(training_epochs):
    # 학습 진행 전의 무작위로 테스트 해본 결과 처음->
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
    print('Test Accuracy:', test_accuracy)

    avg_cost =0
    total_batch = int(mnist.train.num_examples / batch_size) # 55000개를 100으로 나누니까 550개의 batch

    for i in range(total_batch): #이제 각각의 mini_batch에 대해서
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        #그러면 알아서 mnist.train의 데이터 중에서 batchsize만큼 뽑아서 준다.
        #앞에 batch_xs는 이미지, batch_ys는 레이블
        #이것을 딕셔너리를 만들어서
        feed_dict = {X:batch_xs, Y:batch_ys}
        #그리고 세션을 돌려준다.
        c, _ = sess.run([cost,optimizer],feed_dict=feed_dict)
        #그래서 나온 c 코스트는
        avg_cost += c/total_batch

    print('몇번 째 Epoch이냐:','%04d'%(epoch+1),'cost= ','{:.9f}'.format(avg_cost))


    #test
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    #argmax란 hypothesis, 즉 확률 중에 가장 큰 것이 몇번 째에 있냐를 불러오는 것.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})

    #print('Test Accuracy:', test_accuracy)

    #가장 좋은 성능을 보인 모델을 뽑아내는 코드
    if test_accuracy > max:
        max = test_accuracy
        early_stopped_time = epoch+1
print("Learning Finished!")
print("Best Accuracy: ",max)
print("Early stopped time:",early_stopped_time)





