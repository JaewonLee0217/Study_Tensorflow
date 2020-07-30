import tensorflow as tf

#가상의 데이터 x_data가 있다고 가정
x_data = [[1,2],[2,3]]
#가역변수 선언(변할 수 있는 것)(x1,x2 2개 있는 형태)
X = tf.placeholder(tf.float32, shape =[None,2])
#variable: 학습과정을 통해서 업데이트가 되는 애들
#W1,W2가 1행 2열로 만들어 져있는 형태
W = tf.Variable(tf.random_normal([2,1]), name = 'weight')
#(랜덤으로 정의 되고, 2행1열짜리 벡터가 되는 애)
b = tf.Variable(tf.random_normal([1]), name = 'bias')
#텐서플로우의 뉴럴네트워크 패키지에서 시그모이드 함수를 가져와서
hypothesis = tf.nn.relu(tf.matmul(X,W)+b)

#여기까지 그래프 정의..!!
#그 다음 세션을 만든다.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
#여기서 안에서 세션을 만들어서 sess라는 이름을 만들어서 코드를 실행하겠다.
    prediction = sess.run(hypothesis, feed_dict={X:x_data})
    #아까 정의했던, x_data를 X값에 넣어줘서 최종값인 hypothesis값을 구하겠다.
    print(prediction)

#input이 2고 아웃풋이 하나이고 바이어스가 하나 달려있다.
# 이것이 마지막에 sigmoid함수를 거쳐서 prediction정의
# 최종 출력값은 [1,2]가 들어갔을 때 첫번째 [0].