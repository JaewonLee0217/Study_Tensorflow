import tensorflow as tf

#x_data는 input으로 들어가는 데이터
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]

#y_data는 일종의 라벨링을 한 것이고,
# 예를 들면 o을 cat, 1을 dog라고 가정
#최종적으로 x_data에서 3번쨰까지와 나머지 3개를 모델이 구분할 수 있도록 train을 시키는 것,
y_data = [[0],[0],[0],[1],[1],[1]]

X = tf.placeholder(tf.float32, shape =[None,2])

#Output인 Y도 선언을 해줌.
Y = tf.placeholder(tf.float32, shape = [None,1])

W = tf.Variable(tf.random_normal([2,1]),name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = tf.sigmoid(tf.matmul(X,W)+b)
#########여기까지가 그래프 정의고
#다음 부분이 학습(loss, gradient, update)

#reduce_mean이란 텐서들을 평균내서 상수(constant)로 만들어주는 함수.
cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
#reduce_mean안에있는 것이 레이블에다가 확률 로그한 값을 곱하니까 이제 cross entropy식이다.

#
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#만약 확률이 0.5가 넘으면은 1이 나오고 이하면 0이 나오게.
#시각화.
predicted = tf.cast(hypothesis>0.5, dtype = tf.float32)

#실제 값이랑 predicted 한 값이랑 비교해서 같으면은 1 이고 틀리면 0이나오는건데
#이것을 평균을 내면 정확도가 나온다.
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #학습 반복
    for step in range(10001):
        cost_val, _=sess.run([cost,train], feed_dict={X:x_data ,Y:y_data})
        if step%1000 == 0: print(step,cost_val)

        h, c, a = sess.run([hypothesis,predicted,accuracy], feed_dict={X:x_data, Y:y_data})

        print("\nHypothesis : ",h,"\nCorrect(Y): ",c,"\nAccuracy: ",a)

