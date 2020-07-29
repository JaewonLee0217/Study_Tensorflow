#세션 학습
import tensorflow as tf
# tensorflow 라이브러리를 tf로 쓴다
# tf.Session()경우 텐서플로우 버전 1.x.x에서 사용하는 표현방식이므로
# 확인결과, 2.X 버전 경우 Session을 정의하고 Run해주는 과정이 생략된다.
# 버전 확인 방법print(tf.__version__)
node = tf.constant('Hello , My name is Jaewon')
node1 = tf.constant(3.0)
node2 = tf.constant(4.0)
tf.print(node,node1,node2)

#constant는 한 번 정해지면 변하지 않는 , 상수항을 의미.


