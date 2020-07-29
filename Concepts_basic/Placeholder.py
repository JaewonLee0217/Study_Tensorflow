#Placeholder : 프로그램 실행 중에 값을 변경할 수 있는 가역변수
# 그런데 2.0버전 부터는 없어짐..
#placeholder 대신에 @tf.function annotation으로 지정된 함수에 값을 직접 넘겨주면 됨.
import tensorflow as tf

W = tf.Variable(tf.ones(shape=(2,2)), name="W")
b = tf.Variable(tf.zeros(shape=(2)), name="b")

@tf.function
def forward(x):
  return W * x + b

out_a = forward([1,0])
print(out_a)