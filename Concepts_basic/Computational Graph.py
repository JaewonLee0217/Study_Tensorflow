import tensorflow as tf

node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1,node2)
#To build gragh, node, edge definite
#next, 세션만들고 (생략)그래프 그린 다음에 , 뽑아낸다
print("node1: ",node1, "node2: ",node2)
print("node3: ",node3)

#프로세스 실행과정
#1. Build graph using Tensorflow operations
#2. feed data and run graph(operation)
#3. update variables in the graph