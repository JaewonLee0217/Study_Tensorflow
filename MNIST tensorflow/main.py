import matplotlib.pyplot as plt
#그림으로 보여주면서 확인하기 위한 용도

import numpy as np
#넘 파이는 matrix, 연산 다룰 때 많이 쓰는 라이브러리

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# One hot이라는 것은 레이블 처리하는 것인데, mnist안에는 레이블이 0~9까지가 있는데, 표현하면 숫자들이
#많아서 읽기 불편하므로 5:[0,0,0,0,1,0,0,0,0,0]이런 식으로 바꿔 주는 것이다.
#그래서 이것은 cross entropy계산 시에 매우 용이 해진다.

#mnist데이터 가 어떻게 생겼는지 보면
print(np.shape(mnist.train.images))
print(np.shape(mnist.train.labels))
print(np.shape(mnist.test.images))
print(np.shape(mnist.test.labels))
#각각의 트레인 이미지가 55000개 있는 거고,
#28*28 = 784
#레이블은 0~9까지 총 10개가 존재한다.
#test할 때는 10000개의 이미지를 사용해서 test를 하였다,.


plt.imshow(
    mnist.train.images[2].reshape(28, 28),
    cmap="Greys",
    interpolation="nearest",
)
plt.show()




