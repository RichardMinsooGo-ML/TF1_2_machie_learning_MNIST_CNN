# 신경망 구성을 손쉽게 해 주는 유틸리티 모음인 tensorflow.layers 를 사용해봅니다.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time, datetime
from tensorflow.examples.tutorials.mnist import input_data
# 최신 Windows Laptop에서만 사용할것.CPU Version이 높을때 사용.
# AVX를 지원하는 CPU는 Giuthub: How to compile tensorflow using SSE4.1, SSE4.2, and AVX. 
# Ubuntu와 MacOS는 지원하지만 Windows는 없었음. 2018-09-29
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compuntational Graph Initialization
from tensorflow.python.framework import ops
ops.reset_default_graph()

DATA_DIR = "/tmp/ML/MNIST_data"
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

Alpha_Lr = 0.001
N_EPISODES = 15
batch_size = 100

#########
# 신경망 모델 구성
######
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)

# 기본적으로 inputs, outputs size, kernel_size 만 넣어주면
# 활성화 함수 적용은 물론, 컨볼루션 신경망을 만들기 위한 나머지 수치들은 알아서 계산해줍니다.
# 특히 Weights 를 계산하는데 xavier_initializer 를 쓰고 있는 등,
# 크게 신경쓰지 않아도 일반적으로 효율적인 신경망을 만들어줍니다.
CONV_01     = tf.layers.conv2d(X, 32, [3, 3], activation=tf.nn.relu)
MAX_POOL_01 = tf.layers.max_pooling2d(CONV_01, [2, 2], [2, 2])
DROP_OUT_01 = tf.layers.dropout(MAX_POOL_01, 0.7, is_training)

CONV_02     = tf.layers.conv2d(DROP_OUT_01, 64, [3, 3], activation=tf.nn.relu)
MAX_POOL_02 = tf.layers.max_pooling2d(CONV_02, [2, 2], [2, 2])
DROP_OUT_02 = tf.layers.dropout(MAX_POOL_02, 0.7, is_training)

_LAY03_m = tf.contrib.layers.flatten(DROP_OUT_02)
_LAY03_m = tf.layers.dense(_LAY03_m, 256, activation=tf.nn.relu)
DROP_OUT_03 = tf.layers.dropout(_LAY03_m, 0.5, is_training)

Pred_m = tf.layers.dense(DROP_OUT_03, 10, activation=None)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Pred_m, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate = Alpha_Lr).minimize(cost)

#########
# 신경망 모델 학습
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

start_time = time.time()
print('Learning Started!')

total_batch = int(mnist.train.num_examples / batch_size)

for episode in range(N_EPISODES):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs,
                                          Y: batch_ys,
                                          is_training: True})
        total_cost += cost_val

    print('episode:', '%05d' % (episode + 1),
          'Avg. cost =', '{:.5f}'.format(total_cost / total_batch))
    
    elapsed_time = datetime.timedelta(seconds=int(time.time()-start_time))
    print("[{}]".format(elapsed_time))

print('Optimization Completed!')

#########
# 결과 확인
######
is_correct = tf.equal(tf.argmax(Pred_m, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('Accuracy:', sess.run(accuracy,
                        feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1),
                                   Y: mnist.test.labels,
                                   is_training: False}))

elapsed_time = time.time() - start_time
formatted = datetime.timedelta(seconds=int(elapsed_time))
print("=== training time elapsed: {}s ===".format(formatted))

#########
# 결과 확인 (matplot)
######
labels = sess.run(Pred_m,
                  feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1),
                             Y: mnist.test.labels,
                             keep_prob: 1})

fig = plt.figure()
for i in range(60):
    subplot = fig.add_subplot(4, 15, i + 1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(labels[i]))
    subplot.imshow(mnist.test.images[i].reshape((28, 28)),
                   cmap=plt.cm.gray_r)

plt.show()


# 세션을 닫습니다.
sess.close()


# Step 10. Tune hyperparameters:
# Step 11. Deploy/predict new outcomes:

