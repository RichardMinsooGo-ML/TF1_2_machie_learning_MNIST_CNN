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

# Define Hyper Parameters
Alpha_Lr   = 0.001     # Learning Rate Alpha
N_EPISODES = 15
batch_size = 100

#########
# 신경망 모델 구성
######
# 기존 모델에서는 입력 값을 28x28 하나의 차원으로 구성하였으나,
# CNN 모델을 사용하기 위해 2차원 평면과 특성치의 형태를 갖는 구조로 만듭니다.
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

# 각각의 변수와 레이어는 다음과 같은 형태로 구성됩니다.
# W01_m [3 3 1 32] -> [3 3]: 커널 크기, 1: 입력값 X 의 특성수, 32: 필터 갯수
# _LAY01_m Conv shape=(?, 28, 28, 32)
#    Pool     ->(?, 14, 14, 32)
W01_m = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
# tf.nn.conv2d 를 이용해 한칸씩 움직이는 컨볼루션 레이어를 쉽게 만들 수 있습니다.
# padding='SAME' 은 커널 슬라이딩시 최외곽에서 한칸 밖으로 더 움직이는 옵션
CONV_01     = tf.nn.conv2d(X, W01_m, strides=[1, 1, 1, 1], padding='SAME')
RELU_01     = tf.nn.relu(CONV_01)
# Pooling 역시 tf.nn.max_pool 을 이용하여 쉽게 구성할 수 있습니다.
MAX_POOL_01 = tf.nn.max_pool(RELU_01, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# _LAY01_m = tf.nn.dropout(_LAY01_m, keep_prob)

# _LAY02_m Conv shape=(?, 14, 14, 64)
#    Pool     ->(?, 7, 7, 64)
# W02_m 의 [3, 3, 32, 64] 에서 32 는 _LAY01_m 에서 출력된 W01_m 의 마지막 차원, 필터의 크기 입니다.
W02_m = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
CONV_02     = tf.nn.conv2d(MAX_POOL_01, W02_m, strides=[1, 1, 1, 1], padding='SAME')
RELU_02     = tf.nn.relu(CONV_02)
MAX_POOL_02 = tf.nn.max_pool(RELU_02, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# _LAY02_m = tf.nn.dropout(_LAY02_m, keep_prob)

# FC 레이어: 입력값 7x7x64 -> 출력값 256
# Full connect를 위해 직전의 Pool 사이즈인 (?, 7, 7, 64) 를 참고하여 차원을 줄여줍니다.
#    Reshape  ->(?, 256)
W03_m = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
_LAY03_m = tf.reshape(MAX_POOL_02, [-1, 7 * 7 * 64])
FULLY_CONN_03 = tf.matmul(_LAY03_m, W03_m)
RELU_03 = tf.nn.relu(FULLY_CONN_03)
DROP_OUT_03 = tf.nn.dropout(RELU_03, keep_prob)

# 최종 출력값 _LAY03_m 에서의 출력 256개를 입력값으로 받아서 0~9 레이블인 10개의 출력값을 만듭니다.
W04_m = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
Pred_m = tf.matmul(DROP_OUT_03, W04_m)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Pred_m, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate = Alpha_Lr).minimize(cost)
# 최적화 함수를 RMSPropOptimizer 로 바꿔서 결과를 확인해봅시다.
# optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

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
        # 이미지 데이터를 CNN 모델을 위한 자료형태인 [28 28 1] 의 형태로 재구성합니다.
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)

        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs,
                                          Y: batch_ys,
                                          keep_prob: 0.7})
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
                                   keep_prob: 1}))

print('Total_batch =', total_batch)

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

