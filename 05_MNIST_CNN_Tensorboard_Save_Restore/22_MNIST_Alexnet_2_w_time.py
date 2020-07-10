import tensorflow as tf
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

########## set net hyperparameters ##########
N_EPISODES   = 15
batch_size   = 100
display_step = 30
Alpha_Lr     = 0.001   # Learning Rate Alpha

# Network Parameters
INPUT_SIZE  = 784      # MNIST data input (img shape: 28*28)
OUTPUT_SIZE = 10       # MNIST total classes (0-9 digits)
dropout     = 0.7      # Dropout, probability to keep units

########## placeholder ##########
x = tf.placeholder(tf.float32,[None,INPUT_SIZE])
y = tf.placeholder(tf.float32,[None,OUTPUT_SIZE])

########## define conv process ##########
def CONVOLUTION(name,x,W,b,strides=1, padding='SAME'):
    x=tf.nn.conv2d(x,W,strides=[1,strides,strides,1],padding=padding)
    x=tf.nn.bias_add(x,b)
    return tf.nn.relu(x,name=name)

########## define pool process ##########
def POOLING(name, x, k=3, s=2, padding='SAME'):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,s,s,1],padding=padding,name=name)

########## define NORMALIZATION process ##########
def NORMALIZATION(name, l_input, lsize=5):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.0001, beta=0.75, name=name)

########## set net parameters ##########
def weight_var(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_var(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))

weights={
    'WEI_CONV_01': weight_var('WEI_CONV_01',[11,11,1,96]),
    'WEI_CONV_02': weight_var('WEI_CONV_02',[5,5,96,256]),
    'WEI_CONV_03': weight_var('WEI_CONV_03',[3,3,256,384]),
    'WEI_CONV_04': weight_var('WEI_CONV_04',[3,3,384,384]),
    'WEI_CONV_05': weight_var('WEI_CONV_05',[3,3,384,256]),
    'WEI_DENS_01': weight_var('WEI_DENS_01',[4*4*256,4096]),
    'WEI_DENS_02': weight_var('WEI_DENS_02',[4096,4096]),
    'out_w': weight_var('out_w',[4096,10])
}
biases={
    'BIA_CONV_01': bias_var('BIA_CONV_01',[96]),
    'BIA_CONV_02': bias_var('BIA_CONV_02',[256]),
    'BIA_CONV_03': bias_var('BIA_CONV_03',[384]),
    'BIA_CONV_04': bias_var('BIA_CONV_04',[384]),
    'BIA_CONV_05': bias_var('BIA_CONV_05',[256]),
    'BIA_DENS_01': bias_var('BIA_DENS_01',[4096]),
    'BIA_DENS_02': bias_var('BIA_DENS_02',[4096]),
    'out_b': bias_var('out_b',[OUTPUT_SIZE])
}

"""
########## set net parameters ##########
weights = {
    'WEI_CONV_01': tf.Variable(tf.random_normal([11,11,1,96])),
    'WEI_CONV_02': tf.Variable(tf.random_normal([5,5,96,256])),
    'WEI_CONV_03': tf.Variable(tf.random_normal([3,3,256,384])),
    'WEI_CONV_04': tf.Variable(tf.random_normal([3,3,384,384])),
    'WEI_CONV_05': tf.Variable(tf.random_normal([3,3,384,256])),
    'WEI_DENS_01': tf.Variable(tf.random_normal([4*4*256,4096])),
    'WEI_DENS_02': tf.Variable(tf.random_normal([4096,4096])),
    'out_w': tf.Variable(tf.random_normal([4096,10]))
}
biases = {
    'BIA_CONV_01': tf.Variable(tf.random_normal([96])),
    'BIA_CONV_02': tf.Variable(tf.random_normal([256])),
    'BIA_CONV_03': tf.Variable(tf.random_normal([384])),
    'BIA_CONV_04': tf.Variable(tf.random_normal([384])),
    'BIA_CONV_05': tf.Variable(tf.random_normal([256])),
    'BIA_DENS_01': tf.Variable(tf.random_normal([4096])),
    'BIA_DENS_02': tf.Variable(tf.random_normal([4096])),
    'out_b': tf.Variable(tf.random_normal([OUTPUT_SIZE]))
}
"""

##################### build net model ##########################

########## define net structure ##########
def ALEX_NET(x, weights, biases, dropout):
    #### reshape input picture ####
    x=tf.reshape(x, shape=[-1,28,28,1])

    # CONVOLUTION Layer
    CONV_01     = CONVOLUTION('CONV_01', x, weights['WEI_CONV_01'], biases['BIA_CONV_01'], padding='SAME')
    # Max Pooling (down-sampling)
    MAX_POOL_01 = POOLING('MAX_POOL_01',CONV_01,k=3, s=2, padding='SAME')
    # Apply Normalization
    NORM_01     = NORMALIZATION('NORM_01', MAX_POOL_01, lsize=5)

    # CONVOLUTION Layer
    CONV_02     = CONVOLUTION('CONV_02', NORM_01, weights['WEI_CONV_02'], biases['BIA_CONV_02'], padding='SAME')
    # Max Pooling (down-sampling)
    MAX_POOL_02 = POOLING('MAX_POOL_02',CONV_02,k=3, s=2, padding='SAME')
    # Apply Normalization
    NORM_02     = NORMALIZATION('NORM_02', MAX_POOL_02, lsize=5)

    # CONVOLUTION Layer
    CONV_03     = CONVOLUTION('CONV_03', NORM_02, weights['WEI_CONV_03'], biases['BIA_CONV_03'], padding='SAME')

    # CONVOLUTION Layer
    CONV_04     = CONVOLUTION('CONV_04', CONV_03, weights['WEI_CONV_04'], biases['BIA_CONV_04'], padding='SAME')

    # CONVOLUTION Layer
    CONV_05     = CONVOLUTION('CONV_05', CONV_04, weights['WEI_CONV_05'], biases['BIA_CONV_05'], padding='SAME')
    # Max Pooling (down-sampling)
    MAX_POOL_05 = POOLING('MAX_POOL_05',CONV_05,k=3, s=2, padding='SAME')
    # Apply Normalization
    NORM_05     = NORMALIZATION('NORM_05', MAX_POOL_05, lsize=5)

    # Fully connected layer
    DENSE_01=tf.reshape(NORM_05,[-1,weights['WEI_DENS_01'].get_shape().as_list()[0]])
    DENSE_01=tf.add(tf.matmul(DENSE_01,weights['WEI_DENS_01']),biases['BIA_DENS_01'])
    DENSE_01=tf.nn.relu(DENSE_01)

    ## dropout ##
    DENSE_01=tf.nn.dropout(DENSE_01, dropout)

    #### 2 fc ####
    #DENSE_02=tf.reshape(DENSE_01,[-1,weights['WEI_DENS_02'].get_shape().as_list()[0]])
    DENSE_02=tf.add(tf.matmul(DENSE_01,weights['WEI_DENS_02']),biases['BIA_DENS_02'])
    DENSE_02=tf.nn.relu(DENSE_02)

    ## dropout ##
    DENSE_02=tf.nn.dropout(DENSE_02, dropout)

    #### output ####
    Pred_m = tf.add(tf.matmul(DENSE_02,weights['out_w']),biases['out_b'])
    return Pred_m

########## define model, loss and optimizer ##########

#### model ####
pred=ALEX_NET(x, weights, biases, dropout)

#### loss ####
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=Alpha_Lr).minimize(cost)

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

##################### train and evaluate model ##########################

########## initialize variables ##########
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    start_time = time.time()
    step = 1

    # 
    for episode in range(N_EPISODES+1):

        #### iteration ####
        for _ in range(mnist.train.num_examples//batch_size):

            step += 1

            ##### get x,y #####
            batch_xs, batch_ys=mnist.train.next_batch(batch_size)

            ##### optimizer ####
            sess.run(optimizer,feed_dict={x:batch_xs, y:batch_ys})

            
            ##### show loss and acc ##### 
            if step % display_step==0:
                loss,acc=sess.run([cost, accuracy],feed_dict={x: batch_xs, y: batch_ys})
                print("episode "+ str(episode) + ", Minibatch Loss=" + \
                    "{:.6f}".format(loss) + ", Training Accuracy= "+ \
                    "{:.5f}".format(acc))
                
                elapsed_time = datetime.timedelta(seconds=int(time.time()-start_time))
                print("[{}]".format(elapsed_time))

    print("Optimizer Finished!")
    
    elapsed_time = time.time() - start_time
    formatted = datetime.timedelta(seconds=int(elapsed_time))
    print("=== training time elapsed: {}s ===".format(formatted))

    ##### test accuracy #####
    for _ in range(mnist.test.num_examples//batch_size):
        batch_xs,batch_ys=mnist.test.next_batch(batch_size)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))



