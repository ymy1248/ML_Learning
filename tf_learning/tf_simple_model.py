import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# hidden layer function
def add_layer(inputs, in_size, out_size, n_layer, act_func = None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('bias'):
            bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.summary.histogram(layer_name + '/bias', bias)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + bias
        if act_func is None:
            outputs = Wx_plus_b
        else:
            outputs = act_func(Wx_plus_b)
            tf.summary.histogram(layer_name + '/output', outputs)
        return outputs

# creat second order data
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise =np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# creat model structure
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name = 'x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name = 'y_input')
l1 = add_layer(xs, 1, 10, n_layer = 1, act_func = tf.nn.relu)
prediction = add_layer(l1, 10, 1, n_layer = 2, act_func = None)

# optimize trainning
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
            reduction_indices = [1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    tr_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs/', sess.graph)
init = tf.global_variables_initializer()
sess.run(init)
#
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#ax.scatter(x_data, y_data)
#plt.ion()
#plt.show()
for i in range(1000):
    sess.run(tr_step, feed_dict = {xs:x_data, ys:y_data})
    if i % 50 == 0:
        # print(sess.run(loss, feed_dict = {xs:x_data, ys:y_data}))
        result = sess.run(merged, feed_dict = {xs:x_data, ys:y_data})
        writer.add_summary(result, i)
