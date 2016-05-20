import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0] 
def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


data = np.genfromtxt('iris.data', delimiter=",")
np.random.shuffle(data)
x_data = data[:,0:4].astype('f4')
y_data = one_hot(data[:,4].astype(int), 3)

print y_data

print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x = tf.placeholder("float", [None, 4])
y_ = tf.placeholder("float", [None, 3])


W1 = tf.Variable(np.float32(np.random.rand(4, 5))*0.1)
b1 = tf.Variable(np.float32(np.random.rand(5))*0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3))*0.1)
b2 = tf.Variable(np.float32(np.random.rand(3))*0.1)

y_h = tf.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.softmax(tf.matmul(y_h, W2) + b2)

#cross_entropy = tf.reduce_sum(tf.square(y_ - y))
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
errorVector = []
iterVector = []
for step in xrange(1000):
    for jj in xrange(len(x_data) / batch_size):
        batch_xs = x_data[jj*batch_size : jj*batch_size+batch_size]
        batch_ys = y_data[jj*batch_size : jj*batch_size+batch_size]

        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
        if step % 50 == 0:
	    error = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
	    errorVector.append(error)
            print "Iteration #:", step, "Error: ", error
            print sess.run(y, feed_dict={x: batch_xs})
            print batch_ys
            print "----------------------------------------------------------------------------------"

plt.plot(errorVector)
plt.axis([0, 150, 0, 25])
plt.ylabel('Error')
plt.show()

