import tensorflow as tf
import numpy as np

def xavier_init(fan_in, fan_out):
    high = np.sqrt(6.0/(fan_in+fan_out))
    low = -high
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

class Network:
    def __init__(self, X, y):
        self.train_X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]*X.shape[3]))
        self.train_Y = y.reshape((y.shape[0],y.shape[1]*y.shape[2]))
        self.variables()
        self.network()
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.train()
    def variables(self):
        self.x = tf.placeholder(tf.float32, shape=(None, self.train_X.shape[1]))
        self.y = tf.placeholder(tf.int32, shape=(None, self.train_Y.shape[1]))
        self.w = tf.Variable(xavier_init(self.train_X.shape[1],self.train_Y.shape[1]), name="weight")
        self.b = tf.Variable(tf.zeros([self.train_Y.shape[1]]), dtype = tf.float32)
    def network(self, regParam = 0.1, lrate = 0.001):
        self.output = tf.add(tf.matmul(self.x, self.w),self.b)
        loss = -tf.reduce_sum(self.y * tf.log(1e-6 + self.output) + (1-self.y) * tf.log(1e-6 + 1 - self.output))
        regularizer = tf.nn.l2_loss(self.w)
        self.cost = tf.reduce_mean(loss) + regParam*regularizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = lrate).minimize(self.cost)
    def train(self, batch_size=200, epochs=100):
        for j in range(epochs):
            tot_cost = 0
            for i in range(int(train_X.size/batch_size)):
                opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.x: self.train_X[i*batch_size:(i+1)*batch_size,:], self.y: self.train_Y[i*batch_size:(i+1)*batch_size,:]})
                tot_cost+=cost
            print("Epochs: ",j, "\nTotal Cost: ", tot_cost)
        return tot_cost
    def test(self, X):
        return self.sess.run(self.output, feed_dict={self.x: X})