import tensorflow as tf
import numpy as np

def xavier_init(fan_in, fan_out):
    print(fan_in, fan_out)
    high = np.sqrt(6.0/(fan_in+fan_out))
    low = -high
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

class Network:
    def __init__(self, X, y, regParam = 0.2, lrate = 0.001):
        self.train_X = np.swapaxes(X,1,3).reshape((X.shape[0]*X.shape[3]*X.shape[2],X.shape[1]))
        self.train_Y = np.swapaxes(y,1,2).reshape((y.shape[0]*y.shape[1]*y.shape[2],1))
        self.x = tf.placeholder(tf.float32, shape=(None, self.train_X.shape[1]))
        self.y = tf.placeholder(tf.float32, shape=(None, self.train_Y.shape[1]))
        self.w = tf.Variable(xavier_init(self.train_X.shape[1],self.train_Y.shape[1]), name="w1")
        self.b = tf.Variable(tf.zeros([self.train_Y.shape[1]]), dtype = tf.float32, name="b1")
        self.output = tf.nn.sigmoid(tf.add(tf.matmul(self.x, self.w),self.b))
        loss = -tf.reduce_sum(self.y * tf.log(1e-6 + self.output) + (1-self.y) * tf.log(1e-6 + 1 - self.output))
        regularizer = tf.nn.l2_loss(self.w)
        self.cost = tf.reduce_mean(loss) + regParam*regularizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = lrate).minimize(self.cost)
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.train()
    def train(self, batch_size=128*128, epochs=50):
        print("Beginning Training")
        for j in range(epochs):
            tot_cost = 0
            for i in range(int(self.train_X.shape[0]/batch_size)):
                opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.x: self.train_X[i*batch_size:(i+1)*batch_size,:], self.y: self.train_Y[i*batch_size:(i+1)*batch_size,:]})
                tot_cost+=cost
            print("Epochs: ",j, " Total Cost: ", tot_cost)
        return tot_cost
    def test(self, X):
        return self.sess.run(self.output, feed_dict={self.x: np.swapaxes(X,1,3).reshape((X.shape[0]*X.shape[3]*X.shape[2],X.shape[1]))})