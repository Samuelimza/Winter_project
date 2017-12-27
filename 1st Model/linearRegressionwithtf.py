import tensorflow as tf
import numpy as np

alpha = 0.01
iterations = 100
data = np.genfromtxt('doc1.txt', delimiter=',')
#data represents line y = 0.7x + 0.8
m = int(len(data) * 0.7) # Training data from 0 to m - 1
n = len(data) # Testing data from m to n - 1

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
W = tf.Variable([np.random.rand() * 10], dtype = tf.float32)
b = tf.Variable([np.random.rand() * 10], dtype = tf.float32)
myModel = W * x + b # Model y = W * x + b
train = (tf.train.GradientDescentOptimizer(alpha)).minimize(tf.reduce_sum(tf.square(myModel - y)))

xTrainData = [int(data[i][0]) for i in range(0, m)]
yTrainData = [int(data[i][1]) for i in range(0, m)]
xTestData = [int(data[i][0]) for i in range(m, n)]
yTestData = [int(data[i][1]) for i in range(m, n)]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(0, iterations):
    sess.run(train, {x: xTrainData, y: yTrainData})

loss = tf.reduce_sum(tf.square(myModel - y))
print(sess.run(loss, {x: xTestData, y: yTestData}))
print("W, b: ", sess.run([W, b]))
