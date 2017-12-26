import numpy as np
#numpy used for importing data

def predict(x, W, b):
    return W * x + b

alpha = 0.1
iterations = 100
data = np.genfromtxt('doc.txt', delimiter=',')
#data represents line y = 0.7x + 0.8
m = int(len(data) * 0.7) # Training data from 0 to m - 1 (70%)
n = len(data) # Testing data from m to n - 1

x = [data[i][0] for i in range(0, n)]
y = [data[i][1] for i in range(0, n)]
W = np.random.rand()
b = np.random.rand()

# Model y = W * x + b

for counter in range(iterations):
    prediction = [predict(x_i, W, b) for x_i in x]
    t1 = 0
    t2 = 0
    for i in range(0, m - 1):
        t2 += prediction[i] - y[i]
        t1 += (prediction[i] - y[i]) * x[i]
    t1 *= (alpha / m)
    t2 *= (alpha / m)
    W -= t1
    b -= t2

loss = 0
for i in range(m, n):
    loss += predict(x[i], W, b) - y[i]
print(loss / (n - m))
print("W: ", W, " b: ", b)
