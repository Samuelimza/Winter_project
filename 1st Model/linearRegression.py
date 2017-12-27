import numpy as np
import tensorflow as tf

class model1():
    alpha = 0
    iterations = 0
    data = []
    m = 0
    n = 0
    x = []
    y = []
    W = 0
    b = 0

    def __init__(self, dataPath):
        self.alpha = 0.1
        self.iterations = 100
        self.data = np.genfromtxt(dataPath, delimiter=',')
        self.m = int(len(self.data) * 0.7)
        # Training data from 0 to m - 1 (70%)
        self.n = len(self.data)
        # Testing data from m to n - 1 (30%)

        self.x = [self.data[i][0] for i in range(0, self.n)]
        self.y = [self.data[i][1] for i in range(0, self.n)]
        self.W = np.random.rand() * 10
        self.b = np.random.rand() * 10

    def predict(self, x):
        return self.W * x + self.b

    def printModel(self):
        loss = 0
        for i in range(self.m, self.n):
            loss += self.predict(self.x[i]) - self.y[i]
        loss /= (self.n - self.m)
        print("W: %s b: %s loss: %s" %(self.W, self.b, loss))

    def train(self):
        for counter in range(self.iterations):
            prediction = [self.predict(x_i) for x_i in self.x]
            t1 = 0
            t2 = 0
            for i in range(0, self.m - 1):
                t2 += prediction[i] - self.y[i]
                t1 += (prediction[i] - self.y[i]) * self.x[i]
            t1 *= (self.alpha / self.m)
            t2 *= (self.alpha / self.m)
            self.W -= t1
            self.b -= t2

linearModel = model1('doc.txt')
#data represents line y = 0.7x + 0.8
linearModel.printModel()
linearModel.train()
linearModel.printModel()
linearModel = model1('NonLinear.txt')
#data roughly represents y = 0.5x^2 + 0.7x - 1.8
linearModel.printModel()
linearModel.train()
linearModel.printModel()
