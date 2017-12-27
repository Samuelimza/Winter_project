import numpy as np
import matplotlib.pyplot as plt

class model2():
    alpha= 0
    iterations = 0
    data = []
    m = 0
    n = 0
    x = []
    y = []
    W = []

    def __init__(self, dataPath):
        self.alpha = 0.01
        self.iterations = 1000
        self.data = np.genfromtxt(dataPath, delimiter=',')
        self.m = int(len(self.data) * 0.7)
        # Training data from 0 to m - 1 (70%)
        self.n = len(self.data)
        # Testing data from m to n - 1 (30%)

        self.x = [self.data[i][0] for i in range(0, self.n)]
        self.y = [self.data[i][1] for i in range(0, self.n)]
        self.W = [np.random.rand() * 10 for i in range(0, 4)]

    def predict(self, x):
        return self.W[0] + self.W[1] * x + self.W[2] * (x**2) + self.W[3] * (x**3)

    def printModel(self):
        loss = 0
        for i in range(self.m, self.n):
            loss += abs(self.predict(self.x[i]) - self.y[i])
        loss /= (self.n - self.m)
        print("Loss: ", loss)
        print("W: ", self.W)

    def train(self):
        for counter in range(self.iterations):
            prediction = [self.predict(x_i) for x_i in self.x]
            t = [0 for i in range(0, 4)]
            for i in range(0, self.m - 1):
                for j in range(0, 4):
                    t[j] += (prediction[i] - self.y[i]) * (self.x[i])**j

            for i in range(0, 4):
                t[i] *= (self.alpha / self.m)
                self.W[i] -= t[i]

nonlinearModel = model2('doc1.txt')
#data represents line y = 0.7x + 0.8
nonlinearModel.printModel()
nonlinearModel.train()
nonlinearModel.printModel()

nonlinearModel = model2('doc2.txt')
#data roughly represents y = 0.5x^2 + 0.7x - 1.8
nonlinearModel.printModel()
nonlinearModel.train()
nonlinearModel.printModel()

#visualisation of model fitting the data
x = [nonlinearModel.x[i] for i in range(0, nonlinearModel.n)]
x.sort()
y = [nonlinearModel.predict(x[i]) for i in range(0, nonlinearModel.n)]
for i in range(0, nonlinearModel.n):
    plt.plot(nonlinearModel.x[i], nonlinearModel.y[i], 'bo')
plt.plot(x, y)
plt.show()
