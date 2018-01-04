#ACCURACY of 94.42%
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

xtrain = np.vstack([img.reshape(-1,) for img in mnist.train.images])
ytrain = mnist.train.labels

xtest = np.vstack([img.reshape(-1,) for img in mnist.test.images])
ytest = mnist.test.labels

def nonLin(x, deriv = False):
    x = np.clip(x, -50, 50)
    if(deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def printAcc():
    l0 = xtest
    l1 = nonLin(l0.dot(w0))
    l2 = nonLin(l1.dot(w1))
    l3 = nonLin(l2.dot(w2))

    sumnum = 0
    for i in range(0, 10000):
        maxi = 0
        index1 = None
        for j in range(10):
            if(l3[i][j] > maxi):
                maxi = l3[i][j]
                index1 = j
        index2 = None
        for j in range(10):
            if(ytest[i][j] == 1):
                index2 = j
        if(index1 == index2):
            sumnum += 1
    print(sumnum / 100)

np.random.seed(1)

#wieght matrices

#when initialising first time random wieghts are generated
#w0 = 2*np.random.random((784, 16)) - 1
#w1 = 2*np.random.random((16, 16)) - 1
#w2 = 2*np.random.random((16, 10)) - 1

#after saving atleast once, can load from previous save
w0 = np.genfromtxt("w0.txt", delimiter = ',')
w1 = np.genfromtxt("w1.txt", delimiter = ',')
w2 = np.genfromtxt("w2.txt", delimiter = ',')

iterations = 1000
alpha = 0.00004
#training
for j in range(iterations):
    #forward propagation
    l0 = xtrain
    print("Iteration ", j)
    l1 = nonLin(l0.dot(w0))
    l2 = nonLin(l1.dot(w1))
    l3 = nonLin(l2.dot(w2))

    #delta evaluation
    l3Er = ytrain - l3
    if(j % 20) == 0:
    #    printAcc()
        print("Error:" + str(np.mean(np.abs(l3Er))))
    l3Delta = l3Er
    l2Error = l3Delta.dot(w2.T)
    l2Delta = l2Error * nonLin(l2, deriv = True)
    l1Error = l2Delta.dot(w1.T)
    l1Delta = l1Error * nonLin(l1, deriv = True)

    #wieght update step
    w2 += (l2.T.dot(l3Delta)) * alpha
    w1 += (l1.T.dot(l2Delta)) * alpha
    w0 += (l0.T.dot(l1Delta)) * alpha

#saving of weights of 1st layer
w0file = open("w0.txt", "w")
w0file.close()
w0file = open("w0.txt", "a")
for w in range(len(w0)):
    stri = ''
    for ww in range(len(w0[w])):
        if(ww != len(w0[w]) - 1):
            stri += str(w0[w][ww]) + ','
        else:
            stri += str(w0[w][ww]) + '\n'
    w0file.write(stri)
w0file.close()
#saving of weights of 2nd layer
w1file = open("w1.txt", "w")
w1file.close()
w1file = open("w1.txt", "a")
for w in range(len(w1)):
    stri = ''
    for ww in range(len(w1[w])):
        if(ww != len(w1[w]) - 1):
            stri += str(w1[w][ww]) + ','
        else:
            stri += str(w1[w][ww]) + '\n'
    w1file.write(stri)
w1file.close()
#saving of weights of 3rd layer
w2file = open("w2.txt", "w")
w2file.close()
w2file = open("w2.txt", "a")
for w in range(len(w2)):
    stri = ''
    for ww in range(len(w2[w])):
        if(ww != len(w2[w]) - 1):
            stri += str(w2[w][ww]) + ','
        else:
            stri += str(w2[w][ww]) + '\n'
    w2file.write(stri)
w2file.close()

#testing step
print("Testing")
printAcc()
