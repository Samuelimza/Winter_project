import random

for _ in range(0, 50):
    x = -3 + 6 * random.random()
    y = 0.7 * x + 0.8
    print(x,",",y)

for _ in range(0, 50):
    x = -3 + 6 * random.random()
    y = 0.5 * x ** 2 + 0.7 * x - 1.8 + (-1 + 2 * random.random())
    print(x,",",y)
