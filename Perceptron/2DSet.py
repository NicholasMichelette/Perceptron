from Perceptron import *
import matplotlib.pyplot as plt
import random
import numpy as np


dimensions = 2
points = 1000
pointslist = []
pointlabel = []
max = 1.0


for i in range(dimensions):
    pointslist.append([])
    for j in range(points):
        pointslist[i].append(random.uniform(0.0, max))

#gets y intercept and slope for the random function f
y_intercept = random.uniform(0.0, max)
y_max_intercept = random.uniform(0.0, max)
slope = (y_max_intercept-y_intercept)/max

isundefined = False

if random.randint(0, 1) == 1:
    plt.plot([0, max], [y_intercept, y_max_intercept], 'g')
else:
    plt.plot([y_intercept, y_max_intercept], [0, max], 'g')
    if slope == 0:
        isundefined = True
    else:
        slope = max/(y_max_intercept-y_intercept)
        y_intercept = -slope * y_intercept

if isundefined:
    for i in range(points):
        if pointslist[0][i] < y_intercept:
            plt.plot(pointslist[0][i], pointslist[1][i], 'ro')
            pointlabel.append(0)
        else:
            plt.plot(pointslist[0][i], pointslist[1][i], 'bo')
            pointlabel.append(1)
else:
    for i in range(points):
        if pointslist[1][i] < slope * pointslist[0][i] + y_intercept:
            plt.plot(pointslist[0][i], pointslist[1][i], 'ro')
            pointlabel.append(0)
        else:
            plt.plot(pointslist[0][i], pointslist[1][i], 'bo')
            pointlabel.append(1)



p = Perceptron(lr = 0.5, max_iterations = 10000)
p = p.fit(pointslist, pointlabel, max)

slope2 = -(p.bias/p.weights[1])/(p.bias/p.weights[0])
temp3 = -(p.bias/p.weights[1])

#print(str(slope) + " " + str(y_intercept))
#print(str(slope2) + " " + str(temp3))

print(p.updates)

y2 = slope2 * max + temp3

plt.plot([0, max], [temp3, y2])
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis([0, max, 0, max])
plt.show()
