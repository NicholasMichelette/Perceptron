from Perceptron import *
import matplotlib.pyplot as plt
import random
import numpy as np


dimensions = 10
points = 1000
pointslist = []
pointlabel = []
max = 1.0
w_init = []

w_init.append(random.uniform(-1.0, 1.0))

for i in range(dimensions):
    pointslist.append([])
    w_init.append(random.uniform(-1.0, 1.0))
    for j in range(points):
        pointslist[i].append(random.uniform(0.0, max))


#w0 + w1x1 + w2x2 + w3x3 + ... + w10x10 = 0

temp = 0
for i in range(points):
    pointsum = w_init[0]
    for j in range(dimensions):
        pointsum += w_init[j + 1] * pointslist[j][i]
    if pointsum >= 0:
        pointlabel.append(1)
        temp += 1
    else:
        pointlabel.append(0)


print(temp)

randarray = []
for i in range(points):
    randarray.append(i)

if temp >= 250 and temp <= 750:
    f = open("histogram_data.txt", "w")
    for _ in range(100):
        random.shuffle(randarray)
        temppointlabel = []
        temppointslist = []
        for i in range(dimensions):
            temppointslist.append([])
        for i in range(points):
            temppointlabel.append(pointlabel[randarray[i]])
            for j in range(dimensions):
                temppointslist[j].append(pointslist[j][randarray[i]])
        p = Perceptron(lr = 0.5, max_iterations = 10000)
        p = p.fit(temppointslist, temppointlabel, max)
        print(p.updates)
        f.write(str(p.updates) + "\n")
    f.close()