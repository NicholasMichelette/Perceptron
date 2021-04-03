import matplotlib.pyplot as plt
import random
import numpy as np


class Perceptron(object):

    def __init__(self, lr = 0.1, max_iterations = 100):
        self.lr = lr
        self.max_iterations = max_iterations
        self.updates = 0
        self.weights = []
        self.bias = 0
        self.errors = []

    def fit(self, points, labels, max_data_size):
        for _ in range(len(points)):
            self.weights.append(random.uniform(-1.0/max_data_size, 1.0/max_data_size))
        self.bias = random.uniform(-1.0 * max_data_size, 1.0 * max_data_size)

        for _ in range(self.max_iterations):
            errors = 0
            for i in range(len(labels)):
                point = []
                for j in range(len(points)):
                    point.append(points[j][i])
                update = self.lr * (labels[i] - self.predict(point))
                if update != 0.0:
                    for j in range(len(self.weights)):
                        self.weights[j] = self.weights[j] + update * point[j]/max_data_size
                    self.bias += update * max_data_size
                    errors += 1
                    self.updates += len(self.weights) + 1
            if errors == 0:
                break
            self.errors.append(errors)
            
        return self

    def predict(self, P):
        return np.where(np.dot(P, self.weights) + self.bias >= 0.0, 1, 0)

    #currently not used
    def geterrors(self, points, labels):
        errors = 0
        for i in range(len(labels)):
            point = []
            for j in range(len(points)):
                point.append(points[j][i])
            if labels[i] != self.predict(point):
                errors += 1
        return errors

