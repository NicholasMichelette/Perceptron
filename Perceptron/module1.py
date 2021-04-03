import matplotlib.pyplot as plt
import random

def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0
 
# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			sum_error += error**2
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return weights
 
# Calculate weights
dataset = [[20.7810836,20.550537003,0],
	[10.465489372,20.362125076,0],
	[30.396561688,40.400293529,0],
	[10.38807019,10.850220317,0],
	[30.06407232,30.005305973,0],
	[70.627531214,20.759262235,1],
	[50.332441248,20.088626775,1],
	[60.922596716,10.77106367,1],
	[80.675418651,-10.242068655,1],
	[70.673756466,30.508563011,1]]

ds = []
for i in range(20):
	ds.append([random.uniform(0.0, 10.0), random.uniform(0.0, 10.0), 0])
	if ds[i][1] > 5.0:
		ds[i][2] = 1
l_rate = 0.01
n_epoch = 1000
weights = train_weights(ds, l_rate, n_epoch)
print(weights)

for i in range(len(ds)):
	if ds[i][2] == 0:
		plt.plot(ds[i][0], ds[i][1], 'ro')
	else:
		plt.plot(ds[i][0], ds[i][1], 'bo')

slope2 = -(weights[0]/weights[2])/(weights[0]/weights[1])
temp3 = -10*(weights[0]/weights[2])

max = 10

y2 = slope2 * max + temp3

plt.plot([0, max], [temp3, y2])
plt.show()
