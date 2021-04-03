import matplotlib.pyplot as plt


f = open("histogram_data.txt", "r")

arr = f.read().split("\n")
for i in range(len(arr)):
    arr[i] = int(arr[i])


plt.hist(arr, bins=40)
plt.xlabel("Updates")
plt.ylabel("Count")
plt.show()

f.close()