import datetime
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('https://raw.githubusercontent.com/numenta/NAB/master/data/realTraffic/occupancy_6005.csv', parse_dates=['timestamp'], date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S"))
data = df.values
df = df.set_index('timestamp')
df.head()

arr = np.sort(df.values[:, 0])
size = int(len(arr)/2)
q1 = np.median(arr[:size])
q3 = np.median(arr[size:])
print(q1, q3)

iqr = q3 - q1
lowerbound = q1 - 1.5 * iqr
upperbound = q3 + 1.5 * iqr
anomalies = [i for i in arr if i < lowerbound or i > upperbound]
print(anomalies)

print(100 * len(anomalies) / len(arr))

plt.figure(figsize=(15,8))
plt.plot(df, marker='o', ls='')

temp = data[:, 0]
epoch = datetime.datetime.utcfromtimestamp(1441115100)
temp = [(i - epoch).total_seconds() / 100 for i in temp]
data[:, 0] = temp
model = KMeans(n_clusters=14, random_state=5)
y = model.fit_predict(data)

plt.figure(figsize=(15,8))
for i, color in enumerate(['red', 'orange', 'yellow', 'green', 'blue', 'violet', 'brown', 'black', 'gray', 'pink', 'lightgreen', 'deepskyblue', 'purple', 'deeppink']):
    plt.scatter(data[y == i, 0], data[y == i, 1], c=color)
plt.show()

center = model.cluster_centers_
print(center)

for i in range(14):
    x_coords, y_coords = data[y == i, 0], data[y == i, 1]
    distances = []
    for x0, y0 in zip(x_coords, y_coords):
        distances += [[math.sqrt((x0 - center[i, 0]) ** 2 + (y0 - center[i, 1]) ** 2), y0]]
    print(np.sort(distances, axis=0)[-8:, 1])