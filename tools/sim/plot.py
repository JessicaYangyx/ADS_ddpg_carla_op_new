import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_path = f'/home/yanxi/data_collection/OppositeVehicleRunningRedLight_5.csv'

data = pd.read_csv(file_path, encoding='utf-8')
print(data.head())



plt.subplot(2, 3, 1)
plt.scatter(data['Frame'], data['Speed'], color='r')
plt.title("EgoSpeed")
plt.xlabel('points')
plt.ylabel('speed')
plt.grid(True, linestyle=':', alpha=0.7)
coefficients = np.polyfit(data['Frame'], data['Speed'], 5)
poly = np.poly1d(coefficients)
plt.plot(data['Frame'], poly(data['Frame']), color='blue')
# plt.tight_layout()
# plt.show()

# plt.subplot(2, 3, 2)
# plt.scatter(data['Frame'], data['acc'], color='r')
# plt.title("EgoAcc")
# plt.xlabel('points')
# plt.ylabel('acc')
# plt.grid(True, linestyle=':', alpha=0.7)
# coefficients = np.polyfit(data['Frame'], data['acc'], 12)
# poly = np.poly1d(coefficients)
# plt.plot(data['Frame'], poly(data['Frame']), color='blue')
# plt.tight_layout()
# plt.show()

plt.subplot(2, 3, 3)
plt.scatter(data['Frame'], data['Steer'], color='r')
plt.title("EgoSteer")
plt.xlabel('points')
plt.ylabel('steer')
plt.ylim(-0.02, 0.02)
plt.grid(True, linestyle=':', alpha=0.7)
coefficients = np.polyfit(data['Frame'], data['Steer'], 5)
poly = np.poly1d(coefficients)
plt.plot(data['Frame'], poly(data['Frame']), color='blue')
# plt.tight_layout()

plt.subplot(2, 3, 4)
plt.scatter(data['Frame'], data['Throttle'], color='r')
plt.title("EgoThrottle")
plt.xlabel('points')
plt.ylabel('Throttle')
plt.grid(True, linestyle=':', alpha=0.7)
coefficients = np.polyfit(data['Frame'], data['Throttle'], 5)
poly = np.poly1d(coefficients)
plt.plot(data['Frame'], poly(data['Frame']), color='blue')
# plt.tight_layout()

plt.subplot(2, 3, 5)
plt.scatter(data['Frame'], data['Brake'], color='r')
plt.title("EgoBrake")
plt.xlabel('points')
plt.ylabel('Brake')
plt.grid(True, linestyle=':', alpha=0.7)
coefficients = np.polyfit(data['Frame'], data['Brake'], 5)
poly = np.poly1d(coefficients)
plt.plot(data['Frame'], poly(data['Frame']), color='blue')
# plt.tight_layout()

plt.subplot(2, 3, 6)
relDistRaw = data['RelativeDistance'].apply(eval)
relDist = []
for dist in relDistRaw:
  relDist.append(min(dist))
plt.scatter(data['Frame'], relDist, color='r')
plt.title("Relative Distance")
plt.xlabel('points')
plt.ylabel('RelDist')
plt.grid(True, linestyle=':', alpha=0.7)
coefficients = np.polyfit(data['Frame'], relDist, 5)
poly = np.poly1d(coefficients)
plt.plot(data['Frame'], poly(data['Frame']), color='blue')
plt.tight_layout()

plt.show()
