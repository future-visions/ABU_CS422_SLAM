# We need
#   - Squence of observations z
#   - Initial value x0 (This can just be our first observation z)
#   - Initial value for p for prediction error. It can't be 0 otherwise would stay 0 by multiplications. So set it to 1.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Time
k = [i for i in range(10)]
# State estimates
x = []
# Predicted state
xPred = []
# Prediction error
p = []
# Kalman gain
g = []
# Observations
z = []

x0 = 1000
r = 200
a = 0.90
p0 = 1

x.append(x0)
p.append(p0)
g.append(0)

for i in range(9):
    x.append(x[i] * a)

for i in range(len(x)):
    z.append(x[i] + np.random.uniform(-r, r, None))

xPred.append(z[0])

for i in range(1, len(z)):
    # Predict
    xPred.append(a * xPred[i - 1])
    p.append(a * p[i-1] * a)

    # Update
    g.append(p[i - 1] / (p[i - 1] + r))
    xPred[i] = xPred[i] + g[i] * (z[i] - xPred[i])
    p[i] = (1 - g[i]) * p[i]

print(p)
print(xPred)
print(g)

dataFrame = pd.DataFrame({'x': k, 'State Estimates':x, 'Observations':z, 'Predicted State':xPred})
palette = plt.get_cmap('Set1')
plt.style.use('seaborn-darkgrid')

num = 0
for column in dataFrame.drop('x', axis=1):
    num += 1
    plt.plot(dataFrame['x'], dataFrame[column], marker='', color=palette(num), lineWidth=2, alpha=0.8, label=column)

plt.legend(loc=1, ncol=4)
plt.xlabel("Time")
plt.show()



