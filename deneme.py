import numpy as np
from kalman_filter import predict

measurements = [2, 3, 5]

x = np.zeros((2,1)) # initial state (location and velocity)
P = np.eye(2,2)*1000# initial variance
u = np.zeros((2,1)) # external motion
F = np.array([[1., 1.], [0, 1.]]) # next state function
H = np.array([[1., 0.]]) # measurement function
R = np.array([[1.]]) # measurement variance

for m in measurements:
    print(predict(x,u,m,F,P,R,H))