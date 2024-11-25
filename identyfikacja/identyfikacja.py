import numpy as np
import matplotlib.pyplot as plt


'''
Stage 0
Simulate a noisy output signal of a system with the following parameters:
'''

samples = 1000
timeline = np.arange(samples)

theta = np.array([1.0, 1.0, 1.0]) # parameters 
u = np.random.rand(samples) # input signal

# noise from triangular distribution
noise = np.random.triangular(left=0.0, mode=0.5, right=1.0, size=samples)

# output signal
y = np.zeros(samples) 
for k in range(2, samples):
    y[k] = theta[0] * u[k] + theta[1] * u[k-1] + theta[2] * u[k-2] + noise[k]

fig, ax = plt.subplots()
ax.plot(timeline, y)
ax.set(xlabel='time', ylabel='output', title='Noisy output signal')
ax.grid(visible=True)
plt.show()