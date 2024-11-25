import numpy as np
import matplotlib.pyplot as plt

'''
Simulate a noisy output signal of a system with the following parameters:
'''

samples = 1000
timeline = np.arange(samples)

theta = np.array([1.0, 1.0, 1.0]) # parameters 
u = np.random.rand(samples) # input signal

# noise from triangular distribution
noise = np.random.triangular(left=-1.0, mode=0.0, right=1.0, size=samples)

# output signal
y = np.zeros(samples) 
for k in range(2, samples):
    y[k] = theta[0] * u[k] + theta[1] * u[k-1] + theta[2] * u[k-2] + noise[k]

fig, ax = plt.subplots()
ax.plot(timeline, y, label='output signal')
ax.set(xlabel='time', ylabel='output', title='Output signal over Time')
ax.legend()
ax.grid(visible=True)
#plt.show()

'''
Identify the parameters of the system using the least squares method.
'''

theta = np.zeros(3).reshape(3, 1) # initial parameters
thetas = np.zeros((samples, 3, 1)) # list of estimated parameters
p = np.eye(3) * 100 # initial covariance matrix

# least squares method
for k in range(2, samples):
    phi_k = np.array([u[k], u[k-1], u[k-2]]).reshape(3, 1)
    p = p - (p @ phi_k @ phi_k.T @ p) / (1 + phi_k.T @ p @ phi_k)
    theta = theta + p @ phi_k * (y[k] - phi_k.T @ theta)
    thetas[k] = theta

print('Estimated parameters:')
print(f"theta0: {theta[0][0]:.2f}, theta1: {theta[1][0]:.2f}, theta2: {theta[2][0]:.2f}")

fig, ax = plt.subplots()
ax.plot(timeline, thetas[:, 0, 0], label='signal 1')
ax.plot(timeline, thetas[:, 1, 0], label='signal 2')
ax.plot(timeline, thetas[:, 2, 0], label='signal 3')
ax.set(xlabel='time', ylabel='parameters', title='Estimated signal over Time')
ax.legend()
ax.grid(visible=True)
#plt.show()

'''
Identify the parameters of the unstationary system.
'''

# time-varying parameter
theta_true = np.zeros((samples, 3))
theta_true[:, 0] = theta[0][0] + 0.3 * np.sin(0.01 * timeline) * (1 + 0.5 * np.sin(0.005 * timeline))

# output signal
y = np.zeros(samples)
for k in range(2, samples):
    y[k] = theta_true[k, 0] * u[k] + theta_true[k, 1] * u[k-1] + theta_true[k, 2] * u[k-2] + noise[k]

_lambda = 0.98 # forgetting factor
theta = np.zeros(3).reshape(3, 1) # initial parameters
thetas = np.zeros((samples, 3, 1)) # list of estimated parameters
p = np.eye(3) * 100 # initial covariance matrix

# least squares method
for k in range(2, samples):
    phi_k = np.array([u[k], u[k-1], u[k-2]]).reshape(3, 1)
    p = (1 / _lambda) * (p - (p @ phi_k @ phi_k.T @ p) / (_lambda + phi_k.T @ p @ phi_k))
    theta = theta + p @ phi_k * (y[k] - phi_k.T @ theta)
    thetas[k] = theta

fig, ax = plt.subplots()
ax.plot(timeline, theta_true[:, 0], label='true signal')
ax.plot(timeline, thetas[:, 0, 0], '--', label='estimated signal')
ax.set(xlabel='time', ylabel='parameters', title='True and Estimated signals over Time')
ax.legend()
ax.grid(visible=True)
#plt.show()

'''
Identification quality
'''

# mean squared error
mse = np.mean((theta_true[:, 0] - thetas[:, 0, 0])**2)

lambdas = np.linspace(0.9, 0.99, 100)
mse_values = []

for _lambda in lambdas:
    theta = np.zeros(3).reshape(3, 1)
    thetas = np.zeros((samples, 3, 1))
    p = np.eye(3) * 100

    for k in range(2, samples):
        phi_k = np.array([u[k], u[k-1], u[k-2]]).reshape(3, 1)
        p = (1 / _lambda) * (p - (p @ phi_k @ phi_k.T @ p) / (_lambda + phi_k.T @ p @ phi_k))
        theta = theta + p @ phi_k * (y[k] - phi_k.T @ theta)
        thetas[k] = theta

    mse = np.mean((theta_true[:, 0] - thetas[:, 0, 0])**2)
    mse_values.append(mse)

min_mse_index = np.argmin(mse_values)
min_mse_lambda = lambdas[min_mse_index]

fig, ax = plt.subplots()
ax.plot(lambdas, mse_values, label='mse')
ax.axvline(min_mse_lambda, color='r', linestyle='--', label=f'optimal lambda: {min_mse_lambda:.3f}')
ax.set(xlabel='lambda', ylabel='mse', title='Identification quality depending on Lambda')
ax.legend()
ax.grid(visible=True)
plt.show()

'''
'''
