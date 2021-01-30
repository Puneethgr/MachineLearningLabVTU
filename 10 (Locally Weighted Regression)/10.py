# Program 10 (Locally Weighted Regression)
import numpy as np
import matplotlib.pyplot as plt

def radial_kernel(x0, x, tau):
    return np.exp(np.sum((x - x0) ** 2, axis=1) / (-2 * tau ** 2)) # Formula 1 (refer picture)

def local_regression(x0, x, y, tau):
    x0 = np.r_[1, x0] # r = row
    x = np.c_[np.ones(len(x)), x] # c = column
    xw = x.T * radial_kernel(x0, x, tau)
    beta = np.linalg.pinv(xw @ x) @ xw @ y 
    return x0 @ beta

def plot_lr(tau):
    x = np.linspace(-5, 5, 1000)
    y = np.log(np.abs((x ** 2) - 1) + 0.5) # Formula 2 (refer picture)
    x += np.random.normal(scale=0.05, size=1000)  #Add noice

    domain = np.linspace(-5, 5, 1000)
    predictions = [local_regression(x0, x, y, tau) for x0 in domain]
    
    plt.scatter(x, y, alpha=0.3)
    plt.plot(domain, predictions, color="red")
    return plt

print("Tau = ", 0.03)
plot_lr(0.03).show()

print("Tau = ", 5)
plot_lr(5).show()