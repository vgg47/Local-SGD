import numpy as np
import math

def nesterov_descent(func, L, init_x, gradient_func, epsilon=0.05):
    
    x = init_x

    lambda_prev = 0
    lambda_curr = 1
    gamma = 1
    y_prev = x
    alpha = 0.05 / (2 * L)

    gradient = gradient_func(x)

    while np.linalg.norm(gradient) >= epsilon:
        y_curr = x - alpha * gradient
        x = (1 - gamma) * y_curr + gamma * y_prev
        y_prev = y_curr

        lambda_tmp = lambda_curr
        lambda_curr = (1 + math.sqrt(1 + 4 * lambda_prev * lambda_prev)) / 2
        lambda_prev = lambda_tmp

        gamma = (1 - lambda_prev) / lambda_curr

        gradient = gradient_func(x)

    return x