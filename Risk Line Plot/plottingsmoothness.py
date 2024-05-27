import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load your data
data = np.load('data_chunk_long.npy')
averages = np.mean(data, axis=0)[:-2] #remove outliers
x_values = np.linspace(0.2, 0.68, len(averages)) # resize for outliers

# Define the model
def exp_growth(x, A, B, C): # found best model of form y = Ae^A(x-1)
    return A * np.exp(B * (x - C))

# Define the objective function to be minimized (-R^2)
def objective(params):
    A, B, C = params
    predictions = exp_growth(x_values, A, B, C)
    ss_res = np.sum((averages - predictions) ** 2)
    ss_tot = np.sum((averages - np.mean(averages)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return -r_squared  # Minimize the negative R^2 to find the best fit

# Initial guess for parameters A, B, C
initial_guess = [10, 10, 1]

# Perform the optimization
result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=[(0, None), (0, None), (None, None)])

best_A, best_B, best_C = result.x
print("Optimized Parameters:", result.x)
print("Maximized R-squared:", -result.fun)  # Since we minimized negative R^2

# Plot the original data and the fitted model
plt.figure(figsize=(10, 6))
plt.plot(x_values, averages, 'o', label='Empirical Data')
plt.plot(x_values, exp_growth(x_values, best_A, best_B, best_C), '-', label='Fitted Exponential Model')
plt.title('Average Emperical Proportion of Injuries To Non-Injuries At Each Model Output Value\n From 500 Trained Models With Identical Hyperparameters And Random Initialisations ')
plt.xlabel('Model Output Value')
plt.ylabel('Proportions Of Injured To Non-Injured')
plt.legend()
plt.grid(True)
plt.show()
