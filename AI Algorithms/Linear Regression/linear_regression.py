import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

        self.X = self.data['Head Size(cm^3)'].values
        self.y = self.data['Brain Weight(grams)'].values
        self.mean_X = np.mean(self.X)
        self.mean_y = np.mean(self.y)
        self.total_number = len(self.X)

        self.X = (self.X - self.mean_X) / np.std(self.X)
        self.y = (self.y - self.mean_y) / np.std(self.y)

    def coefficients(self):
        numerator = 0
        denominator = 0
        for i in range(self.total_number):
            numerator += (self.X[i] - self.mean_X) * (self.y[i] - self.mean_y)
            denominator += np.square(self.X[i] - self.mean_X)

        self.slope = numerator / denominator
        self.intercept = self.mean_y - (self.slope * self.mean_X)

    def plot_regression(self):
        Y = self.slope * self.X + self.intercept

        plt.plot(self.X, Y, color='blue', label='Regression Line')
        plt.scatter(self.X, self.y, c='green', label='Scatter data')
        plt.xlabel('Head Size(cm^3)')
        plt.ylabel('Brain Weight(grams)')
        plt.legend()
        plt.show()

    def r_squared(self):
        sum_of_square_t = 0
        sum_of_square_res = 0

        for i in range(self.total_number):
            y_pred = self.slope * self.X[i] + self.intercept
            sum_of_square_t += np.square(self.y[i] - self.mean_y)
            sum_of_square_res += np.square(self.y[i] - y_pred)

        self.r_squared_value = 1 - (sum_of_square_res / sum_of_square_t)
        return self.r_squared_value

    def cost_function(self):
        cost = 0
        for i in range(self.total_number):
            y_pred = self.slope * self.X[i] + self.intercept
            cost += np.square(y_pred - self.y[i])
        
        cost /= (2 * self.total_number)
        return cost

    def gradient_descent(self, learning_rate = 0.01, epochs = 100):
        for epoch in range(epochs):
            y_pred = self.slope * self.X + self.intercept

            self.slope -= learning_rate * np.sum((y_pred - self.y) * self.X) / self.total_number
            self.intercept -= learning_rate * np.sum(y_pred - self.y) / self.total_number


###################################################################################################
# Train the Linear Regression
reg = LinearRegression('headbrain.csv')
reg.coefficients()
reg.plot_regression()
r_squared = reg.r_squared()
cost = reg.cost_function()
print("Before Gradient Descent:")
print("R-squared:", r_squared)
print("Cost:", cost)

reg.gradient_descent()
r_squared = reg.r_squared()
cost = reg.cost_function()
print("After Gradient Descent:")
print("R-squared:", r_squared)
print("Cost:", cost)
reg.plot_regression()
