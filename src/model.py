import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000) -> None:
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.learning_rate = learning_rate
        self.iterations = iterations

    def predict(self, mileage):
        return self.theta0 + self.theta1 * mileage
    def cost_function(self, x, y):
        m = len(y)
        prediction = self.predict(x)
        return (1 / (2 * m)) * np.sum(prediction - y) ** 2

    def train(self, x, y):
        m = len(y)
        for _ in range(self.iterations):
            predictions = self.predict(x)
            d_theta0 = (1 / m) * np.sum(predictions - y)
            d_theta1 = (1 / m) * np.sum((predictions - y) * x)
            self.theta0 -= self.learning_rate * d_theta0
            self.theta1 -= self.learning_rate * d_theta1

        print(f"final: {self.theta0}, theta1: {self.theta1}")

"""
    he Calculate the squared differences between the actual and predicted values:
    
    Actual Values: [15, 25, 35, 45, 55]

    Predicted Values: [18, 22, 38, 42, 52]

    Squared Differences: [(15-18)2, (25-22)2, (35-38)2, (45-42)2, (55-52)2]

    = [9, 9, 9, 9, 9]

    Compute the MSE

    MSE = (9 + 9 + 9 + 9 + 9) / 5

    = 45 / 5

    = 9

    Calculate the RMSE:

    RMSE = √(9)

    = 3
"""
def mean_squared_value(actual, predicted):
    return np.mean((actual - predicted) ** 2)

def r_squared(actual, predicted):
    ss_total = np.sum((actual - np.mean(actual)))
    ss_result = np.sum((actual - predicted) ** 2)
    return 1 - (ss_result / ss_total)

def plot_result(mileage, price, predicted_price):
    plt.scatter(mileage, price, color='blue', label='Actual data')
    plt.plot(mileage, predicted_price, color='red', label='line')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Mileage vs price')
    plt.legend()
    plt.show()