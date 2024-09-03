import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, epsilon=1e-6):
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.epsilon = epsilon

    def predict(self, mileage):
        return self.theta0 + self.theta1 * mileage
    
    def cost_function(self, x, y):
        m = len(y)
        prediction = self.predict(x)
        return (1 / (2 * m)) * np.sum(prediction - y) ** 2

    def train(self, x, y):
        m = len(y)
        previous_cost = float('inf')
        cost = []

        for i in range(self.iterations):
            predictions = self.predict(x)
            d_theta0 = (1 / m) * np.sum(predictions - y)
            d_theta1 = (1 / m) * np.sum((predictions - y) * x)

            self.theta0 -= self.learning_rate * d_theta0
            self.theta1 -= self.learning_rate * d_theta1

            current_cost = self.cost_function(x, y)
            if abs(current_cost - previous_cost) < self.epsilon and (d_theta0 < self.epsilon and d_theta1 < self.epsilon):
                print(f"Training done after iteration: {i + 1} with no improvement.")
                break

            cost.append(current_cost)
            previous_cost = current_cost

        print(f"final: {self.theta0}, theta1: {self.theta1}")

        return cost

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
