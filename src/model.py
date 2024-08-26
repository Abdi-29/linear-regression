import numpy as np

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
         