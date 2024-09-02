import numpy as np 

class LeastSquareRegression:
    def __init__(self) -> None:
        self.theta0 = 0.0
        self.theta1 = 0.0

    def predict(self, mileage):
        return self.theta0 + self.theta1 * mileage
    
    def fit(self, x, y):
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x_mean - x) * (y_mean - y))
        denominator = np.sum((x_mean - x) ** 2)

        self.theta1 = numerator / denominator
        self.theta0 = y_mean - self.theta1 * x_mean

        print(f"Least Squares solution -> theta0: {self.theta0}, theta1: {self.theta1}")