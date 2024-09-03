import numpy as np
from utils import read_data, normalize_data, check_data_integer, check_data_ranges, save_model, denormalize
from model import LinearRegression, plot_result, plot_costs, mean_squared_value
from least_square_regression import LeastSquareRegression
import argparse

def main(args):
    path = 'data/data.csv'
    data = read_data(path)

    is_valid, message = check_data_integer(data)
    if not is_valid:
        print(f'data check integrity failed: {message}')

    mileage = data[:, 0]
    price = data[:, 1]

    is_valid, message = check_data_ranges(mileage, price)
    if not is_valid:
        print(f'data ranges check failed: {message}')

    normalize_mileage, min_mileage, max_mileage = normalize_data(mileage)
    normalize_price, min_price, max_price = normalize_data(price)

    model = LinearRegression(learning_rate=args.learningrate, iterations=args.iterations)

    print("training the model...")
    costs = model.train(normalize_mileage, normalize_price)
    
    save_model(model.theta0, model.theta1, min_mileage, max_mileage, min_price, max_price)

    normalized_predicted_price = model.predict(normalize_mileage)
    if args.use_raw_data:
        predicted_price = denormalize(normalized_predicted_price, min_price, max_price)
    else:
        predicted_price = normalized_predicted_price
        mileage = normalize_mileage
        price = normalize_price

    if args.cp:
        mse = mean_squared_value(normalize_price, predicted_price)
        print(f"Mean Squared Error: {denormalize(mse, min_price, max_price)}")

    if args.lsr:
        lmq_model = LeastSquareRegression()
        lmq_model.fit(normalize_mileage, normalize_price)
        te = lmq_model.predict(normalize_mileage)
        plot_result(normalize_mileage, normalize_price, te, title="Least Squares Regression")

    if args.pr:
        plot_result(mileage, price, predicted_price, title="Gradient Descent Regression")

    if args.pcc:
        plot_costs(costs)

def check_learning_rate(value):
    value = float(value)
    if value < 0.0001 or value > 1.0:
        raise argparse.ArgumentTypeError(f"Learning rate must be between 0.0001 and 1.0, but got {value}")
    return value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear Regression Model")
    
    parser.add_argument('-pr', action='store_true', help="Plot the regression line on a graph")
    parser.add_argument('-pcc', action='store_true', help="Plot the cost curve over training iterations")
    parser.add_argument('-lsr', action='store_true', help="Compare with Least Squares regression and plot")
    parser.add_argument('--use_raw_data', action='store_true', help="Use raw (non-normalized) data instead of normalized data")
    parser.add_argument('-cp', action='store_true', help="Calculate and print the precision (MSE) of the model")
    parser.add_argument('--learningrate', '-l', default=0.1, type=check_learning_rate, 
                    help="Learning rate for gradient descent (between 0.0001 and 1.0)")
    parser.add_argument('--iterations', '-i', default= 500, type=int)

    args = parser.parse_args()
    main(args)