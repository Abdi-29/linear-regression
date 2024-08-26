import numpy as np
from utils import read_data, normalize_data, check_data_integer, check_data_ranges
from model import LinearRegression

def main():
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

    mileage, min_mileage, max_mileage = normalize_data(mileage)
    price, min_price, max_price = normalize_data(price)
    learning_rate = 0.01
    iterations = 1000

    model = LinearRegression(learning_rate, iterations)

    print("training the model...")
    model.train(mileage, price)

if __name__ == "__main__":
    main()                                  