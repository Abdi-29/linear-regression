import numpy as np

def read_data(path):
    try:
        data = np.loadtxt(path, delimiter=',', skiprows=1)
        return data
    except Exception as e:
        print(f"Error reading data: {e}")
        exit(0)

def check_data_integer(data):
    if np.isnan(data).any():
        return False, "found NaN value"
    elif not np.issubdtype(data.dtype, np.number):
        return False, "found not numeric"
    return True, "valid data"

def check_data_ranges(mileage, price):
    if mileage.any() < 0:
        return False, "mileage cannot be negative"
    elif price.any() < 0:
        return False, "price cannot be negative"
    return True, "correct ranges"

def normalize_data(data):
    min = np.min(data)
    max = np.max(data)
    return (data - min) / (max - min), min, max

def denormalize(data, min_val, max_val):
    return data * (max_val - min_val) + min_val

def load_model(file):
    data = np.load(file)
    return data['theta0'], data['theta1'], data['min_mileage'], data['max_mileage'], data['min_price'], data['max_price']

def save_model(theta0, theta1, min_mileage, max_mileage, min_price, max_price):
    file = "data/params"
    np.savez(file, theta0=theta0, theta1=theta1, min_mileage=min_mileage, max_mileage=max_mileage, min_price=min_price, max_price=max_price)