import numpy as np
from model import LinearRegression
from utils import load_model, denormalize

def main():
    file = "data/params.npz"
    theta0, theta1, min_mileage, max_mileage, min_price, max_price = load_model(file)

    model = LinearRegression()
    model.theta0 = theta0
    model.theta1 = theta1

    while True:
        try:
            mileage_input = float(input("Enter the mileage: "))

            if mileage_input < 0:
                raise ValueError("Mileage cannot be negative.")
            
            break
        except ValueError as e:
            print(f"Invalid input: {e}. Please enter a valid number.")

    n_mileage = (mileage_input - min_mileage) / (max_mileage - min_mileage)

    n_price = model.predict(np.array([n_mileage]))[0]

    predicted_price = denormalize(n_price, min_price, max_price)

    print(f"Estimated price for mileage {mileage_input} is: ${predicted_price:.2f}")
    

if __name__ == "__main__":
    main()