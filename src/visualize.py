import matplotlib.pyplot as plt

def plot_result(mileage, price, predicted_price, title):
    plt.scatter(mileage, price, color='blue', label='Actual data')
    plt.plot(mileage, predicted_price, color='red', label='line')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_costs(costs):
    plt.plot(costs)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost Function Over Iterations")
    plt.show()