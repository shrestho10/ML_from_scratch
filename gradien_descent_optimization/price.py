import numpy as np

# Sample house sizes in square feet, standardized
house_sizes = np.array([[1000], [1500], [2000]])
house_sizes = (house_sizes - np.mean(house_sizes)) / np.std(house_sizes)
# Sample house prices in 1000s of dollars
house_prices = np.array([[300], [450], [600]])
# We initialize our parameters: slope (a) and intercept (b)
theta_real_estate = np.random.rand(2, 1)
# Learning rate and iterations for gradient descent, adjusted learning rate
alpha_real_estate = 0.01
iterations = 500
# Add a column of ones to the house sizes to accommodate the intercept (b)
X_b_real_estate = np.c_[np.ones((len(house_sizes), 1)), house_sizes]


def threshold(cost_history,i):
    thresh=abs(cost_history[i]-cost_history[i-1])
    return thresh

    
    
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    thresh=1000000
    i=-1
    while thresh >= 0.01 and i < iterations - 1:
        i=i+1
        prediction = np.dot(X, theta)  # Matrix multiplication between X and theta
        # Gradient update rule with correct cost function calculation
        theta = theta - (1/m) * alpha * (X.T.dot(prediction - y))
        cost_history[i] = (1/(2*m)) * np.sum(np.square(prediction - y))
        
        if i>=1:
            thresh= threshold(cost_history,i)

        
    return theta, cost_history, i

# Run gradient descent
theta_real_estate, cost_history,i = gradient_descent(X_b_real_estate, house_prices, theta_real_estate, alpha_real_estate, iterations)

print(i)
cost_history=cost_history[:i]
for i, cost in enumerate(cost_history[::10]):
    print(f'Iteration {i * 10}: Cost = {cost}')