import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import os


def mock_data(n_samples = 100, n_features=1, seed=0):
    key = random.PRNGKey(seed)
    X = random.normal(key, (n_samples, n_features)) #Generates random features
    true_w = jnp.array([2.0]) #True weight
    true_b = 3.0 #True bias
    y = jnp.dot(X, true_w) + true_b + 0.1 * random.normal(key, (n_samples,)) 
    print(3* random.normal(key, (n_samples,)) )
    return X, y

class Jax_Linear_Regression():
    def __init__(self, n_features: int) -> None:
        # Initialize weights and bias
        key = jax.random.key(seed=42)
        w = jax.random.randint(key=key, shape=(n_features, 1))       
        b = jax.random.randint(key=key, shape=(n_features, 1))

    def predict(self, X):
        # Compute predictions
        y_pred = self.w * X + self.b
        return y_pred 

    def loss(self, X, y):
        # Compute loss (MSE)
        pass

    def gradients(self, X, y):
        # Compute gradients
        pass

    def update(self, X, y, alpha):
        # Update parameters w/ Gradient Descent
        pass

    def fit(self, X, y, epochs=100, alpha=0.01):
        # Train model over multiple epochs
        pass

    def score(self, X, y):
        # Evaluate model performance 
        pass

def plot_data(X, y):
    fig, ax = plt.subplots()
    ax.scatter(X, y)
    plt.show()


def main():
    X, y = mock_data()
    
    plot_data(X, y)

    print("Everything ran successfully!")


main()
