import jax
import jax.numpy as jnp
import jax.random as random
import os






def mock_data(n_samples = 100, n_features=1, seed=0):
    key = random.PRNGKey(seed)
    X = random.normal(key, (n_samples, n_features)) #Generates random features
    true_w = jnp.array([2.0]) #True weight
    true_b = 3.0 #True bias
    y = jnp.dot(X, true_w) + true_b + 0.1 * random.normal(key, (n_samples,)) 
    return X, y

def main():
    X, y = mock_data()
    print(f"Input Features: {X}")
    print(f"Output Labels: {y}")
main()
