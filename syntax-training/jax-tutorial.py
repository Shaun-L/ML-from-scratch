import jax.numpy as jnp
import jax
import os

def f(x):
    return jnp.sin(x) + x**3 

x = jnp.array(jnp.pi)
print(f"this is a jnp.array {x}")
y = f(x)

print(f"f(2) = {y}")

# Computes the Derivative
dfdx = jax.grad(f)
print(f"df/dx at x=2: {dfdx(x)}")

# Computes the second derivative
d2fdx2 = jax.grad(dfdx)
print(f"Second derivative at x=2: {d2fdx2(x)}")


















