import jax.numpy as jnp
import jax
import os
	
def test1():
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


def jitTutorial():
    import time
    
    def slow_function(x):
        return jnp.exp(x) + jnp.cos(x)
    
    # JIT-compiled version of the 'slow_function'
    fast_function = jax.jit(slow_function)

    # Input 
    x = jnp.linspace(-10, 10, 100000000) # Large input for Benchmarking
     
    # Benchamrking w/out jit
    start = time.time()
    y_slow = slow_function(x).block_until_ready()
    end = time.time()
    noJit = end-start
    print(f"Without jit: {noJit}")

    # Benchmarkng w/ jit
    start = time.time()
    y_fast = fast_function(x).block_until_ready()
    end = time.time()
    wJit = end-start
    print(f"With jit: {wJit}")
    
    print(f"The function ran {noJit/wJit} times faster w/ jit")
    

def main():
    jitTutorial()
    
main()















