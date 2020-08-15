# Find cost function 

import numpy as np

def cost_function(X,Y):
    weight = 0
    bias = 0
    companies = len(X)
    total_error = 0.0
    for i in range(companies):
        total_error +=(Y[i] - (weight*X[i] + bias))**2
    return total_error / companies , total_error

p = np.array([1,2,4,3,5])
q = np.array([1,3,3,2,5]) 
print(cost_function(p,q))

# Error = 48.0 
