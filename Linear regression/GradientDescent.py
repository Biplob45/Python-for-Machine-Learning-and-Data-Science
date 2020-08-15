import numpy as np
#import matplotlib.pyplot as plt 

def Gradient_descent(x,y):
  m = 0
  c = 0
  L = 0.0001  # The learning Rate
  epochs = 1000  #The number of iterations to perform gradient descent
  n = len(X) # Number of elements in X
  for i in range(epochs): 
    Y_pred = m*X + c  # The current predicted value of Y
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
  # print(m,c,i)
  # print('Independent ',X)
  # print('Dependent ',Y)
  print('predicted values ',Y_pred)


if __name__ == '__main__':
  X = np.array([1,2,4,3,5])
  Y = np.array([1,3,3,2,5])    
  Gradient_descent(X,Y)

''' 
After completing the gradient descent algorithms we found the optimal m & c respectively .
m = 0.7652159238030152 c = 0.21788387785300492 
'''
