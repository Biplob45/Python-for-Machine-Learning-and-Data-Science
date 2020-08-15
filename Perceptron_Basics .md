
# Perceptron


```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

Our example data, weights $w$, bias $b$, and input $x$ are defined as:


```python
w = np.array([0.2, 0.3, 0.8])
b = 0.5
x = np.array([0.5, 0.6, 0.1])
```

Our neural unit would compute $z$ as the dot-product $w \cdot x$ and add the bias $b$ to it. The sigmoid function defined above will convert this $z$ value to the activation value $a$ of the unit:


```python
z = w.dot(x) + b
print("z:", z)
print("a:", sigmoid(z))
```

    z: 0.8600000000000001
    a: 0.7026606543447316


# The XOR Problem

The power of neural units comes from combining them into larger networks. Minsky and Papert (1969): A single neural unit cannot compute the simple logical function XOR.

The task is to implement a simple perceptron to compute logical operations like AND, OR, and XOR.

Input: $x_1$ and $x_2$
Bias: $b = -1$ for AND; $b = 0$ for OR
Weights: $w = [1, 1]$
with the following activation function:

$$
y = \begin{cases}
    \ 0 &amp; \quad \text{if } w \cdot x + b \leq 0\\
    \ 1 &amp; \quad \text{if } w \cdot x + b &gt; 0
  \end{cases}
$$

We can define this activation function in Python as:


```python
def activation(z):
    if z > 0:
        return 1
    return 0
```

For AND we could implement a perceptron as:


```python

w = np.array([1, 1])
b = -1
x = np.array([0, 0])
print("0 AND 0:", activation(w.dot(x) + b))
x = np.array([1, 0])
print("1 AND 0:", activation(w.dot(x) + b))
x = np.array([0, 1])
print("0 AND 1:", activation(w.dot(x) + b))
x = np.array([1, 1])
print("1 AND 1:", activation(w.dot(x) + b))
```

    0 AND 0: 0
    1 AND 0: 0
    0 AND 1: 0
    1 AND 1: 1


For OR we could implement a perceptron as:


```python
w = np.array([1, 1])
b = 0
x = np.array([0, 0])
print("0 OR 0:", activation(w.dot(x) + b))
x = np.array([1, 0])
print("1 OR 0:", activation(w.dot(x) + b))
x = np.array([0, 1])
print("0 OR 1:", activation(w.dot(x) + b))
x = np.array([1, 1])
print("1 OR 1:", activation(w.dot(x) + b))
```

    0 OR 0: 0
    1 OR 0: 1
    0 OR 1: 1
    1 OR 1: 1


There is no way to implement a perceptron for XOR this way
