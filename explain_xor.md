
# XOR

XOR(Exclusive OR) gate is a digital logic gate that gives a true output when the number of true inputs is odd. (wikipedia)

XOR gate is working as below:
| A | B | A XOR B |
| :-: | :-: | :-: |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

# Nonlinear model for XOR
Basically to solve XOR problem, it is essential that nonlinaerity and many of layers are core in a DNN model.

Let's assume that a DNN model has two layer and nonlinear activation function.
The DNN model is as below the figure.

## DNN Model
(reference: https://hunkim.github.io/ml/lec9.pdf)

DNN model has two layers(hidden layer and output layer)

<img width="1633" alt="image" src="https://user-images.githubusercontent.com/93747285/140840664-b8010a80-70a9-4ee8-abd5-d81bfd1b7e79.png">

- $\bar y = f(H*W + b_3) = f( \begin{bmatrix}L_{11} & L_{12} \end{bmatrix} * \begin{bmatrix} w_5 \\\\ w_6 \end{bmatrix} ) = f( w_5 * f(h_1) + w_6 * f(h_2) + b_3 )$ \
$\ \ \=f(w_5 * f(w_1 * x_1 + w_3 * x_2 + b_1) + w_6 * f(w_2 * x_1 + w_4 * x_2 + b_2) + b_3 )$ 

## Assign parameters
When $f = sigmoid,\ w_1,\ w_3=5,\ b_1=-8,\ w_2,\ w_4=-7,\ b_2=3,\ b_3=6 $ \
$\bar y = sigmoid(-11 * sigmoid(5x_1+5x_2-8)-11*sigmoid(-7x_1-7x_2+3)+6)$

- $x_1=0,\ x_2=0$ \
$ sigmoid( -11 * sigmoid(5 \times 0 + 5 \times 0 - 8)- 11 * sigmoid(-7 \times 0 - 7 \times 0 + 3) + 6 )$ \
$sigmoid(-11 * sigmoid(-8)- 11 * sigmoid(3)+6) \approx sigmoid(-11 \times 0 - 11 \times 1 + 6) = sigmoid(-5) \approx 0$ \
$*\ L_{11} = sigmoid(-8),\ L_{12}=sigmoid(3)$

- $x_1=0,\ x_2=1$ \
$ sigmoid( -11 * sigmoid(5 \times 0 + 5 \times 1 - 8)- 11 * sigmoid(-7 \times 0 - 7 \times 1 + 3) + 6 )$ \
$sigmoid(-11 * sigmoid(-3)- 11 * sigmoid(-4) + 6) \approx sigmoid(-11 \times 0 - 11 \times 0 + 6) = sigmoid(6) \approx 1$ \
$*\ L_{11} = sigmoid(-3),\ L_{12}=sigmoid(-4)$

- $x_1=1,\ x_2=0$ \
$ sigmoid( -11 * sigmoid(5 \times 1 + 5 \times 0 - 8)- 11 * sigmoid(-7 \times 1 - 7 \times 0 + 3) + 6 )$ \
$sigmoid(-11 * sigmoid(-3)- 11 * sigmoid(-4) + 6) \approx sigmoid(-11 \times 0 - 11 \times 0 + 6) = sigmoid(6) \approx 1$ \
$*\ L_{11} = sigmoid(-3),\ L_{12}=sigmoid(-4)$

- $x_1=1,\ x_2=1$ \
$ sigmoid( -11 * sigmoid(5 \times 1 + 5 \times 1 - 8)- 11 * sigmoid(-7 \times 1 - 7 \times 1 + 3) + 6 )$ \
$sigmoid(-11 * sigmoid(2)- 11 * sigmoid(-11) + 6) \approx sigmoid(-11 \times 1 - 11 \times 0 + 6) = sigmoid(-5) \approx 0$ \
$*\ L_{11} = sigmoid(2),\ L_{12}=sigmoid(-11)$

| $x_{1}$ | $x_{2}$ | $L_{11}$ | $L_{12}$ | $\bar y$ | XOR |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 0 | 0 | 0 | 1 | $sigmoid(-5) \approx 0$ | 0 |
| 0 | 1 | 0 | 0 | $sigmoid(6) \approx 1$ | 1 |
| 1 | 0 | 0 | 0 | $sigmoid(6) \approx 1$ | 1 |
| 1 | 1 | 1 | 0 | $sigmoid(-5) \approx 0$ | 0 |

## Plot a output
### sigmoid function
```python
from sympy import *
from sympy.plotting import plot
x = symbols('x')
sigmoid = symbols('sigmoid', cls=Function)
sigmoid = 1 / (1+exp(-x))
plot(sigmoid, (x, -5, 5))
```
![image](https://user-images.githubusercontent.com/93747285/140710128-13b7b2c0-3fdf-4798-a672-d215e6d6a8f2.png)

### output of the DNN model
```python
from sympy import *
from sympy.plotting import plot3d
x1, x2, t = symbols('x1 x2 t')
sigmoid = symbols('sigmoid', cls=Function)
sigmoid = 1 / (1+exp(-t))
output = sigmoid.subs(t, -11*sigmoid.subs(t, 5*x1+5*x2-8)-11*sigmoid.subs(t, -7*x1-7*x2+3)+6)
plot3d(output, (x1, -2, 3), (x2, -2, 3))
```
![image](https://user-images.githubusercontent.com/93747285/140711805-33df7fe8-e976-4127-84da-e1cfda231b07.png)


