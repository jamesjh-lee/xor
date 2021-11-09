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

- <img src="https://render.githubusercontent.com/render/math?math=%5Cbar%20y%20%3D%20f(H*W_3%20%2B%20b_3)%20%3D%20f(%5Cbegin%7Bbmatrix%7D%20h_1%20%26%20h_2%20%5Cend%7Bbmatrix%7D%20*%20%5Cbegin%7Bbmatrix%7D%20w_5%20%5C%5C%20w_6%20%5Cend%7Bbmatrix%7D%20%2B%20b_3)%20%3D%0Af(w_5*h_1%20%2B%20w_6*h_2%20%2B%20b_3)%3Df(w_5*f(w_1*x_1%20%2B%20w_3*x_2%20%2B%20b_1)%2Bw_6*f(w_2*x_1%20%2B%20w_4*x_2%20%2B%20b_2)%20%2B%20b_3)">

## Assign parameters
When  <img src="https://render.githubusercontent.com/render/math?math=f%20%3D%20%5Csigmoid">, 
      <img src="https://render.githubusercontent.com/render/math?math=w_1%2C%20w_3%20%3D%205%2C%5C%20b1%3D-8">, 
      <img src="https://render.githubusercontent.com/render/math?math=w_2%2C%20w_4%20%3D%20-7%2C%5C%20b2%3D3">, 
      <img src="https://render.githubusercontent.com/render/math?math=w_5%2C%20w_6%20%3D%20-11%2C%5C%20b2%3D6">
<img src="https://render.githubusercontent.com/render/math?math=%5Cbar%20y%20%3D%20%5Csigmoid(-11*%5Csigmoid(5x_1%2B5x_2-8)%20-11*%5Csigmoid(-7x_1-7x_2%2B3)%20%2B%206)%20%20">

- <img width="174" alt="image" src="https://user-images.githubusercontent.com/93747285/140844231-492f27d3-ad8f-448d-98bf-8e9caf0616b4.png">
<img width="1259" alt="image" src="https://user-images.githubusercontent.com/93747285/140844252-7e1e83aa-fa11-4ed7-a4ce-2e795f49b5a9.png">

- <img src="https://render.githubusercontent.com/render/math?math=x_1%3D0%2C%5C%20x_2%3D0%20%5CRightarrow%20%5Csigmoid(-11*%5Csigmoid(5*0%2B5*0-8)%20-11*%5Csigmoid(-7*0-7*0%2B3)%20%2B%206)">
<img src="https://render.githubusercontent.com/render/math?math=%3D%20%5Csigmoid(-11*%5Csigmoid(-8)%20-%2011*%5Csigmoid(3)%2B6)">
<img src="https://render.githubusercontent.com/render/math?math=%5Capprox%20%5Csigmoid(-11*0%20-%2011*1%2B6)">
<img src="https://render.githubusercontent.com/render/math?math=%3D%20%5Csigmoid(-5)%20%5Capprox%200%2C">
<img src="https://render.githubusercontent.com/render/math?math=L_%7B11%7D%3D%5Csigmoid(-8)%2C%5C%20L_%7B12%7D%3D%5Csigmoid(3)">

- <img src="https://render.githubusercontent.com/render/math?math=x_1%3D0%2C%5C%20x_2%3D1%20%5CRightarrow%20%5Csigmoid(-11*%5Csigmoid(5*0%2B5*1-8)%20-11*%5Csigmoid(-7*0-7*1%2B3)%20%2B%206)">
<img src="https://render.githubusercontent.com/render/math?math=%3D%20%5Csigmoid(-11*%5Csigmoid(-3)%20-%2011*%5Csigmoid(-4)%2B6)">
<img src="https://render.githubusercontent.com/render/math?math=%5Capprox%20%5Csigmoid(-11*0%20-%2011*0%2B6)"><img src="https://render.githubusercontent.com/render/math?math=%3D%20%5Csigmoid(6)%20%5Capprox%201%2C">
<img src="https://render.githubusercontent.com/render/math?math=L_%7B11%7D%3D%5Csigmoid(-3)%2C%5C%20L_%7B12%7D%3D%5Csigmoid(-4)">

- <img src="https://render.githubusercontent.com/render/math?math=x_1%3D1%2C%5C%20x_2%3D0%20%5CRightarrow%20%5Csigmoid(-11*%5Csigmoid(5*1%2B5*0-8)%20-11*%5Csigmoid(-7*1-7*0%2B3)%20%2B%206)%0A">
<img src="https://render.githubusercontent.com/render/math?math=%3D%20%5Csigmoid(-11*%5Csigmoid(-3)%20-%2011*%5Csigmoid(-4)%2B6)">
<img src="https://render.githubusercontent.com/render/math?math=%5Capprox%20%5Csigmoid(-11*0%20-%2011*0%2B6)"><img src="https://render.githubusercontent.com/render/math?math=%3D%20%5Csigmoid(6)%20%5Capprox%201%2C">
<img src="https://render.githubusercontent.com/render/math?math=L_%7B11%7D%3D%5Csigmoid(-3)%2C%5C%20L_%7B12%7D%3D%5Csigmoid(-4)">

- <img src="https://render.githubusercontent.com/render/math?math=x_1%3D1%2C%5C%20x_2%3D1%20%5CRightarrow%20%5Csigmoid(-11*%5Csigmoid(5*1%2B5*1-8)%20-11*%5Csigmoid(-7*1-7*1%2B3)%20%2B%206)">
<img src="https://render.githubusercontent.com/render/math?math=%3D%20%5Csigmoid(-11*%5Csigmoid(2)%20-%2011*%5Csigmoid(-11)%2B6)">
<img src="https://render.githubusercontent.com/render/math?math=%5Capprox%20%5Csigmoid(-11*1%20-%2011*0%2B6)"><img src="https://render.githubusercontent.com/render/math?math=%3D%20%5Csigmoid(-5)%20%5Capprox%200%2C">
<img src="https://render.githubusercontent.com/render/math?math=L_%7B11%7D%3D%5Csigmoid(2)%2C%5C%20L_%7B12%7D%3D%5Csigmoid(-11)">

| <img src="https://render.githubusercontent.com/render/math?math=x_1"> | <img src="https://render.githubusercontent.com/render/math?math=x_2"> | <img src="https://render.githubusercontent.com/render/math?math=L_{11}"> | <img src="https://render.githubusercontent.com/render/math?math=L_{12}"> | <img src="https://render.githubusercontent.com/render/math?math=\bar y"> | XOR |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 0 | 0 | 0 | 1 | 0 | 0 |
| 0 | 1 | 0 | 0 | 1 | 1 |
| 1 | 0 | 0 | 0 | 1 | 1 |
| 1 | 1 | 1 | 0 | 0 | 0 |

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


