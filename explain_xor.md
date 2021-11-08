# XOR
XOR(Exclusive OR) gate is a digital logic gate that gives a true output when the number of true inputs is odd.

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

<img width="1331" alt="image" src="https://user-images.githubusercontent.com/93747285/140689160-3a1a99e2-77da-4569-af25-22215edefcb5.png">

- <img src="https://render.githubusercontent.com/render/math?math=%5Cbar%20y%20%3D%20f(H*W_3%20%2B%20b_3)%20%3D%20f(%5Cbegin%7Bbmatrix%7D%20h_1%20%26%20h_2%20%5Cend%7Bbmatrix%7D%20*%20%5Cbegin%7Bbmatrix%7D%20w_5%20%5C%5C%20w_6%20%5Cend%7Bbmatrix%7D%20%2B%20b_3)%20%3D%0Af(w_5*h_1%20%2B%20w_6*h_2%20%2B%20b_3)">
- 
