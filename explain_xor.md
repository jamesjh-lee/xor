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

- ![formula](https://render.githubusercontent.com/render/math?math=\bar y = f(H*W_3 + b_3) = f(\begin{bmatrix} h_1 & h_2 \end{bmatrix} * \begin{bmatrix} w_5 \\ w_6 \end{bmatrix} + b_3) =
f(w_5*h_1 + w_6*h_2 + b_3)
