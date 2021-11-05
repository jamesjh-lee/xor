# XOR
visualization for output of XOR model using DNN 

## requirement
- tensorflow 2.x
- sympy
- bcolors

## usage
```shell
#> python3 plot_xor.py
```
Arguments
| argument name   | Description                                         |
| :----:          | :---                                                |
| --hidden_layers | a number of the hidden layers, default 2            |
| --units         | the unit size of a Dense layer, default 2           |
| --activation    | the type of a activation function, default sigmoid  |
| --optimizer     | optimizer, default Adam                             |
| --loss          | loss function, default mse                          |
| --metrics       | metrics(accuracy) to compile model                  |
| --learning_rate | learning rate                                       |
| --scheduler     | the scheduler for learning rate                     |
| --epochs        | a number of epoch                                   |
| --patience      | a number of patience                                |
| --save_filename | image filename to save                              |
