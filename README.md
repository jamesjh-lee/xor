# XOR
plotting a output of XOR model using DNN

## requirements (python packages)
- tensorflow 2.x
- sympy
- bcolors

## usage
python 3.5+
```shell
python3 plot_xor.py --hidden_layers 3 --units 2 --activation tanh --optimizer adam \
--loss mae --metrics y --learning_rate 0.0001 --scheduler y --epochs 500 --patience 30 \
--save_filename xor_1.png 
```
Warn: recommend units <= 128, hidden_layers <= 12 because it's required much of computing resource

Arguments
| argument name   |  data type   | Description                                         |
| :---            | :----:       | :---                                                |
| --hidden_layers | integer      | a number of the hidden layers, default 2            |
| --units         | integer      | the unit size of a Dense layer, default 2           |
| --activation    | string       | the type of a activation function, default sigmoid  |
| --optimizer     | string       | optimizer, default Adam                             |
| --loss          | string       | loss function, default mse                          |
| --metrics       | boolean      | metrics(accuracy) to compile model, default false   |
| --learning_rate | float        | learning rate, default 0.001                        |
| --scheduler     | boolean      | the scheduler for learning rate, default false      |
| --epochs        | integer      | a number of epoch, default 1000                     |
| --patience      | integer      | a number of patience, default 50                    |
| --save_filename | string       | image filename to save, default xor.png             |

## Example 
![GitHub Logo](/results/xor.png)
