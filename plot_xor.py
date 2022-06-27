import os, sys
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.setrecursionlimit(1500)
import argparse
import numpy as np
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.compat.v1.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1 import Session, ConfigProto, GPUOptions, get_default_graph, global_variables_initializer
import tensorflow as tf
#from bcolors import *
from utils import *
from sympy import *
import sympy.plotting as plt

HOME = os.path.split(os.path.abspath(__file__))[0]

def parse_args():
	desc = 'Plot the output of XOR trained by DNN'
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument('--hidden_layers', type=int, default=1, help='a number of the hidden layers, default 2')
	parser.add_argument('--units', type=int, default=2, help='the unit size of a Dense layer, default 2')
	parser.add_argument('--activation', type=str, default='sigmoid', help='the type of a activation function, default sigmoid')
	parser.add_argument('--optimizer', type=str, default='adam', help='optimizer, default Adam')
	parser.add_argument('--loss', type=str, default='mse', help='loss function, default mse')
	parser.add_argument('--metrics', type=bool, default=False, help='metrics(accuracy) to compile model')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
	parser.add_argument('--scheduler', type=bool, default=False, help='scheduler for learning rate')
	parser.add_argument('--epochs', type=int, default=1000, help='a number of epoch')
	parser.add_argument('--patience', type=int, default=50, help='number of patience')
	parser.add_argument('--save_filename', type=str, default='xor.png', help='image filename')
	return check_args(parser.parse_args())

def check_args(args):
	# check hidden layers
	try:
		assert args.hidden_layers >= 1
	except:
		print(WARNING + 'WARNING the number of hidden_layers must be greater than 0, set hidden_layers = 1' + ENDC)
		args.hidden_layers = 1
			
	# check activation
	try:
#		args.activation = args.activation.lower()
		assert args.activation in ('sigmoid', 'tanh', 'linear')
	except:
		print('activation function is among [sigmoid, tanh], set activation = sigmoid')
		args.activation = 'sigmoid'
		
	# check optimizer
	try:
		args.optimizer = args.optimizer.lower()
		assert args.optimizer in ('adam', 'sgd', 'rmsprop', 'adadelta', 'adamax', 'adagrad', 'nadam')
	except:
		args.optimizer = 'adam'
	
	# check loss
	try:
		args.loss = args.loss.lower()
		assert args.optimizer in ('mse', 'mae', 'binary_crossentropy')
	except:
		args.optimizer = 'mse'
		
	return args

def build(args):
	# input layer
	inputs = Input(shape=(2,))
	hidden = None
	for i in range(args.hidden_layers):
		if i == 0:
			hidden = Dense(args.units, activation=args.activation)(inputs)
		else:
			hidden = Dense(args.units, activation=args.activation)(hidden)
	if args.activation == 'linear':
		outputs = Dense(1, activation='linear')(hidden)
	else:
		outputs = Dense(1, activation='sigmoid')(hidden)
	return Model(inputs,outputs)

def get_optimizer(optimizer):
	if optimizer == 'sgd':
		return SGD
	return Adam

def build_output(args, weights):
	output = None
	# set input variables
	x1, x2 = symbols('x1, x2')
	
	# create symbols
	variables = {}
	ws = {}
	bs = {}
	w = 1
	b = 1
	W = []
	B = []
	for i, weight in enumerate(weights):
		layer = int(np.ceil((i+1) / 2))
		tmp_w = []
		tmp_b = []
		xx = []
		yy = []
		if len(weight.shape) >= 2:
			for c in weight:
				for d in c:
					if i % 2 == 0:
						# set weight
						tmp_w.append(d)
						ww = 'w' + str(w)
						tmp = symbols(ww)
						xx.append(tmp)
						variables[ww] = None
						w += 1
					else:
						# set bias
						tmp_b.append(d)
						bb = 'w' + str(b)
						tmp = symbols(bb)
						yy.append(tmp)
						variables[bb] = None
						b += 1
		else:
			for d in weight:
				if i % 2 == 0:
					# set weight
					tmp_w.append(d)
					ww = 'w' + str(w)
					tmp = symbols(ww)
					xx.append(tmp)
					variables[ww] = None
					w += 1
				else:
					# set bias
					tmp_b.append(d)
					bb = 'b' + str(b)
					tmp = symbols(bb)
					yy.append(tmp)
					variables[bb] = None
					b += 1
		if tmp_w:
			ws[layer] = tmp_w
		else:
			bs[layer] = tmp_b
		if xx:
			W.append(xx)
		if yy:
			B.append(yy)
	
	# set weights and bias
	bias = 1
	weight = 1
	for l in range(1, args.hidden_layers+1):
		for k in bs[l]:
			variables['b'+str(bias)] = k
			bias += 1
		tmp = []
		for c in range(args.units):
			tmp.append([x for idx, x in enumerate(ws[l]) if idx % args.units == c ])
		tmp = np.array(tmp).flatten()
		for k in tmp:
			variables['w'+str(weight)] = k
			weight += 1
	
	ll = list(bs.keys())[-1]
	for x in ws[ll]:
		variables['w'+str(weight)] = x
		weight += 1
	variables['b'+str(bias)] = bs[ll][0]
	
	
	# set formular
	s = symbols('s')
	l, a = symbols('lambda, alpha')
	f = None
	if args.activation == 'sigmoid':
		f = 1/(1+exp(-s))
	elif args.activation == 'tanh':
		f = (exp(2*s)-1)/(exp(2*s)+1)
		
	X = Matrix([[x1, x2]])
	for i, v in enumerate(zip(W,B)):
		w, b = v
		if i == 0:
			tmp = []
			for c in range(2):
				tmp.append([x for idx, x in enumerate(w) if idx % 2 == c ])
			W[i] = Matrix(tmp)
			B[i] = Matrix([b])
			l = X*W[i] + B[i]
			L = Matrix([[f.subs(s, l[x]) for x in range(args.units)]])
		elif i == args.hidden_layers:
			W[i] = Matrix(w)
			B[i] = Matrix(b)
			l = L * W[i] + B[i]
			L = f.subs(s, l[0])
		else:
			tmp = []
			for c in range(args.units):
				tmp.append([x for idx, x in enumerate(w) if idx % args.units == c ])
			W[i] = Matrix(tmp)
			B[i] = Matrix([b])
			l = L * W[i] + B[i]
			L = Matrix([[f.subs(s, l[x]) for x in range(args.units)]])
	
	output = L.subs(variables)
	print(output)
	return output

def main(args):
	# set xor dataset
	x_train = np.array([[0,0],[0,1],[1,0], [1,1]], dtype=np.float32)
	y_train = np.array([[0],[1],[1],[0]], dtype=np.float32)
	
	print('Start building a DNN model')
	
	# fitting model
	sched, early = None, None
	callbacks = []
	early = EarlyStopping(monitor='loss', patience=args.patience)
	callbacks.append(early)
	if args.scheduler:
		sched = LearningRateScheduler(scheduler, verbose=1)
		callbacks.append(sched)
	y_pred = None
	cnt = 1
	while acc(y_train, y_pred) != 1.0:
		# build model
		model = build(args)
		print(cnt, 'built a model')
#		model.summary()
		
		# compile model
		optimizer = get_optimizer(args.optimizer)
		if args.metrics:
			model.compile(optimizer=optimizer(args.learning_rate), loss=args.loss, metrics=[acc])
		else:		
			model.compile(optimizer=optimizer(args.learning_rate), loss=args.loss)
		print(cnt, 'compiled a model')
			
		print(cnt, 'start fitting a model')
		hist = model.fit(x_train, y_train, epochs=args.epochs, callbacks=callbacks, verbose=0)
		y_pred = np.where(model.predict(x_train) > 0.5, 1, 0)
		print(cnt, 'finished training, train loss:', hist.history['loss'][-1])
		cnt += 1
	
	# check output
	output = build_output(args, model.get_weights())
	
	# plot output
	x1, x2 = symbols('x1, x2')
	print('plotting output of a model')
	p = plt.plot3d(output, (x1, -2, 3), (x2, -2, 3), show=False)
	p.save(HOME+'/results/'+args.save_filename)
	p.show()


if __name__ == '__main__':
	# parse arguments
	args = parse_args()
	if args is None:
		sys.exit()
		
	# main
	print(args)
	main(args)
