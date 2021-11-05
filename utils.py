def scheduler(epoch, lr):
	if epoch > 0 and epoch % 300 == 0:
		return lr * 0.1
	return lr

def acc(y_true, y_pred):
	tot = 4.0
	for c in (y_true == y_pred):
		if c == False:
			tot -= 1
	return tot/4.0

def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('true', 'yes', 't', 'y', '1'):
		return True
	elif v.lower() in ('false', 'no', 'f', 'n', '0'):
		return False

