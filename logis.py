import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def hypothesis(X,theta):
	y_ = np.dot(X,theta)
	y_ =  seg(y_)
	return y_

def seg(z):
	return 1/(1+np.exp(-z))

def gradient(X,Y,theta):
	grad = np.zeros((X.shape[1],1))
	y_ = hypothesis(X,theta)
	n = X.shape[1]
	for j in range(n):
		grad[j][0] = np.sum((Y - y_)*(X[:,j].reshape(-1,1)))

	return grad/X.shape[0]

def gradient_ascent(X,Y,learning_rate=0.1,epochs = 300):
	theta = np.zeros((X.shape[1],1),dtype=np.float64)
	error_list = []

	for i in range(epochs):
		grad = gradient(X,Y,theta)
		e = error_negativeLoss(X,Y,theta)
		error_list.append(e)
		theta = theta + learning_rate*grad
		# break

	# print(theta)

	return theta,error_list

def error_negativeLoss(X,Y,theta):
	hyp = hypothesis(X,theta)
	e = np.sum(Y*np.log10(hyp) + (1-Y)*np.log10(1-hyp))
	return -e/X.shape[0]

def grad_asc(X,Y,learning_rate=0.5,epochs=700):
	ones = np.ones((X.shape[0],1))
	X = np.concatenate((ones,X),axis=1)
	# print(X[:4,:4])
	Y = Y.reshape((X.shape[0],1))
	# print(Y.shape)
	return gradient_ascent(X,Y,learning_rate,epochs)

def predict(X,theta):
	'''if grad_asc has run already'''
	ones = np.ones((X.shape[0],1))
	X = np.concatenate((ones,X),axis=1)
	theta = theta.reshape((-1,1))
	res = np.zeros((X.shape[0],1))
	y_ = hypothesis(X,theta)
	for i in range(X.shape[0]):
		if y_[i]>=0.5:
			res[i][0] = 1
		else:
			res[i][0] = 0

	return res

#################datatsets

X = pd.read_csv('Logistic_X_Train.csv').values
Y = pd.read_csv('Logistic_Y_Train.csv').values
# print(X.shape)
# print(Y.shape)
theta,error_list = grad_asc(X,Y)

print(error_list)
plt.plot(error_list)
plt.show()

# y_ = predict(X,theta)
# print(y_)

test = pd.read_csv('Logistic_X_Test.csv').values
y_= predict(test,theta)

df = pd.DataFrame(y_)
df.to_csv('res.csv',index_label=['label'],index=False)