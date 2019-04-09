import numpy as np
import pandas as pd

def read_data(file_name):
	data = pd.read_csv(file_name)
	X = data.loc[:,'1x1':'28x28']
	Y = data['label']
	return X,Y


def getViews(X):
	Features1 = X.loc[:,'1x1':'16x20']
	Features2 = X.loc[:,'15x7':'28x28']
	Views = []
	Views.append(Features1)
	Views.append(Features2)
	
	return Features1,Features2

X,Y = read_data("mnist_train.csv")
v1,v2=getViews(X)
print(v2)
