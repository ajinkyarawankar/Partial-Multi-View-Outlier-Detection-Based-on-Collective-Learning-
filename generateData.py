import numpy as np
import pandas as pd
import math
import h5py
import random

### READING USPS DATA ###
def usps_read(file_name):
	with h5py.File(file_name, 'r') as hf:
		train = hf.get('train')
		X_tr = train.get('data')[:]
		y_tr = train.get('target')[:]
		y_tr = y_tr.reshape((len(y_tr),1))
		data = np.append(X_tr,y_tr,axis=1)
		return data

### READING MNIST DATA ###
def mnist_read(file_name):
	data = pd.read_csv(file_name)
	X = data.loc[:,'1x1':'28x28']
	Y = data['label']
	X = np.array(X)
	Y = np.array(Y)
	Y = Y.reshape((len(Y),1))
	data = np.append(X,Y,axis=1)
	return data

### GENERATING 500 SAMPLES FOR EACH VIEW ###
def generateSamples(data):
	View = []
	for label in range(10):
		View.append(data[np.where(data[:,-1] == label)][0:50])
	View = np.array(View)
	return View

### PARTIAL DATA GENERATION ###
def generatePartialData(Views):
	size_of_data = 50
	split_c = 0.6					### COMMON DATA POINTS ###
	split_v1 = 0.2 					### VIEW1 SPLIT ### 
	split_v2 = 0.2					### VIEW2 SPLIT ###
	ncommon = math.ceil(split_c*size_of_data)
	
	### PART COMMON IN BOTH VIEWS ###
	XNc = []
	YNc = []
	for labeled_data in Views[0]:
		XNc.append(labeled_data[0:ncommon,:])
	for labeled_data in Views[1]:
		YNc.append(labeled_data[0:ncommon,:])
	
	### PART MISSING IN VIEW 2 ###
	XNx = []
	for labeled_data in Views[0]:
		XNx.append(labeled_data[ncommon:ncommon + math.ceil(split_v1*size_of_data),:])

	### PART MISSING IN VIEW 1 ###
	YNy = []
	for labeled_data in Views[1]:
		YNy.append(labeled_data[ncommon + math.ceil(split_v1*size_of_data):,:])
	return XNc,YNc,XNx,YNy

### FUNCTION TO GENERATE OUTLIERS IN THE AVAILABLE DATA BY SWAPPING 10% OF THE SAMPLES
def generateOutliers(XNc,XNx):
	X = []
	for i in range(10):
		X.append(np.concatenate((XNc[i],XNx[i]),axis = 0)) 				### CONCATENATING XNX AND XNC ###
	swaps = math.floor(0.1 * X[0].shape[0]) 							### 10% OF DATA SWAPPED BETWEEN CLASSES ###
	for i in range(0,10,2):
		swapped_index = random.sample(range(0, X[i].shape[0]), swaps)
		for j in swapped_index:			 								### SWAPPING DATA IN ADJACENT CLASSES ###
			temp = X[i][j]
			X[i][j] = X[i+1][j]
			X[i+1][j] = temp
	return X 															### DATA WITH OUTLIERS ###

### MAIN CALLING FUNCTION ###
def main():
	data_usps = usps_read("usps.h5")
	data_mnist = mnist_read("mnist_train.csv")
	view1 = generateSamples(data_mnist)
	view2 = generateSamples(data_usps)
	Views = []
	Views.append(view1)
	Views.append(view2)
	XNc,YNc,XNx,YNy = generatePartialData(Views)
	X = generateOutliers(XNc,XNx)
	print(X)

main()
