import numpy as np
import pandas as pd
import math
import h5py

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
	split_c = 0.6				### COMMON DATA POINTS ###
	split_v1 = 0.2 				### VIEW1 SPLIT ### 
	split_v2 = 0.2					### VIEW2 SPLIT ###
	ncommon = math.ceil(split_c*size_of_data)
	
	### PART COMMON IN BOTH VIEWS ###
	XNc = []
	YNc = []
	for labeled_data in Views[0]:
		XNc.append(labeled_data[0:ncommon,:])
	for labeled_data in Views[1]:
		YNc.append(labeled_data[0:ncommon,:])
	
	### PART MISSING IN VIEW1 ###
	XNx = []
	for labeled_data in Views[0]:
		XNx.append(labeled_data[ncommon:ncommon + math.ceil(split_v1*size_of_data),:])

	### PART MISSING IN VIEW2 ###
	YNy = []
	for labeled_data in Views[1]:
		YNy.append(labeled_data[ncommon + math.ceil(split_v1*size_of_data):,:])
	print(YNy[0].shape)
	
	print(XNx[0].shape)

data_usps = usps_read("usps.h5")
data_mnist = mnist_read("mnist_train.csv")
view1 = generateSamples(data_mnist)
view2 = generateSamples(data_usps)
Views = []
Views.append(view1)
Views.append(view2)
generatePartialData(Views)
# v1,v2 = getViews(X)
# print(v2)
