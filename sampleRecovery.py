import numpy as np
import pandas as pd
import math
from generateData import *

def sampleRecover(XNc,XNx,YNc,YNy,available_fraction,size):
	temp = 0
	temp1 = 0
	for i in range(10):
		if i == 0 :
			temp = XNc[i]
			temp1 = YNc[i]
		else:
			temp = np.concatenate((temp,XNc[i]),axis = 0)
			temp1 = np.concatenate((temp1,YNc[i]),axis = 0)
	XNc = temp
	YNc = temp1
	temp = 0
	temp1 = 0
	for i in range(10):
		if i == 0 :
			temp = XNx[i]
			temp1 = YNy[i]
		else:
			temp = np.concatenate((temp,XNx[i]),axis = 0)
			temp1 = np.concatenate((temp1,YNy[i]),axis = 0)
	XNx = temp
	YNy = temp1
	XNc_XNx = np.concatenate((XNc,XNx),axis = 0)
	YNc_YNy = np.concatenate((YNc,YNy),axis = 0)
	temp = np.mean(XNc_XNx,axis=0)
	temp1 = np.mean(YNc_YNy,axis=0)
	XNy = np.full((int(size*(1-available_fraction)/2),XNx.shape[1]),temp) 	### INITIALIZING XNy WITH AVERAGE VALUES ###
	YNx = np.full((int(size*(1-available_fraction)/2),YNy.shape[1]),temp1)	### INITIALIZING YNx WITH AVERAGE VALUES ###
	
	print(YNx.shape)
	
def main():
	data_usps = usps_read("usps.h5")
	data_mnist = mnist_read("mnist_train.csv")
	view1 = generateSamples(data_mnist)
	view2 = generateSamples(data_usps)
	Views = []
	Views.append(view1)
	Views.append(view2)
	XNc,YNc,XNx,YNy = generatePartialData(Views)
	XNc,XNx = generateOutliers(XNc,XNx)
	sampleRecover(XNc,XNx,YNc,YNy,0.6,500)
	# print(XNx)


main()
