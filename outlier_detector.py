#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import operator
from sklearn.metrics import mean_squared_error
import copy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.neighbors import NearestNeighbors
import h5py
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn import metrics


# # Reading Data

# In[2]:


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


# # Collecting Samples 500 each from each View

# In[3]:


### GENERATING 500 SAMPLES FOR EACH VIEW ###
def generateSamples(data):
    View = []
    for label in range(10):
        View.append(data[np.where(data[:,-1] == label)][0:50])
    View = np.array(View)
    return View


# # Creating Partial Data 20% missing in each view

# In[4]:


### PARTIAL DATA GENERATION ###
def generatePartialData(Views):
    size_of_data = 50
    split_c = 0.6    ### COMMON DATA POINTS ###
    split_v1 = 0.2   ### VIEW1 SPLIT ### 
    split_v2 = 0.2   ### VIEW2 SPLIT ###
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


# # Creating Outliers 10% in each View

# In[5]:


### FUNCTION TO GENERATE OUTLIERS IN THE AVAILABLE DATA BY SWAPPING 10% OF THE SAMPLES
def generateOutliers(XNc,XNx):
    X = []
    l1 = [1,2,3,4,7]
    l2 = [9,6,5,8,0]
    outliers_index = np.zeros((10,50))
    for i in range(10):
        X.append(np.concatenate((XNc[i],XNx[i]),axis = 0))    ### CONCATENATING XNX AND XNC ###
    swaps = math.floor(0.1 * X[0].shape[0])    ### 10% OF DATA SWAPPED BETWEEN CLASSES ###
    for i in range(len(l1)):
        swapped_index = random.sample(range(0, X[l1[i]].shape[0]), swaps)
        for j in swapped_index:    ### SWAPPING DATA IN l1 & l2	 CLASSES ###
            temp = np.array(X[l1[i]][j])
            X[l1[i]][j] = np.array(X[l2[i]][j])
            X[l2[i]][j] = temp
            outliers_index[l1[i]][j] = 1
            outliers_index[l2[i]][j] = 1
    for i in range(10):
        XNc[i] = X[i][0:XNc[i].shape[0],:]
        XNx[i] = X[i][XNc[i].shape[0]:,:] 
    return XNc,XNx,outliers_index    ### DATA WITH OUTLIERS ###


# # KNN Algorithm

# In[6]:


def get_distance(data,train_data,dist_type):
    dist = 0 
    if (dist_type == "euclid"):
        for i in range(len(train_data)-1):
            dist += pow(data[i]-train_data[i],2)
        dist = math.sqrt(dist)
    elif (dist_type == "manhattan"):
        for i in range(len(train_data)-1):
            dist += abs(data[i]-train_data[i])
    elif (dist_type == "minkowski"):
        for i in range(len(train_data)-1):
            dist += abs(data[i]-train_data[i]) ** 3
        dist = dist ** (1.0/3)
    return dist

def eucidean_distance(x,y):
    dist_sq=0
    for i in range(len(x)):
        dist_sq+=math.pow(x[i]-y[i],2)
    return math.sqrt(dist_sq)

def calc_distances(record,train_data):
    distances=[]
    for i in train_data:
        distances.append((eucidean_distance(record,i),i))
    return distances

def get_k_neighbours(distances,k):
    k_neighbours=[]
    distances.sort(key=lambda x: x[0])
    for i in range(k):
        k_neighbours.append(distances[i][1])
    return k_neighbours


def knn_predict(train_data,k):
    neighbour_list = []
    for data in train_data:
        dis = calc_distances(data,train_data)
        neighbours = get_k_neighbours(dis,k)
        neighbour_list.append([data,neighbours])
    return neighbour_list


# # Initializing Missing Data with Mean

# In[7]:


def initialization(XNc,XNx,YNc,YNy,available_fraction,size):

    XNc = XNc[:,0:XNc.shape[1]-1]
    YNc = YNc[:,0:YNc.shape[1]-1]

    XNx = XNx[:,0:XNx.shape[1]-1]
    YNy = YNy[:,0:YNy.shape[1]-1]

    XNc_XNx = np.concatenate((XNc,XNx),axis = 0)
    YNc_YNy = np.concatenate((YNc,YNy),axis = 0)
    temp = np.mean(XNc_XNx,axis=0)
    temp1 = np.mean(YNc_YNy,axis=0)

    XNy = np.full((math.ceil(size*(1-available_fraction)/2),XNx.shape[1]),temp) 	### INITIALIZING XNy WITH AVERAGE VALUES ###
    
    YNx = np.full((math.ceil(size*(1-available_fraction)/2),YNy.shape[1]),temp1)	### INITIALIZING YNx WITH AVERAGE VALUES ###

    return XNc,XNx,XNy,YNc,YNy,YNx


# # Missing Data Generation and Outlier Detection

# In[8]:


def sampleRecover_OutlierDetection(XNc,XNx,XNy,YNc,YNy,YNx,size,available_fraction,T,k):
    missing_fraction = (1 - available_fraction)/2
    X = np.concatenate((XNc,XNx),axis = 0)
    X = np.concatenate((X,XNy),axis = 0)

    Y = np.concatenate((YNc,YNx),axis = 0)
    Y = np.concatenate((Y,YNy),axis = 0)
#     print("Shape of Y view ",Y.shape)

    ### HSIC ###
    ones_vec = np.array(([1]*size))
    ones_vec = ones_vec.reshape((ones_vec.shape[0],1))
    
    H = np.identity(size) - 1/size * (np.dot(ones_vec,ones_vec.T))
    H = H/(size-1)
    C = np.dot(ones_vec,ones_vec.T)
    
    for i in range(T):
        print("Iteration ",i,"/",T)
        diag_C = np.diag(np.diag(C))
        
        error_y, error_x = 1, 1
        for j in range(500): ###  FOR CONVERGENCE ###
            P = np.dot(diag_C,np.dot(np.dot(H,X),np.dot(X.T,H))) 
            PNcNx = P[int(size * available_fraction) : int(size * (missing_fraction + available_fraction)),0:int(size * available_fraction)]
            PNxNx = P[int(size * available_fraction) : int(size * (missing_fraction + available_fraction)),int(size * available_fraction) : int(size * (missing_fraction + available_fraction))]
            PNxNy = P[int(size * available_fraction) : int(size * (missing_fraction + available_fraction)),int(size * (available_fraction+missing_fraction)) : int(size * (2*missing_fraction + available_fraction))]

            ### TO GUARANTEE THE INVERTIBILITY, WE USUALLY ADD A SMALL PERTURBATION OF 10^-6 TO EACH MAIN DIAGONAL ELEMENT OF PNxNx ###
            a = np.zeros((PNxNx.shape[0], PNxNx.shape[1]), float)
            np.fill_diagonal(a, 0.000001)
            PNxNx = PNxNx + a

            YNx_old = copy.deepcopy(YNx)
            YNx = -1 * np.dot(np.linalg.inv(PNxNx),(np.dot(PNcNx,YNc) + np.dot(PNxNy,YNy)))
            error_y = mean_squared_error(YNx_old,YNx)

            Y = np.concatenate((YNc,YNx),axis = 0)
            Y = np.concatenate((Y,YNy),axis = 0)

            Q = np.dot(diag_C,np.dot(np.dot(H,Y),np.dot(Y.T,H)))
            QNcNy = Q[int(size * (available_fraction+missing_fraction)) :,0:int(size * available_fraction)]
            QNxNy = Q[int(size * (available_fraction + missing_fraction)) :,int(size * available_fraction) : int(size * (missing_fraction + available_fraction))]
            QNyNy = Q[int(size * (available_fraction+missing_fraction)) :,int(size * (available_fraction+missing_fraction)) : int(size * (2*missing_fraction + available_fraction))]
            
            XNy_old = copy.deepcopy(XNy)
            XNy = -1 * np.dot(np.linalg.inv(QNyNy),(np.dot(QNcNy,XNc) + np.dot(QNxNy,XNx)))
            error_x = mean_squared_error(XNy_old,XNy)

            X = np.concatenate((XNc,XNx),axis = 0)
            X = np.concatenate((X,XNy),axis = 0)

        print("After convergence error in XNy and YNx view is - ",error_x,error_y)

        ###KNN
        neighbour_list =  knn_predict(X,k)
        WX = np.zeros(shape =(size,size))
        for i in range(size):
            for j in range(size):
                if any(np.array_equal(x, neighbour_list[i][0]) for x in neighbour_list[j][1]):
                    WX[i][j] = 1
                    WX[j][i] = 1
                if any(np.array_equal(x, neighbour_list[j][0]) for x in neighbour_list[i][1]):
                    WX[i][j] = 1
                    WX[j][i] = 1

        neighbour_list =  knn_predict(Y,k)

        WY = np.zeros(shape =(size,size))
        for i in range(size):
            for j in range(size):
                if any(np.array_equal(x, neighbour_list[i][0]) for x in neighbour_list[j][1]) or any(np.array_equal(x, neighbour_list[j][0]) for x in neighbour_list[i][1]):
                    WY[i][j] = 1
                    WY[j][i] = 1
                if any(np.array_equal(x, neighbour_list[j][0]) for x in neighbour_list[i][1]):
                    WY[i][j] = 1

        s = np.dot(np.dot(H,WX),np.dot(H,WY))
        s = s.diagonal()
        s = np.interp(s, (s.min(), s.max()), (0.1, 1))
        S = np.zeros(shape =(size,size))
        np.fill_diagonal(S, s)
        C = np.dot(S,S.T)

    return S.diagonal(),X,Y


# In[11]:


get_ipython().run_cell_magic('time', '', '# neighbour_list = []\ndata_usps = usps_read("usps.h5")\ndata_mnist = mnist_read("mnist_train.csv")\n\n# scaler = MinMaxScaler()\n# scaler.fit(data_usps)\n# data_usps = scaler.fit_transform(data_usps)\n# scaler.fit(data_mnist)\n# data_mnist = scaler.fit_transform(data_mnist)\n\nview1 = generateSamples(data_mnist)\nview2 = generateSamples(data_usps)\nViews = []\nViews.append(view1)\nViews.append(view2)\nXNc,YNc,XNx,YNy = generatePartialData(Views)\nXNc,XNx,outliers_index = generateOutliers(XNc,XNx)\n\nS,X,Y = [],[],[]\nl1 = [1,2,3,4,7]\nl2 = [9,6,5,8,0]\nsize = 50\nfor i in range(0,10):\n    print(\'Set \',i)\n    xnc,xnx,xny,ync,yny,ynx = initialization(XNc[i],XNx[i],YNc[i],YNy[i],0.6,size)\n    \n    print("X-view")\n    fig, ax = plt.subplots(4, 10, subplot_kw=dict(xticks=[], yticks=[]),figsize=(20,10))\n    c = 0\n    for i in xnc:\n        ax[int(c/10),c%10].imshow(i.reshape(28,28))\n        c += 1\n    for i in xnx:\n        ax[int(c/10),c%10].imshow(i.reshape(28,28))\n        c += 1\n    plt.show()\n    \n    print("Y-view")\n    fig, ax = plt.subplots(4, 10, subplot_kw=dict(xticks=[], yticks=[]),figsize=(20,10))\n    c = 0\n    for i in ync:\n        ax[int(c/10),c%10].imshow(i.reshape(16,16))\n        c += 1\n    for i in yny:\n        ax[int(c/10),c%10].imshow(i.reshape(16,16))\n        c += 1\n    plt.show()\n    \n    s,x,y = sampleRecover_OutlierDetection(xnc,xnx,xny,ync,yny,ynx,size,0.6,100,10)\n    \n    print(\'Generated Images in X View\')\n    fig, ax = plt.subplots(1, 10, subplot_kw=dict(xticks=[], yticks=[]),figsize=(20,10))\n    c = 0\n    for j in x[40:50]:\n        ax[c].imshow(j.reshape(28,28))\n        c = c+1\n    plt.show()\n    \n    print(\'Generated Images in Y View\')\n    fig, ax = plt.subplots(1, 10, subplot_kw=dict(xticks=[], yticks=[]),figsize=(20,10))\n    c = 0\n    for j in y[30:40]:\n        ax[c].imshow(j.reshape(16,16))\n        c = c+1\n    plt.show()\n    \n    S.append(s)\n    X.append(x)\n    Y.append(y)')


# In[23]:


aucs = []
for i in range(0,10):
    outliers_list = np.array([ a<0.3 for a in S[i] ])
    indices = np.array(np.where(outliers_list == 1))[0]
#     print('For set ',i,' outlier are ',len(indices))
    print('For set ',i)
#     print('indices are ',indices)
#     for j in indices:
#         plt.imshow(X[i][j].reshape(28,28))
#         plt.show()
    print("AUC ",roc_auc_score(outliers_list,outliers_index[0]))
    aucs.append(roc_auc_score(outliers_list,outliers_index[0]))
    plt.figure()
    fpr, tpr, thresholds = metrics.roc_curve(outliers_index[i],outliers_list)
    plt.plot(fpr, tpr, color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
plt.plot(aucs)
plt.ylabel('AUC')
plt.xlabel('Set')
plt.show()


# In[21]:


k = [3,4,5,6,7,8,9,10]
aucs = [0.369,0.49,0.472,0.53,0.61,0.45,0.49,0.457]
plt.plot(k,aucs)
plt.ylabel('AUC')
plt.xlabel('K')
plt.show()


# In[ ]:




