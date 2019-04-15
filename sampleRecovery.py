import numpy as np
import pandas as pd
import math
from generateData import *

def initialization(XNc,XNx,YNc,YNy,available_fraction,size):
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
    # XNc = XNc[:,0:XNc.shape[1]-1]
    # YNc = YNc[:,0:YNc.shape[1]-1]
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
    # XNx = XNx[:,0:XNx.shape[1]-1]
    # YNy = YNy[:,0:YNy.shape[1]-1]

    XNc_XNx = np.concatenate((XNc,XNx),axis = 0)
    YNc_YNy = np.concatenate((YNc,YNy),axis = 0)
    temp = np.mean(XNc_XNx,axis=0)
    temp1 = np.mean(YNc_YNy,axis=0)
    XNy = np.full((int(size*(1-available_fraction)/2),XNx.shape[1]),temp) 	### INITIALIZING XNy WITH AVERAGE VALUES ###
    YNx = np.full((int(size*(1-available_fraction)/2),YNy.shape[1]),temp1)	### INITIALIZING YNx WITH AVERAGE VALUES ###



    # print(YNx.shape)
    return XNc,XNx,XNy,YNc,YNy,YNx

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

def knn_predict(train_data,k):
    neighbour_list = []
    for data in train_data:
        distances = []
        for row in train_data:
            dist = get_distance(data,row,"euclid")
            distances.append((row,dist))       
        distances.sort(key=operator.itemgetter(1))
        neighbours = []
        for x in range(k):
            neighbours.append(distances[x][0])
        neighbour_list.append([data,neighbours])
    return neighbour_list

def sampleRecover_OutlierDetection(XNc,XNx,XNy,YNc,YNy,YNx,size,available_fraction,T,k,threshold):
    missing_fraction = (1 - available_fraction)/2
    X = np.concatenate((XNc,XNx),axis = 0)
    X = np.concatenate((X,XNy),axis = 0)

    print(X.shape)
    Y = np.concatenate((YNc,YNx),axis = 0)
    Y = np.concatenate((Y,YNy),axis = 0)
    print(Y.shape)

    ### HSIC ###
    ones_vec = np.array(([1]*size))
    ones_vec = ones_vec.reshape((ones_vec.shape[0],1))
    # print(np.dot(ones_vec,ones_vec.T))
    H = np.identity(size) - 1/size * (np.dot(ones_vec,ones_vec.T))
    H = H/(size-1)
    C = np.dot(ones_vec,ones_vec.T)

    for i in range(1):
        diag_C = np.diag(np.diag(C))
        # print(diag_C.shape)
        # print(i)
        error_y, error_x = 1, 1
        for i in range(100): ###  FOR CONVERGENCE ###
            P = np.dot(diag_C,np.dot(np.dot(H,X),np.dot(X.T,H))) 
            PNcNx = P[int(size * available_fraction) : int(size * (missing_fraction + available_fraction)),0:int(size * available_fraction)]
            PNxNx = P[int(size * available_fraction) : int(size * (missing_fraction + available_fraction)),int(size * available_fraction) : int(size * (missing_fraction + available_fraction))]
            PNxNy = P[int(size * available_fraction) : int(size * (missing_fraction + available_fraction)),int(size * (available_fraction+missing_fraction)) : int(size * (2*missing_fraction + available_fraction))]

            ### TO GUEAGANTEE THE INVERTIBILITY, WE USUALLY ADD A SMALL PERTURBATION OF 10^-6 TO EACH MAIN DIAGONAL ELEMENT OF PNxNx ###
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

#             print(error_x,error_y)
            X = np.concatenate((XNc,XNx),axis = 0)
            X = np.concatenate((X,XNy),axis = 0)

            # print("----------------------------------")
            # print(XNy)
            # print("----------------------------------")

        # print("adadasda")
        # print(QNcNy.shape)
        # print(QNxNy.shape)
        # print(QNyNy.shape)
        # print("----------------------")
        print("After convergence error in XNy and YNx view is - ",error_x,error_y)
#         neighbour_list = knn_predict(X,10)
        neighbour_list =  knn_predict(X,10)
        WX = np.array(([0]*size))
        for i in range(size):
            for j in range(size):
                if any(np.array_equal(x, neighbour_listghbour_list[i][0]) for x in neighbour_list[j][1]):
                    WX[i][j] = 1
        
    # print(X.shape)



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
    size = 500
    XNc,XNx,XNy,YNc,YNy,YNx = initialization(XNc,XNx,YNc,YNy,0.6,size)
    sampleRecover_OutlierDetection(XNc,XNx,XNy,YNc,YNy,YNx,size,0.6,10,7,0.5)

    # print(XNx)


# main()
