##########read data
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import random
#from random import random
from random import randint
from hmmlearn import hmm

import warnings

warnings.filterwarnings('ignore')
dataFile = 'trajectories_train.mat'
data = scio.loadmat(dataFile)
testdata = scio.loadmat('trajectories_Xtest.mat')
xtest=testdata['xtest'][0]
xtrain = data['xtrain'][0]
ytrain = data['ytrain'][0]
def get_y_distribution(xtrain,ytrain):
    l=[]
    for i in range(1,21):
        c = xtrain[ytrain==i].shape[0]
        l.append([i,c])
    return np.array(l)
def plot_bar(D):
    x,y=[],[]
    for i in D:
        x.append(i[0])
        y.append(i[1])
    plt.bar(x,y)
y_distribution = get_y_distribution(xtrain,ytrain)
#plot_bar(y_distribution)
def transpose(x):
    l = []
    for e in x:
        l.extend(list(e.transpose()))    
    return np.array(l)

def cv_fix(trainx, trainy):
    x_dense = []
    y_dense = []
    for i in trainx:
        x_dense.extend(i)
    for i in trainy:
        y_dense.extend(i)
        
    x_fix = [[] for _ in range(20)]
#    x_dense = np.array(x_dense)
    y_dense = np.array(y_dense)
    for i in range(1,21):
        yi = (y_dense==i)
        for j in range(len(x_dense)):
            if yi[j]:
                x_fix[i-1].append(x_dense[j])

    return x_fix

def data_split(xtrain,ytrain, k=10):
    
    list_x = [[] for _ in range(k)]
    list_y = [[] for _ in range(k)]

    random_matrix = np.random.rand(20,k)*0.6 / (k+.0) + 0.7/(k+.0)
#    random_matrix[:,-1] = 1/(k+.0) 
    for i in range(20):
        random_matrix[i] = random_matrix[i] / sum(random_matrix[i])
#        smooth = 0.02 / (1-0.02*k)
#        random_matrix[i] = (random_matrix[i] + smooth) / (sum(random_matrix[i]) + smooth*k)

    for i in range(1,21):
        xi = list(xtrain[ytrain==i])
        random.shuffle(xi)
        
        split = np.around(random_matrix[i-1] * len(xi))
        split = np.array(split, dtype='int32')
        split[k-1] = len(xi) - sum(split[0:k-1])
        skip = 0
        
        yi = [ i for _ in range(len(xi))]
        for j in range(k):
            list_y[j].extend(yi[skip : skip+split[j]])
            list_x[j].extend(xi[skip : skip+split[j]])
            skip += split[j]
        
    return list_x,list_y
##trainx is a list(20classes),element in list is still a list, need transpose
########simple predict function, NO PRIOR!!!!!!!!!!    
def predict(model_list,testx):
    predict = []
    for test in testx:
        t = test.transpose()
        score = -float('Inf')
        for i in range(20):
            model_score=model_list[i].score(t)
            print(model_score)
            if model_score>score:
                score = model_score
                y_pre = i+1
        predict.append(y_pre)
    return predict
def accuracy(y,y_pre):
    if len(y)!=len(y_pre):
        print("SORRY MAN, BUT FUCK U!")
        return 
    correct = 0 
    for i in range(len(y)):
        if y[i] == y_pre[i]:
            correct += 1
    return float(correct)/len(y)
def get_prior(trainy):
    l=[]
    total = 0
    for i in trainy:
        l.append(len(i))
        total+=len(i)
    for j in range(20):
        l[j] = l[j]/total
    return l
    

def main():
    k=5
    list_x, list_y = data_split(xtrain,ytrain,k)
    acc=[]
    for i in range(k):
        testx = list_x[i]
        testy = list_y[i]
        trainx = list_x[:i]+list_x[i+1:]
        trainy = list_y[:i]+list_y[i+1:]

        trainx_fix = cv_fix(trainx, trainy)
        model_list=[hmm.GMMHMM(n_components= 10,n_mix = 10, covariance_type='diag',algorithm='viterbi', 
                               n_iter=30,params='stmc', init_params='stmc') for _ in range(20)]
        for j in range(20):
            X = transpose(trainx_fix[j])
            model_list[j].fit(X)
        predic = predict(model_list,testx)
        accurac = accuracy(testy,predic)
        acc.append(accurac)
        print(accurac)
    return acc

