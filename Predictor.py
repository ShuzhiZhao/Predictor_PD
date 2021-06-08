## Author:SHUZHI ZHAO
## Predict age UPDRS,MMSE in PD
from numpy import linalg as LA
import numpy as np
import pandas as pd
import scipy.io as scio
from scipy import sparse
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.model_selection import cross_val_score, train_test_split,ShuffleSplit
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import *
import os
import lmdb
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops
from PIL import Image
import nibabel as nib
from model.inceptionv4 import Inceptionv4
from model.SNN import SpikingNet
from model.tcn import TemporalConvNet
from utils import readScores

## **********************************************************************************
## **************************************Predictor***********************************
## **********************************************************************************
## Prediction with Inception Resenet
def Prediction(work_dir):
    ## lmbd dir for T1 DTI Bold and ERP
    T1_dir = '/media/lhj/Momery/PD_predictDL/Data/FT_InceptionResnetv4'
    DTI_dir = '/media/lhj/Momery/PD_predictDL/Data/FT_SNN'
    Bold_dir = '/media/lhj/Momery/PD_predictDL/Data/FT_TCAN'
    ERP_dir = '/media/lhj/Momery/PD_predictDL/Data/FT_ERP'
    ## names for T1 DTI Bold and ERP
    T1_names = '/media/lhj/Momery/PD_predictDL/Data/name_T1.txt'
    DTI_names = '/media/lhj/Momery/PD_predictDL/Data/name_DTI.txt'
    Bold_names = '/media/lhj/Momery/PD_predictDL/Data/name_Bold.txt'
    ERP_names = '/media/lhj/Momery/PD_predictDL/Data/name_ERP.txt'
    
    ## get T1's Feature and labels after Inception-Resnetv4 (6480,1000)
    FT_T1,T1_labels = getLmbd(T1_dir,T1_names)
    ## get DTI's Feature and labels after SNN (5200,100)
    FT_DTI,DTI_labels = getLmbd(DTI_dir,DTI_names)
    FT_DTI,DTI_labels = tranData(FT_DTI,DTI_labels,index=65)
    ## get Bold's Feature and labels after SNN (19440,100)
    FT_Bold,Bold_labels = getLmbd(Bold_dir,Bold_names)
    FT_Bold,Bold_labels = tranData(FT_Bold,Bold_labels,index=240)
    ## get ERP's Feature and labels after TCN (5200,8450)
    FT_ERP,ERP_labels = getLmbd(ERP_dir,ERP_names)
    
    ## train and test set
#     print('**************** ',np.array(FT_T1).shape,np.array(subLabels(T1_labels,1)).shape,' ****************')
    trainData_T1,trainLabels_T1,testData_T1,testLabels_T1 = crossDataLabel(FT_T1,subLabels(T1_labels,1))
    trainData_DTI,trainLabels_DTI,testData_DTI,testLabels_DTI = crossDataLabel(FT_DTI,subLabels(DTI_labels,1))
    trainData_Bold,trainLabels_Bold,testData_Bold,testLabels_Bold = crossDataLabel(FT_Bold,subLabels(Bold_labels,1))
    trainData_ERP,trainLabels_ERP,testData_ERP,testLabels_ERP = crossDataLabel(FT_ERP,subLabels(ERP_labels,1))
#     print(trainData_ERP,trainLabels_ERP,testData_ERP,testLabels_ERP)
    
    ## Predictor
    #############################################
    ################## single mode ##############
    #############################################
#     index = 3 
#     nameLabel = ['age', 'UPDRS', 'MMSE']
#     model_test1 = Net(n_feature=np.array(FT_T1).shape[1], n_hidden=200, n_output=index)
#     model_test2 = Net(n_feature=np.array(FT_ERP).shape[1], n_hidden=1000, n_output=index)
#     model_test3 = Net(n_feature=np.array(FT_DTI).shape[1], n_hidden=50, n_output=index)
#     model_test4 = Net(n_feature=np.array(FT_Bold).shape[1], n_hidden=50, n_output=index)
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# #     device = torch.device('cpu')
#     model_test1 = model_test1.to(device)
#     model_test2 = model_test2.to(device)
#     model_test3 = model_test3.to(device)
#     model_test4 = model_test4.to(device)
#     print("{} paramters to be trained in the model\n".format(count_parameters(model_test1)))
#     optimizer1 = optim.Adam(model_test1.parameters(),lr=0.001, weight_decay=5e-4)
#     optimizer2 = optim.Adam(model_test2.parameters(),lr=0.001, weight_decay=5e-4)
#     optimizer3 = optim.Adam(model_test3.parameters(),lr=0.001, weight_decay=5e-4)
#     optimizer4 = optim.Adam(model_test4.parameters(),lr=0.001, weight_decay=5e-4)
#     loss_func = nn.L1Loss(reduction='mean')
#     num_epochs=15
#     print('******************************************'+nameLabel[index-1]+'************************************************')
#     model_fit_evaluate(model_test1,device,trainData_T1,trainLabels_T1,testData_T1,testLabels_T1,optimizer1,loss_func,num_epochs)
#     print("{} paramters to be trained in the model\n".format(count_parameters(model_test2)))
#     model_fit_evaluate(model_test2,device,trainData_ERP,trainLabels_ERP,testData_ERP,testLabels_ERP,optimizer2,loss_func,num_epochs)
#     print("{} paramters to be trained in the model\n".format(count_parameters(model_test3)))
#     model_fit_evaluate(model_test3,device,trainData_DTI,trainLabels_DTI,testData_DTI,testLabels_DTI,optimizer3,loss_func,num_epochs)
#     print("{} paramters to be trained in the model\n".format(count_parameters(model_test4)))
#     model_fit_evaluate(model_test4,device,trainData_Bold,trainLabels_Bold,testData_Bold,testLabels_Bold,optimizer4,loss_func,num_epochs)
    #############################################
    ################## fusion mode ##############
    #############################################
    conName_dir = '/media/lhj/Momery/PD_predictDL/Data/conNames.txt'
#     fusFT,fusLabels = fusionFT(DTI_dir,'DTI',Bold_dir,'Bold',conName_dir)
#     fusFT,fusLabels = fusionFT(T1_dir,'T1',Bold_dir,'Bold',conName_dir)
#     fusFT,fusLabels = fusionFT(DTI_dir,'DTI',T1_dir,'T1',conName_dir)
#     print(np.array(fusFT).shape,np.array(fusLabels).shape)
    
    #############################################
    ############# prediction with fusFT #########
    #############################################
    fusFT_dir = '/media/lhj/Momery/PD_predictDL/Data/FT_fus1'
    preFusFT(fusFT_dir,conName_dir)

def preFusFT(fusFT_dir,conName_dir):
    names = readScores(conName_dir)
    ## get fusFT underlying DTI and Bold
    FT1,scores1,FT2,scores2,fusFT,scores3 = getLmdbFus(fusFT_dir,'DTI','Bold',names)
    print(np.array(FT1).shape,np.array(scores1).shape,np.array(FT2).shape,np.array(scores2).shape,np.array(fusFT).shape,np.array(scores3).shape)
    print('***************** DTI predictor for age UPDRS and MMSE *****************')    
    predLinDL(FT1,scores1)
    print('***************** Bold predictor for age UPDRS and MMSE *****************')
    predLinDL(FT2,scores2)
    print('***************** fusFT predictor for age UPDRS and MMSE *****************')
    predLinDL(fusFT,scores3)
    
    FT1,scores1,FT2,scores2,fusFT,scores3 = getLmdbFus(fusFT_dir,'DTI','T1',names)
    print(np.array(FT1).shape,np.array(scores1).shape,np.array(FT2).shape,np.array(scores2).shape,np.array(fusFT).shape,np.array(scores3).shape)
    print('***************** DTI predictor for age UPDRS and MMSE *****************')    
    predLinDL(FT1,scores1)
    print('***************** T1 predictor for age UPDRS and MMSE *****************')
    predLinDL(FT2,scores2)
    print('***************** fusFT predictor for age UPDRS and MMSE *****************')
    predLinDL(fusFT,scores3)
    
    FT1,scores1,FT2,scores2,fusFT,scores3 = getLmdbFus(fusFT_dir,'T1','Bold',names)
    print(np.array(FT1).shape,np.array(scores1).shape,np.array(FT2).shape,np.array(scores2).shape,np.array(fusFT).shape,np.array(scores3).shape)
    print('***************** T1 predictor for age UPDRS and MMSE *****************')    
    predLinDL(FT1,scores1)
    print('***************** Bold predictor for age UPDRS and MMSE *****************')
    predLinDL(FT2,scores2)
    print('***************** fusFT predictor for age UPDRS and MMSE *****************')
    predLinDL(fusFT,scores3)


    
def getLmdbFus(fusFT_dir,FT1_Type,FT2_Type,names):
    FT1 = []
    scores1 = []
    FT2 = []
    scores2 = []
    fusFT = []
    scores3 = []
    lmdb_env = lmdb.open(fusFT_dir+FT1_Type+FT2_Type,readonly=True)
    with lmdb_env.begin() as lmdb_mod_txn:
        for ii in names:
            sliceNum1 = 0
            sliceNum2 = 0
            if "T1" in FT1_Type and "Bold" in FT2_Type:
                sliceNum1 = 80
                sliceNum2 = 240                    
            elif "DTI" in FT1_Type and "Bold" in FT2_Type:
                sliceNum1 = 65
                sliceNum2 = 240 
            elif "DTI" in FT1_Type and "T1" in FT2_Type:
                sliceNum1 = 65
                sliceNum2 = 80
            for jj in range(sliceNum1):
                ## get FT1 and scores1
                FT1.append(np.frombuffer(lmdb_mod_txn.get((ii+'_'+FT1_Type+str(jj)+'_FT1').encode()),dtype=np.float64))
                scores1.append(np.frombuffer(lmdb_mod_txn.get((ii+'_'+FT1_Type+str(jj)+'_'+FT2_Type+str(0)+'_scores').encode()),dtype=np.float64))    
                
            for gg in range(sliceNum2):
                ## get FT2 and scores2
                FT2.append(np.frombuffer(lmdb_mod_txn.get((ii+'_'+FT2_Type+str(gg)+'_FT2').encode()),dtype=np.float64))
                scores2.append(np.frombuffer(lmdb_mod_txn.get((ii+'_'+FT1_Type+str(jj)+'_'+FT2_Type+str(0)+'_scores').encode()),dtype=np.float64))
                
            for jj in range(sliceNum1):
                for gg in range(sliceNum2):
                    ## get fusFT and scores3
                    fusFT.append(np.frombuffer(lmdb_mod_txn.get((ii+'_'+FT1_Type+str(jj)+'_'+FT2_Type+str(gg)+'_fusFTs').encode()),dtype=np.float64))
                    scores3.append(np.frombuffer(lmdb_mod_txn.get((ii+'_'+FT1_Type+str(jj)+'_'+FT2_Type+str(gg)+'_scores').encode()),dtype=np.float64))
                    
    return FT1,scores1,FT2,scores2,fusFT,scores3


def predLinDL(FT,scores):
    index = 3 
    nameLabel = ['age', 'UPDRS', 'MMSE']
    model_test = Net(n_feature=np.array(FT).shape[1], n_hidden=200, n_output=index)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_test = model_test.to(device)
    print("{} paramters to be trained in the model\n".format(count_parameters(model_test)))
    optimizer = optim.Adam(model_test.parameters(),lr=0.001, weight_decay=5e-4)
    loss_func = nn.L1Loss(reduction='mean')
    num_epochs=15
    trainData,trainLabels,testData,testLabels = crossDataLabel(FT,subLabels(scores,index))
    print('******************************************'+nameLabel[index-1]+'************************************************')
    model_fit_evaluate(model_test,device,trainData,trainLabels,testData,testLabels,optimizer,loss_func,num_epochs)
    

##  FT1_dir:lmdb dir for FT1; FT1_names:txt dir for FT1   
def fusionFT(FT1_lmdb,FT1_Type,FT2_lmdb,FT2_Type,conName_dir):
    from model.fusNet import fusNet
    fusFT = []
    fusLabels = []
    lmdb_dir = '/media/lhj/Momery/PD_predictDL/Data/FT_fus1'
    env = lmdb.open(lmdb_dir+FT1_Type+FT2_Type, map_size = int(1e12)*2)
    txn = env.begin(write=True)
    names = readScores(conName_dir)
    ## for subjects
    for ii in names:
        ## get FT1 and FT2 with 3D from lmdb
        FT1_data,FT1_scores = get3DFT(FT1_lmdb,ii,FT1_Type)
        FT2_data,FT2_scores = get3DFT(FT2_lmdb,ii,FT2_Type)
        if np.array(FT1_data).shape[2] != np.array(FT2_data).shape[2]:
            m = nn.Conv1d(np.array(FT1_data).shape[2], np.array(FT2_data).shape[2], 1, stride=1)
            FT1_data = m(torch.FloatTensor(FT1_data).transpose(1,2)).transpose(1,2).tolist()
#         print(np.array(FT1_data).shape,np.array(FT1_scores).shape,np.array(FT2_data).shape,np.array(FT2_scores).shape)
#         print(FT1_scores,FT2_scores)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mod = fusNet().to(device)
        for jj in range(np.array(FT1_data).shape[1]):
            for gg in range(np.array(FT2_data).shape[1]):
                f = mod(torch.cuda.FloatTensor(FT1_data[0][jj]).view(1,1,np.array(FT1_data).shape[2]),torch.cuda.FloatTensor(FT2_data[0][gg]).view(1,1,np.array(FT2_data).shape[2]))
                txn.put((ii+'_'+FT1_Type+str(jj)+'_FT1').encode(), np.array(f[0].tolist()))
                txn.put((ii+'_'+FT2_Type+str(gg)+'_FT2').encode(), np.array(f[1].tolist()))
                txn.put((ii+'_'+FT1_Type+str(jj)+'_'+FT2_Type+str(gg)+'_fusFTs').encode(), np.array(f[2].tolist()))
                txn.put((ii+'_'+FT1_Type+str(jj)+'_'+FT2_Type+str(gg)+'_scores').encode(), np.array(FT1_scores[0]))
#                 fusFT.append(f)
#                 fusLabels.append(FT1_scores[0])        
    txn.commit() 
    env.close()
    return fusFT,fusLabels


def get3DFT(FT1_lmdb,name,dataType):
    data = []
    scores = []
    lmdb_env = lmdb.open(FT1_lmdb,readonly=True)
    with lmdb_env.begin() as lmdb_mod_txn:
        temp = []
        if "T1" in dataType:
            temp1 = []
            for j in range(80):
                temp1.append(np.frombuffer(lmdb_mod_txn.get((name+'-'+str(j)+'_RGB').encode()),dtype=np.float64))
            data.append(temp1)
            name = name+'-0'
        elif "DTI" in dataType:
            temp1 = []
            for j in range(65):
                temp1.append(np.frombuffer(lmdb_mod_txn.get((name+'_DTI_Slice_'+str(j)).encode()),dtype=np.float64))
            data.append(temp1)
            name = name+'_DTI'                
        elif "Bold" in dataType:
            temp1 = []
            for j in range(240):
                temp1.append(np.frombuffer(lmdb_mod_txn.get(('PD_'+name+'_Bold_Slice_'+str(j)).encode()),dtype=np.float64))
            data.append(temp1)
            name = 'PD_'+name+'_Bold'            
        temp.append(float(np.frombuffer(lmdb_mod_txn.get((name+'_age').encode()),dtype=np.float64)))
        temp.append(float(np.frombuffer(lmdb_mod_txn.get((name+'_UPDRS').encode()),dtype=np.float64)))
        temp.append(float(np.frombuffer(lmdb_mod_txn.get((name+'_MMSE').encode()),dtype=np.float64)))
        scores.append(temp)
    
    return data,scores
        
def tranData(FT_DTI,DTI_labels,index):
    FT = []
    labels = []
    temp = np.array(FT_DTI)
    temp = temp.reshape((temp.shape[0]*temp.shape[1],temp.shape[2]))
    print(temp.shape)
    for i in range(temp.shape[0]):
#         print(temp[i],DTI_labels[int(i/index)])
        FT.append(temp[i])
        labels.append(DTI_labels[int(i/index)])
    
    return FT,labels
    
def subLabels(labels,number):
    subLabels = []
    for i in range(len(labels)):
        subLabels.append(labels[i][number-1])
#     print(np.array(subLabels).shape)
    return subLabels
    
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x    
    
## train and test set
def crossDataLabel(FT,labels):
    ## 5 cross valition
    test_size = 0.2
    randomseed=1234
    test_sub_num = len(FT)
    print('test_sub_num: ',test_sub_num)
    rs = np.random.RandomState(randomseed)
    train_sid, test_sid = train_test_split(range(test_sub_num), test_size=test_size, random_state=rs, shuffle=True)
    print('training on %d subjects, validating on %d subjects' % (len(train_sid), len(test_sid)))
    ####train set 
    fmri_data_train = [FT[i] for i in train_sid]
    trainLabels = pd.DataFrame(np.array([labels[i] for i in train_sid]))
#     print(type(trainLabels),'\n',trainLabels)
    ERP_train_dataset = ERP_matrix_datasets(fmri_data_train, trainLabels, isTrain='train')
    trainData = DataLoader(ERP_train_dataset)

    ####test set
    fmri_data_test = [FT[i] for i in test_sid]
    testLabels = pd.DataFrame(np.array([labels[i] for i in test_sid]))
#     print(type(testLabels),'\n',testLabels)
    ERP_test_dataset = ERP_matrix_datasets(fmri_data_test, testLabels, isTrain='test')
    testData = DataLoader(ERP_test_dataset)
    
    return trainData,trainLabels,testData,testLabels

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

##load data from array
class ERP_matrix_datasets(Dataset):
    ##build a new class for own dataset
    import numpy as np
    def __init__(self, fmri_data_matrix, label_matrix,
                 isTrain='train', transform=False):
        super(ERP_matrix_datasets, self).__init__()

        if not isinstance(fmri_data_matrix, np.ndarray):
            self.fmri_data_matrix = np.array(fmri_data_matrix)
        else:
            self.fmri_data_matrix = fmri_data_matrix
        
        self.Subject_Num = self.fmri_data_matrix.shape[0]
        self.Region_Num = self.fmri_data_matrix[0].shape[-1]

        if isinstance(label_matrix, pd.DataFrame):
            self.label_matrix = label_matrix
        elif isinstance(label_matrix, np.ndarray):
            self.label_matrix = pd.DataFrame(data=np.array(label_matrix))

        self.data_type = isTrain
        self.transform = transform

    def __len__(self):
        return self.Subject_Num

    def __getitem__(self, idx):
        #step1: get one subject data
        fmri_trial_data = self.fmri_data_matrix[idx]
        fmri_trial_data = fmri_trial_data.reshape(1,fmri_trial_data.shape[0])
        label_trial_data = np.array(self.label_matrix.iloc[idx])
#         print('fmri_trial_data\n{}\n======\nlabel_trial_data\n{}\n'.format(fmri_trial_data.shape,label_trial_data.shape))
        tensor_x = torch.stack([torch.FloatTensor(fmri_trial_data[ii]) for ii in range(len(fmri_trial_data))])  # transform to torch tensors
        tensor_y = torch.stack([torch.LongTensor([label_trial_data[ii]]) for ii in range(len(label_trial_data))])
#         print('tensor_x\n{}\n=======\ntensor_y\n{}\n'.format(tensor_x.size(),tensor_y.size()))
        return tensor_x, tensor_y

## get feature and labels from lmdb
def getLmbd(lmdb_dir,names_dir):
    FT = []
    labels = []
    lmdb_env = lmdb.open(lmdb_dir,readonly=True)
    names = readScores(names_dir)
    for name in names:
#         print(name)
        with lmdb_env.begin() as lmdb_mod_txn:
            temp = []
            if "T1.txt" in names_dir:
                FT.append(np.frombuffer(lmdb_mod_txn.get((name+'_RGB').encode()),dtype=np.float64))
            elif "DTI.txt" in names_dir:
                temp1 = []
                for j in range(65):
                    temp1.append(np.frombuffer(lmdb_mod_txn.get((name+'_DTI_Slice_'+str(j)).encode()),dtype=np.float64))
                FT.append(temp1)
                name = name+'_DTI'
            elif "Bold.txt" in names_dir:
                temp1 = []
                for j in range(240):
                    temp1.append(np.frombuffer(lmdb_mod_txn.get((name+'_Bold_Slice_'+str(j)).encode()),dtype=np.float64))
                FT.append(temp1)
                name = name+'_Bold'
            elif "ERP.txt" in names_dir:
                import re
                FT.append(np.frombuffer(lmdb_mod_txn.get((name+'_FTERP').encode()),dtype=np.float64))
                for j in re.findall(r'P(.*?)_ERP_Matrix_',name):
                    name =j
                for j in re.findall(r'H(.*?)_ERP_Matrix_',name):
                    name =j
            temp.append(float(np.frombuffer(lmdb_mod_txn.get((name+'_age').encode()),dtype=np.float64)))
            temp.append(float(np.frombuffer(lmdb_mod_txn.get((name+'_UPDRS').encode()),dtype=np.float64)))
            temp.append(float(np.frombuffer(lmdb_mod_txn.get((name+'_MMSE').encode()),dtype=np.float64)))
            labels.append(temp)
            
    print(np.array(labels).shape,np.array(FT).shape)        
    
    return FT,labels

## train and test model
def model_fit_evaluate(model,device,trainData,trainLabels,testData,testLabels,optimizer,loss_func,num_epochs=100):
    best_acc = 0 
    model_history={}
    model_history['train_MAE']=[];
    model_history['train_CS2']=[];
    model_history['train_CS5']=[];
    model_history['test_MAE']=[];
    model_history['test_CS2']=[];
    model_history['test_CS5']=[];

    for epoch in range(num_epochs):
        train_MAE,train_CS2,train_CS5 =train(model, device, trainData, trainLabels, optimizer,loss_func, epoch)
        model_history['train_MAE'].append(train_MAE)
        model_history['train_CS2'].append(train_CS2)
        model_history['train_CS5'].append(train_CS5)

        test_MAE,test_CS2,test_CS5 = test(model, device, testData, testLabels, loss_func)
        model_history['test_MAE'].append(test_MAE)
        model_history['test_CS2'].append(test_CS2)
        model_history['test_CS5'].append(test_CS5)
        if test_CS5 > best_acc:
            best_acc = test_CS5
            print("Model updated: Best-Acc = {:4f}".format(best_acc))

    print("best testing accuarcy:",best_acc)
#     plot_history(model_history)
    
##training the model
def train(model, device,train_loader, trainLabels, optimizer,loss_func, epoch):
    model.train()

    MAE = 0.0
    CS2 = 0.0
    CS5 = 0.0
    t0 = time.time()
    Predict_Scores = []
    True_Scores = []
    L1_MAE = []
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        data = data.squeeze(0)
        target = target.view(-1).float()
#         print('inputs ',data.size(),'labels:',target.size())
#         print('inputs ',data,'labels:',target)
        out = model(data)
        Predict_Scores.append(out),True_Scores.append(target)
        loss = loss_func(out,target)
        L1_MAE.append(loss)
        
        loss.backward()
        optimizer.step()
    num2,CS2 = LowerCount(L1_MAE,5)
    num5,CS5 = LowerCount(L1_MAE,8)
    MAE = sum(L1_MAE)/len(L1_MAE)   
    print("\nEpoch {}: \nTime Usage:{:4f} | Training MAE {:4f} | CS2 {:4f} | CS5 {:4f}".format(epoch,time.time()-t0,MAE,CS2,CS5))
    return MAE,CS2,CS5

def test(model, device, test_loader, testLabels, loss_func):
    model.eval()
    MAE = 0.0
    CS2 = 0.0
    CS5 = 0.0 
    ##no gradient desend for testing
    with torch.no_grad():
        Predict_Scores = []
        True_Scores = []
        L1_MAE = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
#             print('input:',target,'predict:',out)
            Predict_Scores.append(out),True_Scores.append(target)
            
            loss = loss_func(out,target)
            L1_MAE.append(loss)
#             print('MAE:',loss)
        num2,CS2 = LowerCount(L1_MAE,5)
        num5,CS5 = LowerCount(L1_MAE,8)
        MAE = sum(L1_MAE)/len(L1_MAE)
#         plotScatter(Predict_Scores,True_Scores) 
        
    return MAE,CS2,CS5    

def LowerCount(a,b):
    num = 0
    for i in a:
        if i<b:
            num+=1
    percent = num/len(a)
    return num,percent

def plotScatter(Predict_Scores,True_Scores):
    ## plot scatter for Predict_Scores,True_Scores
    import matplotlib.pyplot as plt
    print('Predict_Scores shape:',np.array(Predict_Scores).shape,'\nTrue_Scores shape:',np.array(True_Scores).shape)
    plt.figure(figsize=(10,10), dpi=100)
    plt.scatter(Predict_Scores,True_Scores)
    plt.show()

def plot_history(model_history):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.plot(model_history['train_CS2'], color='r')
    plt.plot(model_history['test_CS2'], color='b')
    plt.plot(model_history['train_CS5'], color='g')
    plt.plot(model_history['test_CS5'], color='y')
    plt.xlabel('Epochs')
    plt.ylabel('Prediction Accuracy')
    plt.legend(['Training_CS2', 'Validation_CS2','Training_CS5', 'Validation_CS5'])

    plt.subplot(122)
    plt.plot(model_history['train_MAE'], color='r')
    plt.plot(model_history['test_MAE'], color='b')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend(['Training', 'Validation'])
    plt.show()
    
## **********************************************************************************
## ************************************END Predictor*********************************
## **********************************************************************************