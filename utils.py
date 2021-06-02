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
# from model.tcanet import TCANet
# from model.backbone import ImgeT1Net,InceptionBoldNet,InceptionDTINet


## read txt filename
def readScores(name_dir):
    scores = []
    with open(name_dir, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline() 
            if not lines:
                break
                pass
            for i in lines.split():
                scores.append(i) 
            pass 
    return scores

## **********************************************************************************
## *******************************LOAD DATA(T1 Bold DTI and ERP)*********************
## **********************************************************************************
## load data and score of all subjects (256,288,3)
def loadT1(data_dir,age,UPDRS,MMSE):
#     names = []
    lmdb_dir = '/media/lhj/Momery/PD_predictDL/Data'
    env = lmdb.open(lmdb_dir+"/FT_InceptionResnetv4", map_size = int(1e12)*2)
    txn = env.begin(write=True)
    f=open(lmdb_dir+'/name_T1.txt',mode='w')
    for file in os.listdir(data_dir):
        data_T1 = {}
#         T1_label = {}        
#         print("++++++++++++++++++++++ The process of "+file+" ++++++++++++++++++++++")
        myFile = os.fsencode(file)
        myFile = myFile.decode('utf-8')
        imge = Image.open((data_dir+'/'+myFile))
        arr = np.array(imge)
        f.write(file.replace('.jpg','')+'\n')
#         names.append(file.replace('.jpg',''))
        data_T1[file.replace('.jpg','')+'_RGB'] = []
        data_T1[file.replace('.jpg','')+'_RGB'].append(arr)      
        
#         print(int(file.replace('.jpg','')[0:5]),age[int(file.replace('.jpg','')[0:5])-1])        
#         T1_label[file.replace('.jpg','')+'_age'] = []
#         T1_label[file.replace('.jpg','')+'_age'].append(age[int(file.replace('.jpg','')[0:5])-1])
#         T1_label[file.replace('.jpg','')+'_UPDRS'] = []
#         T1_label[file.replace('.jpg','')+'_UPDRS'].append(UPDRS[int(file.replace('.jpg','')[0:5])-1])
#         T1_label[file.replace('.jpg','')+'_MMSE'] = []
#         T1_label[file.replace('.jpg','')+'_MMSE'].append(MMSE[int(file.replace('.jpg','')[0:5])-1])  
        
        FT = FeatureT1(data_T1,[file.replace('.jpg','')+'_RGB'])     ## get Feature after Inception-Resnetv4 
        ## store FT age UPDRS and MMSE in lmbd        
        txn.put((file.replace('.jpg','')+'_RGB').encode(), np.array(FT))
#         print(float(age[int(file.replace('.jpg','')[0:5])-1]))
        txn.put((file.replace('.jpg','')+'_age').encode(), np.array([float(age[int(file.replace('.jpg','')[0:5])-1])]))
        txn.put((file.replace('.jpg','')+'_UPDRS').encode(), np.array([float(UPDRS[int(file.replace('.jpg','')[0:5])-1])]))
        txn.put((file.replace('.jpg','')+'_MMSE').encode(), np.array([float(MMSE[int(file.replace('.jpg','')[0:5])-1])]))
        
#     txn.put(('sum_names').encode(), np.array(names))
    txn.commit() 
    env.close()
    f.close()
    print("++++++++++++++++++++++ Finish the process of Inceptions-Resnetv4 for all T1 jpg ++++++++++++++++++++++")

def FeatureT1(data_T1,names):
    FT = []
    ## model
    model = Inceptionv4()
    model = model.cuda()
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for i in names:
#         print("++++++++++++++++++++++ The process of "+i+" ++++++++++++++++++++++")
        inputs = torch.FloatTensor(np.array(data_T1[i]).transpose((0,3,2,1))).to(device)
        FT = model(inputs).tolist()
#         print(np.array(FT).shape,'\n',np.array(FT))
#         print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return FT    
    
## load data and score of all subjects (64,64,32,240)
def loadBold(data_dir,age,UPDRS,MMSE):
#     names = []
    lmdb_dir = '/media/lhj/Momery/PD_predictDL/Data'
#     env1 = lmdb.open(lmdb_dir+"/FT_TCAN", map_size = int(1e12)*2)
#     txn1 = env1.begin(write=True)
    f=open(lmdb_dir+'/name_Bold.txt',mode='w')
    for file in os.listdir(data_dir):
        data_T1 = {}
        T1_label = {}
#         print("++++++++++++++++++++++ The process of "+file+" ++++++++++++++++++++++")
        myFile = os.fsencode(file)
        myFile = myFile.decode('utf-8')
        myNifti = nib.load((data_dir+'/'+myFile))
        data = myNifti.get_fdata()
        data = data*(185.0/np.percentile(data, 97))
        arr = np.ascontiguousarray(data)
        f.write(file.replace('.nii.gz','')+'\n')
        data_T1[file.replace('.nii.gz','')+'_Bold'] = []
        data_T1[file.replace('.nii.gz','')+'_Bold'].append(arr)
        age_temp = [float(age[int(file.replace('.nii.gz','')[3:8])-1])]
        UPDRS_temp = [float(UPDRS[int(file.replace('.nii.gz','')[3:8])-1])]
        MMSE_temp = [float(MMSE[int(file.replace('.nii.gz','')[3:8])-1])]
        ## get and store Feature after Temporal Convolutional Attention-based Network
        FeatureBold(data_T1,[file.replace('.nii.gz','')+'_Bold'],age_temp,UPDRS_temp,MMSE_temp)     
        
#         print(int(file.replace('.jpg','')[3:8]),age[int(file.replace('.jpg','')[3:8])-1])        
#         T1_label[file.replace('.nii.gz','')+'_age'] = []
#         T1_label[file.replace('.nii.gz','')+'_age'].append(age[int(file.replace('.jpg','')[3:8])-1])        
#         T1_label[file.replace('.nii.gz','')+'_UPDRS'] = []
#         T1_label[file.replace('.nii.gz','')+'_UPDRS'].append(UPDRS[int(file.replace('.jpg','')[3:8])-1])
#         T1_label[file.replace('.nii.gz','')+'_MMSE'] = []
#         T1_label[file.replace('.nii.gz','')+'_MMSE'].append(MMSE[int(file.replace('.jpg','')[3:8])-1])
        
#         txn1.put((file.replace('.nii.gz','')+'_age').encode(), np.array([float(age[int(file.replace('.nii.gz','')[3:8])-1])]))
#         txn1.put((file.replace('.nii.gz','')+'_UPDRS').encode(), np.array([float(UPDRS[int(file.replace('.nii.gz','')[3:8])-1])]))
#         txn1.put((file.replace('.nii.gz','')+'_MMSE').encode(), np.array([float(MMSE[int(file.replace('.nii.gz','')[3:8])-1])]))
        
#     txn.put(('sum_names').encode(), np.array(names))
#     txn1.commit() 
#     env1.close()
    f.close()
    print("++++++++++++Finish the store of Temporal Convolutional Attention-based Network for all Bold +++++++++++")    
    
def FeatureBold(data,names,age_temp,UPDRS_temp,MMSE_temp):
    from model.tcanet import TCANet
    ## model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ## 'emb_size', 'input_output_size', 'num_channels', 'seq_len', 'num_sub_blocks', 'temp_attn', 'nheads', 'en_res', 'conv', and 'key_size'
#     model = TCANet(emb_size=5, input_output_size=64*64*32*240, num_channels=[4,4,2], seq_len=240, num_sub_blocks=5, temp_attn=5, nheads=5, en_res=5,conv=3, key_size=3)
    model = SpikingNet(device, n_time_steps=128, begin_eval=0)
    model = model.cuda()
    model.eval()
    lmdb_dir = '/media/lhj/Momery/PD_predictDL/Data'
    env = lmdb.open(lmdb_dir+"/FT_TCAN", map_size = int(1e12)*2)
    txn = env.begin(write=True)
    for i in names:        
        print("++++++++++++++++++++++ The process of "+i+" ++++++++++++++++++++++")
        txn.put((i+'_age').encode(), np.array(age_temp))
        txn.put((i+'_UPDRS').encode(), np.array(UPDRS_temp))        
        txn.put((i+'_MMSE').encode(), np.array(MMSE_temp))        
        for j in range(240):
            FT = []
            inputs = torch.FloatTensor(np.array(data[i])[:,:,:,:,j:j+1]).to(device)
#             print(inputs.size())
            FT = model(inputs).tolist()
            txn.put((i+'_Slice_'+str(j)).encode(), np.array(FT))    
#             print(np.array(FT).shape,j,'\n',FT)
    txn.commit() 
    env.close()
#     print("++++++++++++Finish the store of Temporal Convolutional Attention-based Network for one Bold +++++++++++")

## load data and score of all subjects (128,128,75,65)
def loadDTI(data_dir,age,UPDRS,MMSE):
#     names = []
    lmdb_dir = '/media/lhj/Momery/PD_predictDL/Data'
#     env1 = lmdb.open(lmdb_dir+"/FT_SNN", map_size = int(1e12)*2)
#     txn1 = env1.begin(write=True)
    f=open(lmdb_dir+'/name_DTI.txt',mode='w')
    for file in os.listdir(data_dir):
        data_T1 = {}
#         T1_label = {}
#         print("++++++++++++++++++++++ The process of "+file+" ++++++++++++++++++++++")
#         names.append(file.replace('.nii.gz',''))
        myFile = os.fsencode(file)
        myFile = myFile.decode('utf-8')
        myNifti = nib.load((data_dir+'/'+myFile))
        data = myNifti.get_fdata()
        data = data*(185.0/np.percentile(data, 97))
        arr = np.ascontiguousarray(data)
#         print(arr.shape)
        f.write(file.replace('.nii.gz','')+'\n')
        data_T1[file.replace('.nii.gz','')+'_DTI'] = []
        data_T1[file.replace('.nii.gz','')+'_DTI'].append(arr)
        age_temp = [float(age[int(file.replace('.nii.gz','')[0:5])-1])]
        UPDRS_temp = [float(UPDRS[int(file.replace('.nii.gz','')[0:5])-1])]
        MMSE_temp = [float(MMSE[int(file.replace('.nii.gz','')[0:5])-1])]
        ## get Feature and store after Spiking Neural Network
        FeatureDTI(data_T1,[file.replace('.nii.gz','')+'_DTI'],age_temp,UPDRS_temp,MMSE_temp)            
        
#         print(int(file.replace('.jpg','')[0:5]),age[int(file.replace('.jpg','')[0:5])-1])        
#         T1_label[file.replace('.nii.gz','')+'_age'] = []
#         T1_label[file.replace('.nii.gz','')+'_age'].append(age[int(file.replace('.jpg','')[0:5])-1])        
#         T1_label[file.replace('.nii.gz','')+'_UPDRS'] = []
#         T1_label[file.replace('.nii.gz','')+'_UPDRS'].append(UPDRS[int(file.replace('.jpg','')[0:5])-1])
#         T1_label[file.replace('.nii.gz','')+'_MMSE'] = []
#         T1_label[file.replace('.nii.gz','')+'_MMSE'].append(MMSE[int(file.replace('.jpg','')[0:5])-1])       
        
#         txn.put((file.replace('.jpg','')+'_DTI').encode(), np.array(FT))
#         print(float(age[int(file.replace('.nii.gz','')[0:5])-1]))
#         txn1.put((file.replace('.nii.gz','')+'_age').encode(), np.array([float(age[int(file.replace('.nii.gz','')[0:5])-1])]))
#         txn1.put((file.replace('.nii.gz','')+'_UPDRS').encode(), np.array([float(UPDRS[int(file.replace('.nii.gz','')[0:5])-1])]))
#         txn1.put((file.replace('.nii.gz','')+'_MMSE').encode(), np.array([float(MMSE[int(file.replace('.nii.gz','')[0:5])-1])]))
        
#     txn.put(('sum_names').encode(), np.array(names))
#     txn1.commit() 
#     env1.close() 
    f.close()
    print("++++++++++++++++++++++ Finish the process of Spiking Neural Network for all DTI ++++++++++++++++++++++")
        

def FeatureDTI(data_DTI,names,age_temp,UPDRS_temp,MMSE_temp):
    ## model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SpikingNet(device, n_time_steps=128, begin_eval=0)
    model = model.cuda()
    model.eval()
    lmdb_dir = '/media/lhj/Momery/PD_predictDL/Data'
    env = lmdb.open(lmdb_dir+"/FT_SNN", map_size = int(1e12)*2)
    txn = env.begin(write=True)
    for i in names:
        print("++++++++++++++++++++++ The process of "+i+" ++++++++++++++++++++++")
        txn.put((i+'_age').encode(), np.array(age_temp))
        txn.put((i+'_UPDRS').encode(), np.array(UPDRS_temp))        
        txn.put((i+'_MMSE').encode(), np.array(MMSE_temp))                        
        for j in range(65):
            FT = []
            inputs = torch.FloatTensor(np.array(data_DTI[i])[:,:,:,:,j:j+1]).to(device)
    #         print(inputs.size())
            FT = model(inputs).tolist()
            txn.put((i+'_Slice_'+str(j)).encode(), np.array(FT))    
#             print(np.array(FT).shape,j,'\n',FT)
    txn.commit() 
    env.close()
#     print("++++++++++++++++++++++ Finish the store of Spiking Neural Network for one DTI +++++++++++++++++++++")

## load data from lmdb and get feature from TCAN, then store feature to lmdb
def loadERP(ERP_dir,age,UPDRS,MMSE):
    lmdb_env = lmdb.open(ERP_dir,readonly=True)
    f=open('/media/lhj/Momery/PD_predictDL/Data/name_ERP.txt',mode='w')
    with lmdb_env.begin() as lmdb_mod_txn:
        mod_cursor = lmdb_mod_txn.cursor()
        for idx,(key, value) in enumerate(mod_cursor.iterprev()):
            key = str(key, encoding='utf-8')
            f.write(key+'\n')
#             print("++++++++++++++++++++++ The process of "+key+" ++++++++++++++++++++++")
            Matrix = np.frombuffer(value,dtype=np.float64)
            Matrix = Matrix.reshape(65,350,2)
            index = 0
            FeatureERP(Matrix,key,age,UPDRS,MMSE,index) 
    f.close()
    lmdb_env.close()
    print("++++++++++++++++++++++ Finish the process of Spiking Neural Network for all ERP ++++++++++++++++++++++")

def FeatureERP(Matrix,key,age,UPDRS,MMSE,index):
    import re
    FT = []
    ## lmdb
    lmdb_dir = '/media/lhj/Momery/PD_predictDL/Data'
    env = lmdb.open(lmdb_dir+"/FT_ERP", map_size = int(1e12)*2)
    txn = env.begin(write=True)
    ## model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model = TCANet(emb_size=3, input_output_size=1*65*700, num_channels=[65], seq_len=700, num_sub_blocks=3, temp_attn=3, nheads=3, en_res=3,conv=3, key_size=3)
    model = TemporalConvNet(num_inputs=350,num_channels=[65])
    model = model.cuda()
    model.eval()
    inputs = torch.FloatTensor(Matrix).to(device)
#     print(inputs.size())                              
    FT = model(inputs).tolist()
#     print(np.array(FT).shape,'\n',FT)
    txn.put((key+'_FTERP').encode(), np.array(FT))
#     print(type(re.findall(r'P(.*?)_ERP_Matrix_',key)))
    for i in re.findall(r'P(.*?)_ERP_Matrix_',key):
        age_temp = [float(age[int(i)-index])]
        UPDRS_temp = [float(UPDRS[int(i)-index])]
        MMSE_temp = [float(MMSE[int(i)-index])]
        txn.put((i+'_age').encode(), np.array(age_temp))
        txn.put((i+'_UPDRS').encode(), np.array(UPDRS_temp))        
        txn.put((i+'_MMSE').encode(), np.array(MMSE_temp))    
    txn.commit()
    env.close()
    
## **********************************************************************************
## *****************************END LOAD DATA(T1 Bold DTI and ERP)*******************
## **********************************************************************************    



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
    index = 3 
    nameLabel = ['age', 'UPDRS', 'MMSE']
    model_test1 = Net(n_feature=np.array(FT_T1).shape[1], n_hidden=200, n_output=index)
    model_test2 = Net(n_feature=np.array(FT_ERP).shape[1], n_hidden=1000, n_output=index)
    model_test3 = Net(n_feature=np.array(FT_DTI).shape[1], n_hidden=50, n_output=index)
    model_test4 = Net(n_feature=np.array(FT_Bold).shape[1], n_hidden=50, n_output=index)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     device = torch.device('cpu')
    model_test1 = model_test1.to(device)
    model_test2 = model_test2.to(device)
    model_test3 = model_test3.to(device)
    model_test4 = model_test4.to(device)
    print("{} paramters to be trained in the model\n".format(count_parameters(model_test1)))
    optimizer1 = optim.Adam(model_test1.parameters(),lr=0.001, weight_decay=5e-4)
    optimizer2 = optim.Adam(model_test2.parameters(),lr=0.001, weight_decay=5e-4)
    optimizer3 = optim.Adam(model_test3.parameters(),lr=0.001, weight_decay=5e-4)
    optimizer4 = optim.Adam(model_test4.parameters(),lr=0.001, weight_decay=5e-4)
    loss_func = nn.L1Loss(reduction='mean')
    num_epochs=15
    print('******************************************'+nameLabel[index-1]+'************************************************')
    model_fit_evaluate(model_test1,device,trainData_T1,trainLabels_T1,testData_T1,testLabels_T1,optimizer1,loss_func,num_epochs)
    print("{} paramters to be trained in the model\n".format(count_parameters(model_test2)))
    model_fit_evaluate(model_test2,device,trainData_ERP,trainLabels_ERP,testData_ERP,testLabels_ERP,optimizer2,loss_func,num_epochs)
    print("{} paramters to be trained in the model\n".format(count_parameters(model_test3)))
    model_fit_evaluate(model_test3,device,trainData_DTI,trainLabels_DTI,testData_DTI,testLabels_DTI,optimizer3,loss_func,num_epochs)
    print("{} paramters to be trained in the model\n".format(count_parameters(model_test4)))
    model_fit_evaluate(model_test4,device,trainData_Bold,trainLabels_Bold,testData_Bold,testLabels_Bold,optimizer4,loss_func,num_epochs)
    #############################################
    ################## fusion mode ##############
    #############################################
    fusionFT(DTI_dir,DTI_names,Bold_dir,Bold_names)

##  FT1_dir:lmdb dir for FT1; FT1_names:txt dir for FT1   
def fusionFT(FT1_lmdb,name1_dir,FT2_lmdb,name2_dir):
    fusFT = []
    fusLabels = []
    FT1_names = readScores(name1_dir)
    FT2_names = readScores(name2_dir)
    conNames = finConNames(FT1_names,name1_dir,FT2_names,name2_dir,)        
    
    return fusFT,fusLabels

def finConNames(FT1_names,name1_dir,FT2_names,name2_dir,):
    conNames = []
    if "T1.txt" in names_dir:
                
    elif "DTI.txt" in names_dir:
        
    elif "Bold.txt" in names_dir:
        
    elif "ERP.txt" in names_dir:

    return conNames    
        
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