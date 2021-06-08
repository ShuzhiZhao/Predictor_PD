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