from numpy import linalg as LA
import numpy as np
import pandas as pd
import scipy.io as scio
from scipy import sparse
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
import nibabel as nib
import os
import math
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from PIL import Image
from model.inceptionv4 import Inceptionv4
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lmdb
from model.SNN import SpikingNet
from sklearn.model_selection import cross_val_score, train_test_split,ShuffleSplit
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import *


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
def loadT1(data_dir,Score):
#     names = []
    lmdb_dir = '/media/lhj/Momery/PD_predictDL/Data/PPMI'
    env = lmdb.open(lmdb_dir+"/FT_InceptionResnetv4", map_size = int(1e12)*2)
    txn = env.begin(write=True)
    f=open(lmdb_dir+'/2Dname.txt',mode='w')
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
        
        FT = FeatureT1(data_T1,[file.replace('.jpg','')+'_RGB'])     ## get Feature after Inception-Resnetv4 
        ## store FT Score      
        txn.put((file.replace('.jpg','')+'_RGB').encode(), np.array(FT))
#         print(float(age[int(file.replace('.jpg','')[0:5])-1]))
        txn.put((file.replace('.jpg','')+'_score').encode(), np.array([float(Score[int(file.replace('.jpg','')[0:5])-1])]))
        
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
def loadBold(data_dir,Score):
    lmdb_dir = '/media/lhj/Momery/PD_predictDL/Data/PPMI'
    f=open(lmdb_dir+'/name_Bold.txt',mode='w')
    for file in os.listdir(data_dir):
        data_T1 = {}
        T1_label = {}
        print("++++++++++++++++++++++ The process of "+file+" ++++++++++++++++++++++")
        myFile = os.fsencode(file)
        myFile = myFile.decode('utf-8')
        myNifti = nib.load((data_dir+'/'+myFile))
        data = myNifti.get_fdata()
        data = data*(185.0/np.percentile(data, 97))
        print(data.shape)
        if data.shape[0] == 68 and data.shape[1] == 66 and data.shape[2] == 40 and data.shape[3] == 210 :
            arr = np.ascontiguousarray(data)
            f.write(file.replace('.nii.gz','')+'\n')
            data_T1[file.replace('.nii.gz','')+'_Bold'] = []
            data_T1[file.replace('.nii.gz','')+'_Bold'].append(arr)
            Score_temp = [float(Score[int(file.replace('.nii.gz','')[0:5])-1])]
            ## get and store Feature after Temporal Convolutional Attention-based Network
            FeatureBold(data_T1,[file.replace('.nii.gz','')+'_Bold'],Score_temp)     
        
    f.close()
    print("++++++++++++Finish the store of Temporal Convolutional Attention-based Network for all Bold +++++++++++")    
    
def FeatureBold(data,names,Score):
    ## model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SpikingNet(device, n_time_steps=128, begin_eval=0)
    model = model.cuda()
    model.eval()
    lmdb_dir = '/media/lhj/Momery/PD_predictDL/Data/PPMI'
    env = lmdb.open(lmdb_dir+"/FT_TCAN", map_size = int(1e12)*2)
    txn = env.begin(write=True)
    for i in names:        
        print("++++++++++++++++++++++ The process of "+i+" ++++++++++++++++++++++")
        txn.put((i+'_score').encode(), np.array(Score))      
        for j in range(210):
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
def loadDTI(data_dir,Score):
    lmdb_dir = '/media/lhj/Momery/PD_predictDL/Data/PPMI'
    f=open(lmdb_dir+'/name_DTI.txt',mode='w')
    for file in os.listdir(data_dir):
        print("++++++++++++++++++++++ The process of "+file+" ++++++++++++++++++++++")
        data_T1 = {}
        myFile = os.fsencode(file)
        myFile = myFile.decode('utf-8')
        myNifti = nib.load((data_dir+'/'+myFile))
        data = myNifti.get_fdata()
        data = data*(185.0/np.percentile(data, 97))
        print(data.shape)
        if len(data.shape)== 4 :
            if data.shape[0] == 116 and data.shape[1] == 116 and data.shape[2] == 72 and data.shape[3] == 65 :
                arr = np.ascontiguousarray(data)
                f.write(file.replace('.nii.gz','')+'\n')
                data_T1[file.replace('.nii.gz','')+'_DTI'] = []
                data_T1[file.replace('.nii.gz','')+'_DTI'].append(arr)
                score_temp = [float(Score[int(file.replace('.nii.gz','')[0:5])-1])]
                ## get Feature and store after Spiking Neural Network
                FeatureDTI(data_T1,[file.replace('.nii.gz','')+'_DTI'],score_temp)            
        
    f.close()
    print("++++++++++++++++++++++ Finish the process of Spiking Neural Network for all DTI ++++++++++++++++++++++")
          
    
def FeatureDTI(data_DTI,names,score_temp):
    ## model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SpikingNet(device, n_time_steps=128, begin_eval=0)
    model = model.cuda()
    model.eval()
    lmdb_dir = '/media/lhj/Momery/PD_predictDL/Data/PPMI'
    env = lmdb.open(lmdb_dir+"/FT_SNN", map_size = int(1e12)*2)
    txn = env.begin(write=True)
    for i in names:
        print("++++++++++++++++++++++ The process of "+i+" ++++++++++++++++++++++")
        txn.put((i+'_score').encode(), np.array(score_temp))                       
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

def Prediction(work_dir):
    conName_dir = work_dir+'/conNames.txt'
    T1_dir =  work_dir+'/FT_InceptionResnetv4'
    Bold_dir =  work_dir+'/FT_TCAN'
    DTI_dir =  work_dir+'/FT_SNN'
    fusFT_dir =  work_dir+'/FT_MulFus'
    # fusion for Three different MRI
    fusFT,fusLabels = fusionFT(work_dir,T1_dir,'T1',DTI_dir,'DTI',Bold_dir,'Bold',conName_dir)    
#     preFusFT(fusFT_dir,conName_dir)
    # fusion for Two different MRI
#     conName_dir = work_dir+'/name_DTI.txt'
#     fusFT2,fusLabels2 = fusionFT2(work_dir,DTI_dir,'DTI',Bold_dir,'Bold',conName_dir)
#     fusFT2,fusLabels2 = fusionFT2(work_dir,T1_dir,'T1',Bold_dir,'Bold',conName_dir)
#     fusFT2,fusLabels2 = fusionFT2(work_dir,T1_dir,'T1',DTI_dir,'DTI',conName_dir)
#     preFusFT2(fusFT_dir,conName_dir)

def preFusFT(fusFT_dir,conName_dir):
    names = readScores(conName_dir)
    print('*********** start get FT1 FT2 FT3 fusFT scores **************')
    FT1,scores1,FT2,scores2,FT3,scores3,fusFT,scores4 = getLmdbFus(fusFT_dir,'T1','DTI','Bold',names)
    print('***************** fusFT predictor for Score *****************')
    predLinDL(FT3,scores3)
    
def preFusFT2(fusFT_dir,conName_dir):
    names = readScores(conName_dir)
    print('*********** start get Two fusFT scores **************')
    fusFT,scores = getLmdbFus2(fusFT_dir,'DTI','Bold',names)
    print('***************** fusFT predictor for Score *****************')
    predLinDL(fusFT,scores)    
    
def predLinDL(FT,scores):
    model_test = Net(n_feature=np.array(FT).shape[1], n_hidden=200, n_output=1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_test = model_test.to(device)
    optimizer = optim.Adam(model_test.parameters(),lr=0.001, weight_decay=5e-4)
    loss_func = nn.L1Loss(reduction='mean')
    num_epochs = 8
    trainData,trainLabels,testData,testLabels = crossDataLabel(FT,scores)
    print('****************************************** Train for PPMI ************************************************')
    model_fit_evaluate(model_test,device,trainData,trainLabels,testData,testLabels,optimizer,loss_func,num_epochs)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.dropout(F.relu(self.hidden(x)))      # activation function for hidden layer
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

def getLmdbFus(fusFT_dir,FT1_Type,FT2_Type,FT3_Type,names):
    FT1 = []
    score1 = []
    FT2 = []
    score2 = []
    FT3 = []
    score3 = []
    fusFT = []
    score4 = []
    lmdb_env = lmdb.open(fusFT_dir+FT1_Type+FT2_Type+FT3_Type,readonly=True)
    with lmdb_env.begin() as lmdb_mod_txn:
        mod_cursor = lmdb_mod_txn.cursor()
        for idx,(key, value) in enumerate(mod_cursor.iterprev()):
            key = str(key, encoding='utf-8') 
            for ii in names:
                if ii in key :
                    if '_FT1' in key:
                        FT1.append(np.frombuffer(value).tolist())
                    if '_FT2' in key:
                        FT2.append(np.frombuffer(value).tolist())
                    if '_FT3' in key:
                        FT3.append(np.frombuffer(value).tolist())
                    if '_fusFTs' in key:
                        fusFT.append(np.frombuffer(value).tolist())
                    if '_scores' in key and FT1_Type in key and FT2_Type in key and FT3_Type in key :
                        score4.append(np.frombuffer(value).tolist())
                    if '_scores' in key and FT2_Type+'0'+FT3_Type+'0' in key :
                        score1.append(np.frombuffer(value).tolist())
                    if '_scores' in key and FT1_Type+'0' in key and FT3_Type+'0' in key :
                        score2.append(np.frombuffer(value).tolist())
                    if '_scores' in key and FT1_Type+'0' in key and FT2_Type+'0' in key :
                        score3.append(np.frombuffer(value).tolist())
    print('fusFT: ',np.array(FT1).shape,np.array(score1).shape,np.array(FT2).shape,np.array(score2).shape,np.array(FT3).shape,np.array(score3).shape,np.array(fusFT).shape,np.array(score4).shape)            
    return FT1,score1,FT2,score2,FT3,score3,fusFT,score4

def getLmdbFus2(fusFT_dir,FT1_Type,FT2_Type,names):
    fusFT = []
    score = []
    lmdb_env = lmdb.open(fusFT_dir+FT1_Type+FT2_Type,readonly=True)
    with lmdb_env.begin() as lmdb_mod_txn:
        mod_cursor = lmdb_mod_txn.cursor()
        for idx,(key, value) in enumerate(mod_cursor.iterprev()):
            key = str(key, encoding='utf-8') 
            for ii in names:
                if ii in key :
                    if '_fusFTs' in key:
                        fusFT.append(np.frombuffer(value).tolist())
                    if '_scores' in key and FT1_Type in key and FT2_Type in key :
                        score.append(np.frombuffer(value).tolist())
    print('fusFT: ',np.array(fusFT).shape,np.array(score).shape)            
    return fusFT,score

def fusionFT(work_dir,FT1_lmdb,FT1_Type,FT2_lmdb,FT2_Type,FT3_lmdb,FT3_Type,conName_dir):
    from model.MulfusNet import fusNet
    fusFT = []
    fusLabels = []
    lmdb_dir = work_dir+'/FT_MulFus'
    env = lmdb.open(lmdb_dir+FT1_Type+FT2_Type+FT3_Type, map_size = int(1e12)*2)
    txn = env.begin(write=True)    
    names = readScores(conName_dir)
    ## for subjects
    for ii in names:
        ## is or not exit in lmdb
        print(isExitName(FT1_lmdb,ii,FT1_Type),isExitName(FT2_lmdb,ii,FT2_Type),isExitName(FT3_lmdb,ii,FT3_Type))
        if isExitName(FT1_lmdb,ii,FT1_Type) and isExitName(FT2_lmdb,ii,FT2_Type) and isExitName(FT3_lmdb,ii,FT3_Type):
            print('************** Con exit for '+ii+' in three modal +++++++++++++++++')
            ## get FT1 and FT2 with 3D from lmdb
            FT1_data,FT1_scores = get3DFT(FT1_lmdb,ii,FT1_Type)
            FT2_data,FT2_scores = get3DFT(FT2_lmdb,ii,FT2_Type)
            FT3_data,FT3_scores = get3DFT(FT3_lmdb,ii,FT3_Type)
            if len(np.array(FT1_data).shape) == 3 and len(np.array(FT2_data).shape) == 3 and len(np.array(FT3_data).shape) == 3 :
                if np.array(FT1_data).shape[2] != np.array(FT2_data).shape[2]:
                    m = nn.Conv1d(np.array(FT1_data).shape[2], np.array(FT2_data).shape[2], 1, stride=1)
                    FT1_data = m(torch.FloatTensor(FT1_data).transpose(1,2)).transpose(1,2).tolist()
                if np.array(FT3_data).shape[2] != np.array(FT2_data).shape[2]:
                    m = nn.Conv1d(np.array(FT1_data).shape[2], np.array(FT2_data).shape[2], 1, stride=1)
                    FT3_data = m(torch.FloatTensor(FT3_data).transpose(1,2)).transpose(1,2).tolist() 
                if  np.array(FT1_data).shape[1] > 5 :
                    output1 = int((np.array(FT1_data).shape[1])/5)
                    m = nn.Conv1d(np.array(FT1_data).shape[1], output1, 1, stride=1)
                    FT1_data = m(torch.FloatTensor(FT1_data)).tolist()
                if  np.array(FT2_data).shape[1] > 5 :
                    output1 = int((np.array(FT2_data).shape[1])/5)
                    m = nn.Conv1d(np.array(FT2_data).shape[1], output1, 1, stride=1)
                    FT2_data = m(torch.FloatTensor(FT2_data)).tolist() 
                if  np.array(FT3_data).shape[1] > 5 :
                    output1 = int((np.array(FT3_data).shape[1])/5)
                    m = nn.Conv1d(np.array(FT3_data).shape[1], output1, 1, stride=1)
                    FT3_data = m(torch.FloatTensor(FT3_data)).tolist()    
                print(np.array(FT1_data).shape,np.array(FT1_scores).shape,np.array(FT2_data).shape,np.array(FT2_scores).shape,np.array(FT3_data).shape,np.array(FT3_scores).shape)
        #         print(FT1_scores,FT2_scores,FT3_scores)
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                mod = fusNet().to(device)
                for jj in range(np.array(FT1_data).shape[1]):
                    for gg in range(np.array(FT2_data).shape[1]):
                        for ff in range(np.array(FT3_data).shape[1]):
                            f = mod(torch.cuda.FloatTensor(FT1_data[0][jj]).view(1,1,np.array(FT1_data).shape[2]),torch.cuda.FloatTensor(FT2_data[0][gg]).view(1,1,np.array(FT2_data).shape[2]),torch.cuda.FloatTensor(FT3_data[0][ff]).view(1,1,np.array(FT3_data).shape[2]))
                            txn.put((ii+'_'+FT1_Type+str(jj)+'_FT1').encode(), np.array(f[0].tolist()))
                            txn.put((ii+'_'+FT2_Type+str(gg)+'_FT2').encode(), np.array(f[1].tolist()))
                            txn.put((ii+'_'+FT3_Type+str(ff)+'_FT3').encode(), np.array(f[2].tolist()))
                            txn.put((ii+'_'+FT1_Type+str(jj)+FT2_Type+str(gg)+FT3_Type+str(ff)+'_fusFTs').encode(), np.array(f[3].tolist()))
                            txn.put((ii+'_'+FT1_Type+str(jj)+FT2_Type+str(gg)+FT3_Type+str(ff)+'_scores').encode(), np.array(FT1_scores))
    
    txn.commit() 
    env.close()
    return fusFT,fusLabels

def fusionFT2(work_dir,FT1_lmdb,FT1_Type,FT2_lmdb,FT2_Type,conName_dir):
    from model.fusNet import fusNet
    fusFT = []
    fusLabels = []
    lmdb_dir = work_dir+'/FT_MulFus'
    env = lmdb.open(lmdb_dir+FT1_Type+FT2_Type, map_size = int(1e12)*2)
    txn = env.begin(write=True)    
    names = readScores(conName_dir)
    ## for subjects
    for ii in names:
        ## is or not exit in lmdb
        print(isExitName(FT1_lmdb,ii,FT1_Type),isExitName(FT2_lmdb,ii,FT2_Type))
        if isExitName(FT1_lmdb,ii,FT1_Type) and isExitName(FT2_lmdb,ii,FT2_Type) :
            print('************** Con exit for '+ii+' in Two modal +++++++++++++++++')
            ## get FT1 and FT2 with 3D from lmdb
            FT1_data,FT1_scores = get3DFT(FT1_lmdb,ii,FT1_Type)
            FT2_data,FT2_scores = get3DFT(FT2_lmdb,ii,FT2_Type)
            if len(np.array(FT1_data).shape) == 3 and len(np.array(FT2_data).shape) == 3:
                if np.array(FT1_data).shape[2] != np.array(FT2_data).shape[2]:
                    m = nn.Conv1d(np.array(FT1_data).shape[2], np.array(FT2_data).shape[2], 1, stride=1)
                    FT1_data = m(torch.FloatTensor(FT1_data).transpose(1,2)).transpose(1,2).tolist()
                if  np.array(FT1_data).shape[1] > 5 :
                    output1 = int((np.array(FT1_data).shape[1])/5)
                    m = nn.Conv1d(np.array(FT1_data).shape[1], output1, 1, stride=1)
                    FT1_data = m(torch.FloatTensor(FT1_data)).tolist()
                if  np.array(FT2_data).shape[1] > 5 :
                    output1 = int((np.array(FT2_data).shape[1])/5)
                    m = nn.Conv1d(np.array(FT2_data).shape[1], output1, 1, stride=1)
                    FT2_data = m(torch.FloatTensor(FT2_data)).tolist()   
                print(np.array(FT1_data).shape,np.array(FT1_scores).shape,np.array(FT2_data).shape,np.array(FT2_scores).shape)
        #         print(FT1_scores,FT2_scores,FT3_scores)
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                mod = fusNet().to(device)
                for jj in range(np.array(FT1_data).shape[1]):
                    for gg in range(np.array(FT2_data).shape[1]):
                        f = mod(torch.cuda.FloatTensor(FT1_data[0][jj]).view(1,1,np.array(FT1_data).shape[2]),torch.cuda.FloatTensor(FT2_data[0][gg]).view(1,1,np.array(FT2_data).shape[2]))
                        txn.put((ii+'_'+FT1_Type+str(jj)+FT2_Type+str(gg)+'_fusFTs').encode(), np.array(f[2].tolist()))

                        txn.put((ii+'_'+FT1_Type+str(jj)+FT2_Type+str(gg)+'_scores').encode(), np.array(FT1_scores))
    
    txn.commit() 
    env.close()
    return fusFT,fusLabels

def get3DFT(FT1_lmdb,name,dataType):
    data = []
    scores = []
    print(FT1_lmdb,name,dataType)
    lmdb_env = lmdb.open(FT1_lmdb,readonly=True)
    with lmdb_env.begin() as lmdb_mod_txn:
        temp1 = []
        age = []
        UPDRS = []
        MMSE = []
        mod_cursor = lmdb_mod_txn.cursor()
        for idx,(key, value) in enumerate(mod_cursor.iterprev()):
            key = str(key, encoding='utf-8')
            
            if "T1" in dataType :
                if name in key and 'RGB' in key :
#                     print('+++++++++++ T1 '+key+' +++++++++++++++')
                    temp1.append(np.frombuffer(value).tolist())
                    age.append(float(np.frombuffer(lmdb_mod_txn.get((name+'-0_age').encode()),dtype=np.float64)))
                    UPDRS.append(float(np.frombuffer(lmdb_mod_txn.get((name+'-0_UPDRS').encode()),dtype=np.float64)))
                    MMSE.append(float(np.frombuffer(lmdb_mod_txn.get((name+'-0_MMSE').encode()),dtype=np.float64)))
                
            elif "DTI" in dataType :
                if name in key and 'DTI_Slice' in key :
#                     print('+++++++++++ DTI '+key+' +++++++++++++++')
                    temp1.append(np.frombuffer(value).tolist())
                    age.append(float(np.frombuffer(lmdb_mod_txn.get((name+'_DTI_age').encode()),dtype=np.float64)))
                    UPDRS.append(float(np.frombuffer(lmdb_mod_txn.get((name+'_DTI_UPDRS').encode()),dtype=np.float64)))
                    MMSE.append(float(np.frombuffer(lmdb_mod_txn.get((name+'_DTI_MMSE').encode()),dtype=np.float64)))
                
            elif "Bold" in dataType :
                if 'PD_'+name in key and 'Bold_Slice' in key:
#                     print('+++++++++++ Bold '+key+' +++++++++++++++')
                    temp1.append(np.frombuffer(value).tolist())
                    age.append(float(np.frombuffer(lmdb_mod_txn.get(('PD_'+name+'_Bold_age').encode()),dtype=np.float64)))
                    UPDRS.append(float(np.frombuffer(lmdb_mod_txn.get(('PD_'+name+'_Bold_UPDRS').encode()),dtype=np.float64)))
                    MMSE.append(float(np.frombuffer(lmdb_mod_txn.get(('PD_'+name+'_Bold_MMSE').encode()),dtype=np.float64)))
        scores.append([age[0],UPDRS[0],MMSE[0]])       
        data.append(temp1)
    print('data shape: ',np.array(data).shape,'score shape: ',np.array(scores).shape,scores)
    lmdb_env.close()
    return data,scores

def isExitName(FT1_lmdb,name,dataType):
    flag = False
    lmdb_env = lmdb.open(FT1_lmdb,readonly=True)
    with lmdb_env.begin() as lmdb_mod_txn:
        mod_cursor = lmdb_mod_txn.cursor()
        for idx,(key, value) in enumerate(mod_cursor.iterprev()):
            key = str(key, encoding='utf-8')                
            if "T1" in dataType :
                if name in key and 'RGB' in key :
                    flag = True                
            elif "DTI" in dataType :
                if name in key and 'DTI_Slice' in key :
                    flag = True                
            elif "Bold" in dataType :
                if name in key and 'Bold_Slice' in key:
                    flag = True
    
    return flag

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
        ## save model
#         torch.save(model.state_dict(),'/media/lhj/Momery/PD_predictDL/Data/PPMI/model_dro0.3/fusNet/DBfusModel_'+str(epoch)+'.pt')
        torch.save(model.state_dict(),'/media/lhj/Momery/PD_predictDL/Data/PPMI/model_dro0.3/sinNet/BoldsinNet_'+str(epoch)+'.pt')
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
    num2,CS2 = LowerCount(L1_MAE,2)
    num5,CS5 = LowerCount(L1_MAE,5)
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
#             print('input:',target,'predict:',out.tolist()[0][0][0])
            Predict_Scores.append(out.tolist()[0][0][0]+np.random.uniform(-2,2)),True_Scores.append(target.tolist()[0][0][0]+np.random.uniform(-2,2))
            
            loss = loss_func(out,target)
            L1_MAE.append(loss)
#             print('MAE:',loss)
        num2,CS2 = LowerCount(L1_MAE,2)
        num5,CS5 = LowerCount(L1_MAE,5)
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
#     print(type(Predict_Scores),type(True_Scores))
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


def trainModPPMI(work_dir):
    ## data path
    T1_dir = work_dir+'/image'
    Bold_dir = work_dir+'/Bold'
    DTI_dir = work_dir+'/DTI'
    ## get score
    Score_dir = '/media/lhj/Momery/PD_predictDL/Data/PPMI/MRIScore.txt'
    Score = []
    ## read scores
    score = readScores(Score_dir)

    ## load T1 DTI Bold and ERP, and extract Features from them
#     loadT1(T1_dir,score)
#     loadDTI(DTI_dir,score)
#     loadBold(Bold_dir,score)
    
    ## Main funtion for predict
    Prediction(work_dir)


## main program
work_dir = '/media/lhj/Momery/PD_predictDL/Data'
trainModPPMI(work_dir)

