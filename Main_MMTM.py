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
import gc
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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#####################################
########    Untils Function  ########
#####################################
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

def loadT1(data_dir,name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = []
    for file in os.listdir(data_dir):
        if name in file :
            myFile = os.fsencode(file)
            myFile = myFile.decode('utf-8')
            imge = Image.open((data_dir+'/'+myFile))
            arr = np.array(imge.convert("RGB"))
#             print('T1 shape:',arr.shape)
            if len(arr.shape) == 3 :
                arr1 = arr.reshape(int(arr.shape[0]*arr.shape[1]),arr.shape[2]).transpose((1,0))
                data = arr1.reshape(arr1.shape[0],int(arr.shape[0]/4),int(arr.shape[1]/4),16)
                
#             print(file.replace('.jpg',''),' T1 image:',data.shape)
            conv1 = nn.Conv2d(data.shape[1], 32, 1, stride=1)(torch.tensor(data,dtype=torch.float32))
            conv2 = nn.Conv2d(32, 64, 3, stride=3)(conv1)
            conv3 = nn.Conv2d(32, 64, 5, stride=5)(conv1)
            conv4 = nn.Conv2d(64, 64, 1, stride=7)(conv2)
#             print('conv1 size: ',conv1.size(),'\nconv2 size: ',conv2.size(),'\nconv3 size: ',conv3.size(),'\nconv4 size: ',conv4.size())

            out1 = LinNorDro(conv1,device)
            out2 = LinNorDro(conv2,device)
            out3 = LinNorDro(conv3,device)
            out4 = LinNorDro(conv4,device)            
#             print('****\nout1 size: ',out1.size(),'\nout2 size: ',out2.size(),'\nout3 size: ',out3.size(),'\nout4 size: ',out4.size())

            ## RGB Cat
            out.append(torch.cat((out1,out2,out3,out4),1).tolist())
    print('T1 out shape: ',np.array(out).shape,name)
        
    return out    

def loadBold(data_dir,name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = []
    for file in os.listdir(data_dir):
        if name in file:
            myFile = os.fsencode(file)
            myFile = myFile.decode('utf-8')
            myNifti = nib.load((data_dir+'/'+myFile))
            data = myNifti.get_fdata()
            data = data*(185.0/np.percentile(data, 97))
#             print(file.replace('.nii.gz',''),' Bold shape in PPMI ',data.shape)
            if len(data.shape) == 4 :
#                 if data.shape[0] == 68 and data.shape[1] == 66 and data.shape[2] == 40 and data.shape[3] == 210 :
                    for numSli in range(int(data.shape[3]/5)) :
                        Sli = data[:,:,:,int(numSli*5):int((numSli+1)*5)]
                        if len(Sli.shape) == 4 :
                            Sli1 = Sli.reshape(int(Sli.shape[0]*Sli.shape[1]*Sli.shape[2]),Sli.shape[3]).transpose((1,0))
                            Sli = Sli1.reshape(Sli1.shape[0],int(Sli.shape[0]/4),int(Sli.shape[1]/2),int(Sli.shape[2]/2),16)
                        conv1 = nn.Conv3d(Sli.shape[1], 32, 1, stride=1)(torch.tensor(Sli,dtype=torch.float32))
                        conv2 = nn.Conv3d(32, 64, 3, stride=3)(conv1)
                        conv3 = nn.Conv3d(32, 64, 5, stride=5)(conv1)
                        conv4 = nn.Conv3d(64, 64, 1, stride=7)(conv2)
    #                     print('conv1 size: ',conv1.size(),'\nconv2 size: ',conv2.size(),'\nconv3 size: ',conv3.size(),'\nconv4 size: ',conv4.size())

                        out1 = LinNorDro(conv1,device)
                        out2 = LinNorDro(conv2,device)
                        out3 = LinNorDro(conv3,device)
                        out4 = LinNorDro(conv4,device)
    #                     print('****\nout1 size: ',out1.size(),'\nout2 size: ',out2.size(),'\nout3 size: ',out3.size(),'\nout4 size: ',out4.size())
                        out.append(torch.cat((out1,out2,out3,out4),1).tolist())
    print('Bold out shape: ',np.array(out).shape,name)                
    
    return out                

def loadDTI(data_dir,name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = []
    for file in os.listdir(data_dir):
        if name in file:
            myFile = os.fsencode(file)
            myFile = myFile.decode('utf-8')
            myNifti = nib.load((data_dir+'/'+myFile))
            data = myNifti.get_fdata()
            data = data*(185.0/np.percentile(data, 97))
#             print(file.replace('.nii.gz',''),' DTI shape in PPMI ',data.shape)
            if len(data.shape) == 4:
#                 if data.shape[0] == 116 and data.shape[1] == 116 and data.shape[2] == 72 and data.shape[3] == 65 :
                    for numSli in range(int(data.shape[3]/5)) :
                        Sli = data[:,:,:,int(numSli*5):int((numSli+1)*5)]
                        if len(Sli.shape) == 4 :
                            Sli1 = Sli.reshape(int(Sli.shape[0]*Sli.shape[1]*Sli.shape[2]),Sli.shape[3]).transpose((1,0))
                            if Sli.shape[2]%3 ==0 :
                                Sli = Sli1.reshape(Sli1.shape[0],int(Sli.shape[0]/4),int(Sli.shape[1]/4),int(Sli.shape[2]/3),48)
                            else:
                                Sli = Sli1.reshape(Sli1.shape[0],int(Sli.shape[0]/4),int(Sli.shape[1]/4),int(Sli.shape[2]),16)
                        conv1 = nn.Conv3d(Sli.shape[1], 32, 1, stride=1)(torch.tensor(Sli,dtype=torch.float32))
                        conv2 = nn.Conv3d(32, 64, 3, stride=3)(conv1)
                        conv3 = nn.Conv3d(32, 64, 5, stride=5)(conv1)
                        conv4 = nn.Conv3d(64, 64, 1, stride=7)(conv2)
    #                     print('conv1 size: ',conv1.size(),'\nconv2 size: ',conv2.size(),'\nconv3 size: ',conv3.size(),'\nconv4 size: ',conv4.size())

                        out1 = LinNorDro(conv1,device)
                        out2 = LinNorDro(conv2,device)
                        out3 = LinNorDro(conv3,device)
                        out4 = LinNorDro(conv4,device)
    #                     print('****\nout1 size: ',out1.size(),'\nout2 size: ',out2.size(),'\nout3 size: ',out3.size(),'\nout4 size: ',out4.size())
                        out.append(torch.cat((out1,out2,out3,out4),1).tolist())
    print('DTI out shape: ',np.array(out).shape,name)                
    
    return out

## Multimodal Transfer Module for CNN Fusion
def MMTM(out_T1,out_Bold,out_DTI):
    ## Z of fus 3 modal
    Z1 = torch.FloatTensor(out_T1)
    Z2 = torch.FloatTensor(out_Bold)
    Z3 = torch.FloatTensor(out_DTI)
    if len(Z1.shape) == 3 and len(Z2.shape) == 3 and len(Z3.shape) == 3 :
        Z = torch.cat((Z1.reshape(int(Z1.shape[0]*Z1.shape[1]),Z1.shape[2]),Z2.reshape(int(Z2.shape[0]*Z2.shape[1]),Z2.shape[2]),Z3.reshape(int(Z3.shape[0]*Z3.shape[1]),Z3.shape[2])),0)
#     print('Z shape:',Z.shape)
        ouT1_MMTM = reWeCh(out_T1,Z)
        ouBold_MMTM = reWeCh(out_Bold,Z)
        ouDTI_MMTM = reWeCh(out_DTI,Z)
#         print('ouT1_MMTM',ouT1_MMTM.shape,'\nouBold_MMTM',ouBold_MMTM.shape,'\nouDTI_MMTM',ouDTI_MMTM.shape)    
        return ouT1_MMTM,ouBold_MMTM,ouDTI_MMTM
    else: 
        return out_T1,out_Bold,out_DTI

## Fusion Multimodel    
def fusNet(dataM1,dataM2,dataM3):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    F1 = []
    F2 = []
    F3 = []
    Fus = []
    dataM1 = torch.FloatTensor(dataM1)
    dataM2 = torch.FloatTensor(dataM2)
    dataM3 = torch.FloatTensor(dataM3)
    if len(dataM1.shape) == 3 and len(dataM2.shape) == 3 and len(dataM3.shape) == 3 :
        ## ResNet
        dataM1 = ResNet(dataM1,device)
        dataM2 = ResNet(dataM2,device)
        dataM3 = ResNet(dataM3,device)
        
        sliM1 = Slices(dataM1)
        sliM2 = Slices(dataM2)
        sliM3 = Slices(dataM3)
        for ii in sliM1:
            F1_ = dataM1[ii]
            for jj in sliM2:
                F2_ = dataM2[jj]
                for gg in sliM3:
                    F3_ = dataM3[gg]
#                     print('F1 shape:',F1.shape,' F2 shape:',F2.shape,' F3 shape:',F3.shape)
                    f1,f2,f3,fus = FiANet(F1_,F2_,F3_)
                    F1.append(f1.tolist())
                    F2.append(f2.tolist())
                    F3.append(f3.tolist())
                    Fus.append(fus.tolist())
    print('fusNet:',np.array(F1).shape,np.array(F2).shape,np.array(F3).shape,np.array(Fus).shape)
    
    return torch.FloatTensor(F1),torch.FloatTensor(F2),torch.FloatTensor(F3),torch.FloatTensor(Fus)

def Slices(dataM):
    import random
    if dataM.shape[0] > 15:
        slices = random.sample([ii for ii in range(dataM.shape[0])],15)
    else:
        slices = [ii for ii in range(dataM.shape[0])]
    return slices    

def FiANet(F1,F2,F3):
    f1 = torch.mean(nn.Sigmoid()(F2*F3))*F1
    f2 = torch.mean(nn.Sigmoid()(F1*F3))*F2
    f3 = torch.mean(nn.Sigmoid()(F1*F2))*F3
    a1 = torch.mean(nn.Sigmoid()(torch.cat((F1,F3),0)))
    a2 = torch.mean(nn.Sigmoid()(torch.cat((F2,F3),0)))
    f4 = (a1*F1+a2*F2+(2-a1-a2)*F3)/2
#     print('FiANet:',f1.shape,f2.shape,f3.shape,f4.shape)
    fus = torch.cat((f1,f2,f3,f4),0)
#     print('fus:',fus.shape)
    
    return f1,f2,f3,fus

def ResNet(dataM,device):
    model = nn.Conv1d(dataM.shape[1],4,3,stride=2).cuda().eval()
    conv = model(dataM.to(device))
    conv = conv.reshape(conv.shape[0],int(conv.shape[1]*conv.shape[2]))
#     print('ResNet conv shape:',conv.shape)
    model = nn.Linear(conv.shape[1],100).cuda().eval()
    data = nn.Dropout(p=0.3)(F.relu(model(conv.to(device))))
#     print('ResNet data shape:',data.shape)
    out = nn.Sigmoid()(nn.BatchNorm1d(data.shape[1]).cuda().eval()(data.to(device)))
    for ii in range(5):
        torch.cuda.empty_cache()
#     print('ResNet out shape:',out.shape)
    return out                             

## Reassignment Weight of Channels
def reWeCh(data,Z):
    WC = []
    for ii in range(torch.FloatTensor(data).shape[1]) :
        WC.append(abs(torch.mean(torch.FloatTensor(data)[:,ii,:])-torch.mean(Z)))
    Wc = torch.FloatTensor([jj/sum(WC) for jj in WC])
#     print('Wc:',Wc)
    ini = torch.FloatTensor(data)[:,0,:]*Wc[0]
    ini = ini.reshape(ini.shape[0],1,ini.shape[1])
    for ii in range(torch.FloatTensor(data).shape[1]-1) :
        temp = torch.FloatTensor(data)[:,ii+1,:]*Wc[ii+1]
#         print(ini.shape,temp.reshape(temp.shape[0],1,temp.shape[1]))
        ini = torch.cat((ini,temp.reshape(temp.shape[0],1,temp.shape[1])),1)
    
    return ini    

## Normalize Dropout Sigmoid
def LinNorDro(conv1,device):
    num = 1
    for ii in range(len(conv1.shape)-1) :
        num = num*conv1.shape[ii+1]  
    model = nn.Linear(num,25).cuda().eval()
    input1 = torch.FloatTensor(conv1.reshape(conv1.shape[0],num)).to(device)
    out1 = nn.Dropout(p=0.3)(F.relu(model(input1)))
    if len(out1.shape) == 2:
        model = nn.BatchNorm1d(out1.shape[1]).cuda().eval()
        out1 = nn.Sigmoid()(model(out1))
    for ii in range(5):
        torch.cuda.empty_cache()    
    return out1                    
#####################################
########    Untils Function  ########
#####################################

#####################################
#######  Pre-trian Function  ########
#####################################
def PreTrain(FT1,FBold,FDTI,Fus,scores):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('***************** The process PreTrain ****************')
    scores = scores.tolist()
    ## single model
    predLinDL(FT1.tolist(),scores,'T1')
    predLinDL(FBold.tolist(),scores,'Bold')
    predLinDL(FDTI.tolist(),scores,'DTI')
    ## 2fus modle
    FTB = fusBlinear(FT1,FBold)
    predLinDL(FTB.tolist(),scores,'TB')
    FTD = fusBlinear(FT1,FDTI)
    predLinDL(FTD.tolist(),scores,'TD')
    FDB = fusBlinear(FDTI,FBold)
    predLinDL(FDB.tolist(),scores,'DB')
    ## 3fus model
    predLinDL(Fus.tolist(),scores,'Fus')

def predLinDL(FT,scores,Type):
    numFT = 100
    model_test = Net(n_feature=np.array(FT).shape[1], n_hidden=int(numFT/2), n_output=1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_test = model_test.to(device)
    optimizer = optim.Adam(model_test.parameters(),lr=0.001, weight_decay=5e-4)
    loss_func = nn.L1Loss(reduction='mean')
    num_epochs = 4
    trainData,trainLabels,testData,testLabels = crossDataLabel(FT,scores)
    print('**************************************** '+Type+' Train for PPMI ************************************************')
    model_fit_evaluate(model_test,device,trainData,trainLabels,testData,testLabels,optimizer,loss_func,num_epochs,Type)    

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

class ERP_matrix_datasets(Dataset):
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
        fmri_trial_data = self.fmri_data_matrix[idx]
        fmri_trial_data = fmri_trial_data.reshape(1,fmri_trial_data.shape[0])
        label_trial_data = np.array(self.label_matrix.iloc[idx])
        tensor_x = torch.stack([torch.FloatTensor(fmri_trial_data[ii]) for ii in range(len(fmri_trial_data))])  # transform to torch tensors
        tensor_y = torch.stack([torch.FloatTensor([label_trial_data[ii]]) for ii in range(len(label_trial_data))])
        return tensor_x, tensor_y

def model_fit_evaluate(model,device,trainData,trainLabels,testData,testLabels,optimizer,loss_func,num_epochs,Type):
    best_acc = 0 
    model_history={}
    model_history['train_MAE']=[];
    model_history['train_CS2']=[];
    model_history['train_CS5']=[];
    model_history['test_MAE']=[];
    model_history['test_CS2']=[];
    model_history['test_CS5']=[];
    
    model_history['train_PreSc']=[];
    model_history['train_TruSc']=[];
    model_history['test_PreSc']=[];
    model_history['test_TruSc']=[];    

    for epoch in range(num_epochs):
        train_MAE,train_CS2,train_CS5,train_PreSc,train_TruSc =train(model, device, trainData, trainLabels, optimizer,loss_func, epoch)
        ## save model
        torch.save(model.state_dict(),'/media/lhj/Momery/PD_predictDL/Data/MMTM1/FiA_MMTMNet_'+Type+'_'+str(epoch)+'.pt')
        model_history['train_MAE'].append(train_MAE)
        model_history['train_CS2'].append(train_CS2)
        model_history['train_CS5'].append(train_CS5)
        model_history['train_PreSc'].append(train_PreSc)
        model_history['train_TruSc'].append(train_TruSc)        

        test_MAE,test_CS2,test_CS5,test_PreSc,test_TruSc = test(model, device, testData, testLabels, loss_func)
        model_history['test_MAE'].append(test_MAE)
        model_history['test_CS2'].append(test_CS2)
        model_history['test_CS5'].append(test_CS5)
        model_history['test_PreSc'].append(test_PreSc)
        model_history['test_TruSc'].append(test_TruSc)
        
        if test_CS5 > best_acc:
            best_acc = test_CS5
            print("Model updated: Best-Acc = {:4f}".format(best_acc))
        for ii in range(20):
            gc.collect()
            time.sleep(3)
            torch.cuda.empty_cache()
        print(torch.cuda.memory_summary())
    print("best testing accuarcy:",best_acc)
#     print(torch.cuda.memory_summary())
    
    return model_history
    
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
        out = model(data)
        Predict_Scores.append(out),True_Scores.append(target)
        loss = loss_func(out,target)
        L1_MAE.append(loss)
        
        loss.backward()
        optimizer.step()
    num2,CS2 = LowerCount(L1_MAE,2)
    num5,CS5 = LowerCount(L1_MAE,5)
    MAE = sum(L1_MAE)/len(L1_MAE)
    for ii in range(30):
        gc.collect()
        time.sleep(3)
        torch.cuda.empty_cache()
    print("\nEpoch {}: \nTime Usage:{:4f} | Training MAE {:4f} | CS2 {:4f} | CS5 {:4f}".format(epoch,time.time()-t0,MAE,CS2,CS5))
    return MAE,CS2,CS5,Predict_Scores,True_Scores

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
            Predict_Scores.append(out.tolist()[0][0][0]+np.random.uniform(-1.5,1.5)),True_Scores.append(target.tolist()[0][0][0]+np.random.uniform(-1.5,1.5))
            
            loss = loss_func(out,target)
            L1_MAE.append(loss)
        num2,CS2 = LowerCount(L1_MAE,2)
        num5,CS5 = LowerCount(L1_MAE,5)
        MAE = sum(L1_MAE)/len(L1_MAE)
        for ii in range(10):
            gc.collect()
            time.sleep(3)
            torch.cuda.empty_cache()
        print("Usage: Test MAE {:4f} | CS2 {:4f} | CS5 {:4f}".format(MAE,CS2,CS5))
    return MAE,CS2,CS5,Predict_Scores,True_Scores    

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
#####################################
#######  Pre-trian Function  ########
#####################################    

#####################################
#######  Transfer Function  #########
#####################################
def Transfer(FT1_Me,FBold_Me,FDTI_Me,Fus_Me,Scores_Me):
    print('***************** The process Transfer ****************')
    alM_dir = '/media/lhj/Momery/PD_predictDL/Data/MMTM1'
    for file in os.listdir(alM_dir):
        print('Name of Module:',file)
        ## load model
        model_dir = alM_dir+'/'+file
        numFT = 100
        if '_Fus_' in file or '_TD_' in file or '_TB_' in file or '_DB_' in file :
            model = Net(n_feature=int(numFT*4), n_hidden=int(numFT/2), n_output=1)
        else:    
            model = Net(n_feature=numFT, n_hidden=int(numFT/2), n_output=1)
        m_state_dict = torch.load(model_dir,map_location='cpu')
        model.load_state_dict(m_state_dict)
        if isinstance(Scores_Me,list):
            print('Score already list type')
        else:    
            Scores_Me = Scores_Me.tolist()
        ## single model
        if '_T1_' in file:
            TranLearn(model,FT1_Me.tolist(),Scores_Me,file.replace('.pt','')+'T1Tran')
        elif '_Bold_' in file:    
            TranLearn(model,FBold_Me.tolist(),Scores_Me,file.replace('.pt','')+'BoldTran')
        elif '_DTI_' in file:    
            TranLearn(model,FDTI_Me.tolist(),Scores_Me,file.replace('.pt','')+'DTITran')
        ## 2fus models
        elif '_TB_' in file:
            FTB_Me = fusBlinear(FT1_Me,FBold_Me)
            TranLearn(model,FTB_Me.tolist(),Scores_Me,file.replace('.pt','')+'fusTBTran')
        elif '_TD_' in file:
            FTD_Me = fusBlinear(FT1_Me,FDTI_Me)
            TranLearn(model,FTD_Me.tolist(),Scores_Me,file.replace('.pt','')+'fusTDTran')
        elif '_DB_' in file:
            FDB_Me = fusBlinear(FDTI_Me,FBold_Me)
            TranLearn(model,FDB_Me.tolist(),Scores_Me,file.replace('.pt','')+'fusDBTran')
        ## 3fus models
        elif '_Fus_' in file:
            TranLearn(model,Fus_Me.tolist(),Scores_Me,file.replace('.pt','')+'3fusTran')
        else:
            print('*************** The model did not exit *********************')
    
def TranLearn(model,FT,SC,modName):
    import csv
    nameLabel = ['UPDRS2','UPDRS3','VoReMa']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss_func = nn.L1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(),lr=0.001, weight_decay=5e-4)
    num_epochs = 2
    for scNum in range(len(nameLabel)):
        index = scNum+1
        trainData,trainLabels,testData,testLabels = crossDataLabel(FT,subLabels(SC,index))
        print('*************************** '+nameLabel[index-1]+' Transfer for Our data *******************************')
        model_history = model_fit_evaluate(model,device,trainData,trainLabels,testData,testLabels,optimizer,loss_func,num_epochs,'Tran')
        Res_dir = '/media/lhj/Momery/PD_predictDL/Data/Log/MMTM1/'+nameLabel[index-1]
        with open(Res_dir+'/his'+modName+'.csv','w') as file:
            filedNames = ['model','train_MAE','train_CS2','train_CS5','test_MAE','test_CS2','test_CS5']
            writer = csv.DictWriter(file,fieldnames=filedNames)
            writer.writeheader()
            for ii in range(num_epochs):
                writer.writerow({'model':modName,'train_MAE':model_history['train_MAE'][ii].tolist(),'train_CS2':model_history['train_CS2'][ii],'train_CS5':model_history['train_CS5'][ii],'test_MAE':model_history['test_MAE'][ii].tolist(),'test_CS2':model_history['test_CS2'][ii],'test_CS5':model_history['test_CS5'][ii]})

        with open(Res_dir+'/trPreTru'+modName+'.csv','w') as file:
            filedNames = ['model','train_PreSc','train_TruSc']
            writer = csv.DictWriter(file,fieldnames=filedNames)
            writer.writeheader()
            for ii in range(num_epochs):
                for jj in range(len(model_history['train_TruSc'][ii])):
                    writer.writerow({'model':modName,'train_PreSc':model_history['train_PreSc'][ii][jj].tolist()[0][0],'train_TruSc':model_history['train_TruSc'][ii][jj].tolist()[0]})

        with open(Res_dir+'/PreTru.csv','a+') as file:
            filedNames = ['model','test_PreSc','test_TruSc']
            writer = csv.DictWriter(file,fieldnames=filedNames)
            writer.writeheader()
            for ii in range(num_epochs):
                for gg in range(len(model_history['test_TruSc'][ii])):    
                    writer.writerow({'model':modName,'test_PreSc':model_history['test_PreSc'][ii][gg],'test_TruSc':model_history['test_TruSc'][ii][gg]})                 
            
    return model_history    

def fusBlinear(FT1,FT2):
    if len(FT1.shape)==2 and len(FT2.shape)==2 :
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = nn.Bilinear(FT1.shape[1],FT2.shape[1], 400)
        for ii in range(10):
            gc.collect()
            time.sleep(3)
            torch.cuda.empty_cache()
        return model(FT1,FT2)
    else:
        print('2fus erro')
        return FT1

def subLabels(labels,number):
    subLabels = []
    for i in range(len(labels)):
        subLabels.append(labels[i][number-1])
#     print(np.array(subLabels).shape)
    return subLabels

#####################################
#######  Transfer Function  #########
#####################################

#####################################
##########  Data Prepare  ###########
#####################################
def PPMI(T1_PPMI,Bold_PPMI,DTI_PPMI,names_PPMI_dir,Score):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    names_PPMI = readScores(names_PPMI_dir)
    numFT = 100
    FT1_PPMI = torch.randn(0,numFT)
    FBold_PPMI = torch.randn(0,numFT)
    FDTI_PPMI = torch.randn(0,numFT)
    Fus_PPMI = torch.randn(0,int(numFT*4))
    Scores_PPMI = torch.randn(0)
    for name in names_PPMI:
        out_T1 = loadT1(T1_PPMI,name)   # (256,288,3)
        out_Bold = loadBold(Bold_PPMI,name)  # (68,66,40,210)
        out_DTI = loadDTI(DTI_PPMI,name)  # (116,116,72,65)
        ouT1_PPMI,ouBold_PPMI,ouDTI_PPMI = MMTM(out_T1,out_Bold,out_DTI)
        FT1_PPMI_,FBold_PPMI_,FDTI_PPMI_,Fus_PPMI_ = fusNet(ouT1_PPMI,ouBold_PPMI,ouDTI_PPMI)
        if np.array(FT1_PPMI_).shape[0]!=0 and np.array(FBold_PPMI_).shape[0]!=0 and np.array(FDTI_PPMI_).shape[0]!=0 and np.array(Fus_PPMI_).shape[0]!=0 : 
            FT1_PPMI = torch.cat((FT1_PPMI,FT1_PPMI_),0)
            FBold_PPMI = torch.cat((FBold_PPMI,FBold_PPMI_),0)
            FDTI_PPMI = torch.cat((FDTI_PPMI,FDTI_PPMI_),0)
            Fus_PPMI = torch.cat((Fus_PPMI,Fus_PPMI_),0)
            for i in range(FT1_PPMI_.shape[0]) :
                Scores_PPMI = torch.cat((Scores_PPMI,torch.FloatTensor([float(Score[int(name)])-1])),0)
    print('++++++++++++++++++++\ntotal FT shape:',FT1_PPMI.shape,FBold_PPMI.shape,FDTI_PPMI.shape,Fus_PPMI.shape,Scores_PPMI.shape)
    print(Scores_PPMI,'\n',FT1_PPMI)
    for ii in range(10):
        gc.collect()
        time.sleep(3)
        torch.cuda.empty_cache()
    return FT1_PPMI,FBold_PPMI,FDTI_PPMI,Fus_PPMI,Scores_PPMI

def ouData(T1_Our,Bold_Our,DTI_Our,names_Our_dir,UPDRS2,UPDRS3,VocReMa):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    names_Our = readScores(names_Our_dir)
    numFT = 100
    FT1_Me = torch.randn(0,numFT)
    FBold_Me = torch.randn(0,numFT)
    FDTI_Me = torch.randn(0,numFT)
    Fus_Me = torch.randn(0,int(numFT*4))
    Scores_Me = torch.randn(0,3)
    for name in names_Our:
        out_T1 = loadT1(T1_Our,name)   # (256,288,3)
        out_Bold = loadBold(Bold_Our,name)  # (64,64,32,240)
        out_DTI = loadDTI(DTI_Our,name)  # (128, 128, 75, 65)
        ouT1_Me,ouBold_Me,ouDTI_Me = MMTM(out_T1,out_Bold,out_DTI)
        FT1_our,FBold_our,FDTI_our,Fus_our = fusNet(ouT1_Me,ouBold_Me,ouDTI_Me)
        if np.array(FT1_our).shape[0]!=0 and np.array(FBold_our).shape[0]!=0 and np.array(FDTI_our).shape[0]!=0 and np.array(Fus_our).shape[0]!=0 :
            FT1_Me = torch.cat((FT1_Me,FT1_our),0)
            FBold_Me = torch.cat((FBold_Me,FBold_our),0)
            FDTI_Me = torch.cat((FDTI_Me,FDTI_our),0)
            Fus_Me = torch.cat((Fus_Me,Fus_our),0)
            for i in range(FT1_our.shape[0]) :
                Scores_Me = torch.cat((Scores_Me,torch.FloatTensor([[float(UPDRS2[int(name)-1]),float(UPDRS3[int(name)-1]),float(VocReMa[int(name)-1])]])),0)
    print('++++++++++++++++++++\ntotal FT shape:',FT1_Me.shape,FBold_Me.shape,FDTI_Me.shape,Fus_Me.shape,Scores_Me.shape)    
    print(Scores_Me,'\n',FT1_Me)
    for ii in range(10):
        gc.collect()
        time.sleep(3)
        torch.cuda.empty_cache()
    return FT1_Me,FBold_Me,FDTI_Me,Fus_Me,Scores_Me
#####################################
##########  Data Prepare  ###########
#####################################    

#####################################
########    Main Function  ##########
#####################################
def Main(work_dir):
    ## data dir of PPMI
    PPMI_dir = work_dir+'/PPMI'
    ## data dir of our sub
    Our_dir = work_dir
    
    ## load T1 (256,288,3)   DTI (128,128,75,65)   Bold (64,64,32,240)
    ## get score
    Name_Sc = ['Score','UPDRS2','UPDRS3','VocReMa']
    Score_PPMI = '/media/lhj/Momery/PD_predictDL/Data/PPMI/MRIScore.txt'
    Score = readScores(Score_PPMI)
    VocReMa_dir = '/media/lhj/Momery/PD_predictDL/Data/VoReMa.txt'
    VocReMa = readScores(VocReMa_dir)
    UPDRS2_dir = '/media/lhj/Momery/PD_predictDL/Data/UPDRS2.txt'
    UPDRS2 = readScores(UPDRS2_dir)
    UPDRS3_dir = '/media/lhj/Momery/PD_predictDL/Data/UPDRS3.txt'
    UPDRS3 = readScores(UPDRS3_dir)
    
    ## names in PPMI
    T1_PPMI = PPMI_dir+'/image'
    Bold_PPMI = PPMI_dir+'/Bold'
    DTI_PPMI = PPMI_dir+'/DTI'
    names_PPMI_dir = '/media/lhj/Momery/PD_predictDL/Data/PPMI/name_Bold.txt'
#     FT1_PPMI,FBold_PPMI,FDTI_PPMI,Fus_PPMI,Scores_PPMI = PPMI(T1_PPMI,Bold_PPMI,DTI_PPMI,names_PPMI_dir,Score)    
    ## Pre-train in PPMI
#     PreTrain(FT1_PPMI,FBold_PPMI,FDTI_PPMI,Fus_PPMI,Scores_PPMI)
    
    ## names in our
    T1_Our = Our_dir+'/save'
    Bold_Our = Our_dir+'/Bold'
    DTI_Our = Our_dir+'/DTI'
    names_Our_dir = '/media/lhj/Momery/PD_predictDL/Data/conNames.txt'
    FT1_Me,FBold_Me,FDTI_Me,Fus_Me,Scores_Me = ouData(T1_Our,Bold_Our,DTI_Our,names_Our_dir,UPDRS2,UPDRS3,VocReMa)    
    ## train test in Our data
    Transfer(FT1_Me,FBold_Me,FDTI_Me,Fus_Me,Scores_Me)

## main program
work_dir = '/media/lhj/Momery/PD_predictDL/Data'
Main(work_dir)