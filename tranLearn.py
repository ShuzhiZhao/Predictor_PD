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
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.dropout(F.relu(self.hidden(x)))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

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
    VocReMa_dir = '/media/lhj/Momery/PD_predictDL/Data/VoReMa.txt'
    VocReMas = readScores(VocReMa_dir)
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
                        s = np.frombuffer(value).tolist()
                        s.append(float(VocReMas[int(ii)-1]))
                        score4.append(s)
                    if '_scores' in key and FT2_Type+'0'+FT3_Type+'0' in key :
                        s = np.frombuffer(value).tolist()
                        s.append(float(VocReMas[int(ii)-1]))
                        score1.append(s)
                    if '_scores' in key and FT1_Type+'0' in key and FT3_Type+'0' in key :
                        s = np.frombuffer(value).tolist()
                        s.append(float(VocReMas[int(ii)-1]))
                        score2.append(s)
                    if '_scores' in key and FT1_Type+'0' in key and FT2_Type+'0' in key :
                        s = np.frombuffer(value).tolist()
                        s.append(float(VocReMas[int(ii)-1]))
                        score3.append(s)
    print('fusFT: ',np.array(FT1).shape,np.array(score1).shape,np.array(FT2).shape,np.array(score2).shape,np.array(FT3).shape,np.array(score3).shape,np.array(fusFT).shape,np.array(score4).shape)            
    return FT1,score1,FT2,score2,FT3,score3,fusFT,score4

def getLmdbFus2(fusFT_dir,FT1_Type,FT2_Type,names):
    fusFT = []
    score = []
    VocReMa_dir = '/media/lhj/Momery/PD_predictDL/Data/VoReMa.txt'
    VocReMas = readScores(VocReMa_dir)
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
                        s = np.frombuffer(value).tolist()
                        s.append(float(VocReMas[int(ii)-1]))
                        score.append(s)
    print('fusFT: ',np.array(fusFT).shape,np.array(score).shape)            
    return fusFT,score    

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

def TranLearn(model,FT,SC,modName):
    import csv
    nameLabel = ['age', 'UPDRS', 'MMSE','VoReMa']
    index = 4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss_func = nn.L1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(),lr=0.001, weight_decay=5e-4)
    num_epochs = 2
    trainData,trainLabels,testData,testLabels = crossDataLabel(FT,subLabels(SC,index))
    print('******************************************'+nameLabel[index-1]+'************************************************')
    model_history = model_fit_evaluate(model,device,trainData,trainLabels,testData,testLabels,optimizer,loss_func,num_epochs)
    Res_dir = '/media/lhj/Momery/PD_predictDL/Data/Log/'+nameLabel[index-1]
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
    
    model_history['train_PreSc']=[];
    model_history['train_TruSc']=[];
    model_history['test_PreSc']=[];
    model_history['test_TruSc']=[];    

    for epoch in range(num_epochs):
        train_MAE,train_CS2,train_CS5,train_PreSc,train_TruSc =train(model, device, trainData, trainLabels, optimizer,loss_func, epoch)
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

    print("best testing accuarcy:",best_acc)
    print(torch.cuda.memory_summary())
    for ii in range(5):
        torch.cuda.empty_cache()
    print(torch.cuda.memory_summary())
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
        
    return MAE,CS2,CS5,Predict_Scores,True_Scores    

def LowerCount(a,b):
    num = 0
    for i in a:
        if i<b:
            num+=1
    percent = num/len(a)
    return num,percent    

def plotScatter(csv_dir):
    ## plot scatter for Predict_Scores,True_Scores
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.simplefilter('ignore')
    # Import Data
    csv_dir = csv_dir+'/tePreTrusin0YBold.csv'
    df = pd.read_csv(csv_dir)
#     print(df)
    # Plot
    sns.set_style("white")
    gridobj = sns.lmplot(x="test_TruSc", y="test_PreSc", hue="model", data=df)
    # Decorations
    plt.show()
    
## load Model
# Three Mod fus with different years from PPMI
mod_dir = '/media/lhj/Momery/PD_predictDL/Data/PPMI/model_dro0.3'
model_3fus0Y = Net(n_feature=512, n_hidden=200, n_output=1)
model_3fus2Y = Net(n_feature=512, n_hidden=200, n_output=1)
model_3fus4Y = Net(n_feature=512, n_hidden=200, n_output=1)
m_state_dict = torch.load(mod_dir+'/fusNet/3fusModel_1.pt')
model_3fus0Y.load_state_dict(m_state_dict)
m_state_dict = torch.load(mod_dir+'/fusNet/3fusModel_3.pt')
model_3fus2Y.load_state_dict(m_state_dict)
m_state_dict = torch.load(mod_dir+'/fusNet/3fusModel_5.pt')
model_3fus4Y.load_state_dict(m_state_dict)
# Two Mod fus with different years
name_2Mod = ['TB','TD','DB']
index = 1
model_2fus0Y = Net(n_feature=512, n_hidden=200, n_output=1)
model_2fus2Y = Net(n_feature=512, n_hidden=200, n_output=1)
model_2fus4Y = Net(n_feature=512, n_hidden=200, n_output=1)
m_state_dict = torch.load(mod_dir+'/fusNet/'+str(name_2Mod[index-1])+'fusModel_1.pt')
model_2fus0Y.load_state_dict(m_state_dict)
m_state_dict = torch.load(mod_dir+'/fusNet/'+str(name_2Mod[index-1])+'fusModel_3.pt')
model_2fus2Y.load_state_dict(m_state_dict)
m_state_dict = torch.load(mod_dir+'/fusNet/'+str(name_2Mod[index-1])+'fusModel_5.pt')
model_2fus4Y.load_state_dict(m_state_dict)
# Single Mod with different years
name_Sin = ['T1','DTI','Bold']
index1 = 1
model_sin0Y = Net(n_feature=512, n_hidden=200, n_output=1)
model_sin2Y = Net(n_feature=512, n_hidden=200, n_output=1)
model_sin4Y = Net(n_feature=512, n_hidden=200, n_output=1)
m_state_dict = torch.load(mod_dir+'/sinNet/'+str(name_Sin[index1-1])+'sinNet_1.pt')
model_sin0Y.load_state_dict(m_state_dict)
m_state_dict = torch.load(mod_dir+'/sinNet/'+str(name_Sin[index1-1])+'sinNet_3.pt')
model_sin2Y.load_state_dict(m_state_dict)
m_state_dict = torch.load(mod_dir+'/sinNet/'+str(name_Sin[index1-1])+'sinNet_5.pt')
model_sin4Y.load_state_dict(m_state_dict)

## load data from Our collection
# Tree fus data and Single
work_dir = '/media/lhj/Momery/PD_predictDL/Data'
fusFT_dir = work_dir+'/FT_MulFus'
conName_dir = '/media/lhj/Momery/PD_predictDL/Data/conNames.txt'
names = readScores(conName_dir)
print('*********** start get FT1 FT2 FT3 fusFT scores **************')
FT1,scores1,FT2,scores2,FT3,scores3,fus3FT,scores4 = getLmdbFus(fusFT_dir,'T1','DTI','Bold',names)
print('*********** start get Two '+str(name_2Mod[index-1])+' fusFT scores **************')
if index == 1 :
    fus2FT,fus2Scores = getLmdbFus2(fusFT_dir,'T1','Bold',names)
    sinFT,sinScores = FT1,scores1
elif index == 2 :
    fus2FT,fus2Scores = getLmdbFus2(fusFT_dir,'T1','DTI',names)
    sinFT,sinScores = FT2,scores2
elif index == 3 :
    fus2FT,fus2Scores = getLmdbFus2(fusFT_dir,'DTI','Bold',names)
    sinFT,sinScores = FT3,scores3

## Transfor Learning Model
# different modality
# model_his3fus0Y = TranLearn(model_3fus0Y,fus3FT,scores4,'3fus0Y')
model_his2fus0Y = TranLearn(model_2fus0Y,fus2FT,fus2Scores,'2fus0Y'+str(name_2Mod[index-1]))
model_hissin0Y = TranLearn(model_sin0Y,sinFT,sinScores,'sin0Y'+str(name_Sin[index-1]))

# model_his3fus2Y = TranLearn(model_3fus2Y,fus3FT,scores4,'3fus2Y')
model_his2fus2Y = TranLearn(model_2fus2Y,fus2FT,fus2Scores,'2fus2Y'+str(name_2Mod[index-1]))
model_hissin2Y = TranLearn(model_sin2Y,sinFT,sinScores,'sin2Y'+str(name_Sin[index-1]))

# model_his3fus4Y = TranLearn(model_3fus4Y,fus3FT,scores4,'3fus4Y')
model_his2fus4Y = TranLearn(model_2fus4Y,fus2FT,fus2Scores,'2fus4Y'+str(name_2Mod[index-1]))
model_hissin4Y = TranLearn(model_sin4Y,sinFT,sinScores,'sin4Y'+str(name_Sin[index-1]))
# plot scotter figure
# csv_dir = '/media/lhj/Momery/PD_predictDL/Data/Log'
# plotScatter(csv_dir)






