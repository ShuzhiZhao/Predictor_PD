from utils import readScores,Prediction,loadT1,loadDTI,loadBold,loadERP
    
## predict age UPDRS and MMSE
def mainPredict(work_dir):
    ## data path
    T1_dir = work_dir+'/Data/save'
    Bold_dir = work_dir+'/Data/Bold'
    DTI_dir = work_dir+'/Data/DTI'
    ERP_dir = '/media/lhj/Momery/PD_GCN/Script/test_ChebNet/ERP_Matrix'
    ## get all score of age UPDRS and MMSE
    age_dir = '/media/lhj/Momery/PD_predictDL/Data/age.txt'
    age = []
    UPDRS_dir = '/media/lhj/Momery/PD_predictDL/Data/UPDRS.txt'
    UPDRS = []
    MMSE_dir = '/media/lhj/Momery/PD_predictDL/Data/MMSE.txt'
    MMSE = []
    ## read scores
    age = readScores(age_dir)
    UPDRS = readScores(UPDRS_dir)
    MMSE = readScores(MMSE_dir)
#     print(np.array(age).shape,np.array(UPDRS).shape,np.array(MMSE).shape)
    ## load T1 DTI Bold and ERP, and extract Features from them
#     loadT1(T1_dir,age,UPDRS,MMSE)
#     loadDTI(DTI_dir,age,UPDRS,MMSE)
#     loadBold(Bold_dir,age,UPDRS,MMSE)
#     loadERP(ERP_dir,age,UPDRS,MMSE)
    
    ## Main funtion for predict
    Prediction(work_dir)


## main program
work_dir = '/media/lhj/Momery/PD_predictDL'
mainPredict(work_dir)