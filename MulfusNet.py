#*****************************************************
#
#  This is the PyTorch code for fusionFT with Three Multi-model
#  
#  Medical Image Analysis, Volume 72, 2021, 102091
#
#  Author: zhaoshuzhi
#
#*****************************************************

import torch
import torch.nn as nn
   
class first_conv(nn.Module):
    def __init__(self,inplace,outplace):
        super().__init__()
        self.conv = nn.Conv1d(inplace, outplace, 3, stride=2)
        self.bn = nn.BatchNorm1d(outplace)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x
            
class Resnet(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
                
        self.conv1d_1 = first_conv(in_channels,64)
        self.conv1d_layer1 = nn.Conv1d(64, 128, 3, stride=2)
        self.conv1d_layer2 = nn.Conv1d(128, 256, 3, stride=2)
        self.conv1d_layer3 = nn.Conv1d(256, 512, 3, stride=2)
        self.conv1d_layer4 = nn.Conv1d(512, 512, 3, stride=2)
        
        self.avgpool_1d = nn.AdaptiveAvgPool1d(1)
#         self.regression_1d = nn.Linear(512,1)    
        
    def forward(self,x):
        
        x1d_0 = self.conv1d_1(x)
        x1d_1 = self.conv1d_layer1(x1d_0)
        x1d_2 = self.conv1d_layer2(x1d_1)
        x1d_3 = self.conv1d_layer3(x1d_2)
        x1d_4 = self.conv1d_layer4(x1d_3)

        feat1d = self.avgpool_1d(x1d_4)
        feat1d = torch.flatten(feat1d, 1)
        
#         o1d = self.regression_1d(feat1d)
        
        return x1d_1,x1d_2,x1d_3,x1d_4,feat1d

class fusion(nn.Module):
        def __init__(self,inplace,outplace,first):
            super().__init__()

            self.first = first

            self.conv = nn.Conv1d(8 * inplace, outplace,1,stride=1,bias=False)
            self.bn = nn.InstanceNorm1d(outplace)
            self.relu = nn.ReLU(inplace=True)
            if self.first:
                self.conv2 = nn.Conv1d(outplace, outplace,1,stride=1,bias=False)
            else:
                self.conv2 = nn.Conv1d(outplace+inplace, outplace,1,stride=1,bias=False)

            self.bn2 = nn.InstanceNorm1d(outplace)
            
            self.kernel_se_conv = nn.Conv1d(4*inplace,2*inplace,1,stride=1)

            self.convs1 = nn.Conv1d(2*inplace,2*inplace,1,stride=1,bias=False)
            self.convs2 = nn.Conv1d(2*inplace,2*inplace,1,stride=1,bias=False)
            self.convm1 = nn.Conv1d(2*inplace,2*inplace,1,stride=1,bias=False)
            self.convm2 = nn.Conv1d(2*inplace,2*inplace,1,stride=1,bias=False)

            #self.softmax = nn.Softmax(dim=1)

        def forward(self,x1,x2,x3,z=None):
            x = torch.cat([x1,x2],1)
            x = self.kernel_se_conv(x)
            x = torch.sigmoid(x)
            h = torch.cat([x1,x3],1)
            h = self.kernel_se_conv(h)
            h = torch.sigmoid(h)
            
            y1 = (1-x-h) * x1 + x * x2 + h * x3
            y2 = torch.sigmoid(self.convs1(x1)) * x2
            y3 = torch.sigmoid(self.convs2(x2)) * x1
            y4 = torch.sigmoid(self.convs2(x3)) * x3
#             y4 = torch.max(self.convm1(x1),self.convm2(x2))

            y = torch.cat([y1,y2,y3,y4],1)
            y = self.relu(self.bn(self.conv(y)))
            if self.first:
                y = self.relu(self.bn2(self.conv2(y)))
            else:
                y = torch.cat([y,z],1)
                y = self.relu(self.bn2(self.conv2(y)))

            return y

class fusNet(nn.Module):
        def __init__(self,num_classes=1):
            super().__init__()

            self.net1 = Resnet(num_classes)
            self.net2 = Resnet(num_classes)
            self.net3 = Resnet(num_classes)

            self.fus1 = fusion(64,128,True)
            self.fus2 = fusion(128,256,False)
            self.fus3 = fusion(256,512,False)


            self.maxp = nn.MaxPool1d(3, stride=2)
            self.avgpool_1d = nn.AdaptiveAvgPool1d(1)
#             self.regression_1d = nn.Linear(512,num_classes)

        def forward(self,ix1,ix2,ix3):
            x1,x2,x3,x4,f1 = self.net1(ix1)
            y1,y2,y3,y4,f2 = self.net2(ix2)
            g1,g2,g3,g4,f3 = self.net3(ix3)
            
#             print(x1.size(),y1.size(),x2.size(),y2.size(),x3.size(),y3.size(),x4.size(),y4.size(),f1.size(),f2.size())
            z = self.fus1(x1,y1,g1)
            z = self.maxp(z)
            z = self.fus2(x2,y2,g2,z)
            z = self.maxp(z)
            z = self.fus3(x3,y3,g3,z)
            z = self.maxp(z)

            z = self.avgpool_1d(z)
            z = torch.flatten(z,1)
            fus = z
#             f3 = self.regression_1d(z)
#             print('FT1 size',f1.size(),'FT2 size',f2.size(),'FT3 size',f3.size())
            
            return f1,f2,f3,fus