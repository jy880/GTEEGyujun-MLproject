from re import L
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from torch import optim
from torch.utils import data
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


import statistics
import datetime
import os
import csv
import math
import time
import numpy as np
import os

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

os.getcwd()

start = time.time()

idvg_temp=pd.read_csv(r'./idvg_iwo_0206.csv', encoding='utf8')
cv_temp=pd.read_csv(r'./cv_iwo_0206.csv', encoding='utf8')
# idvg=idvg_temp.values

lch = [0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.09]
vd_temp=[0.01, 0.012, 0.015, 0.018, 0.022, 0.027, 0.034, 0.041, 0.050, 0.062, 0.075, 0.092, 0.113, 0.138, 0.169, 0.207, 0.253, 0.310, 0.379, 0.464, 0.568, 0.695, 0.851, 1.042, 1.275, 1.560, 1.9600, 2.337, 2.86, 3.4]
vd = np.array(vd_temp)
vg_temp=idvg_temp.iloc[:,0]
vg = np.array(vg_temp.values)
lch = np.array(lch)

def Logset(target):
    temp = np.array(target)
    # temp[temp<0]=abs(temp)
    # temp = temp.tolist() not use
    temp = np.log10(temp)
    return temp

It = []
for l in list(range(len(lch))):
    for i in list(range(len(vd))):
        temp = idvg_temp.iloc[:, 2*i+1+2*len(vd)*l]
        temp = np.array(temp.values)
        It.extend(temp)

It = Logset(It)
# vd = Logset(vd)


def normaliz(target): #Minmax normalization
    Min = min(target)
    Val = target-Min
    Val = Val
    Max = max(Val)
    Norm = 1/Max
    return (Norm, Val, Min)

(normVg, Vg_1, MinVg)=normaliz(vg)
(normVd, Vd_1, MinVd)=normaliz(vd)
(normIt, It_1, MinIt)=normaliz(It)
(normLch, Lch_1, MinLch) = normaliz(lch)

Vg = normVg*Vg_1
Vd = normVd*Vd_1
I = normIt*It_1
Lch = normLch*Lch_1

datasets = []
for l in list(range(len(Lch))):
    for i in list(range(len(vd))):
        for j in list(range(len(vg))):
            temp=[Vg[j],Vd[i],Lch[l], I[j+len(vg)*(i+len(vd)*l)]]
            datasets.append(temp)

V = []
for i in list(range(len(datasets))):
    temp = [datasets[i][0], datasets[i][1], datasets[i][2]]
    V.append(temp)

I = []
for i in list(range(len(datasets))):
    temp = [datasets[i][3]]
    I.append(temp)

V = torch.tensor(V)
I = torch.tensor(I)

# dataset = list(zip(V, I))
x_train, x_test, y_train, y_test = train_test_split(V, I, test_size=0.1, random_state=41)
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size = 32)
testdataloader = DataLoader(TensorDataset(x_test, y_test))

# print(idvg_temp.values)
It_g = [10**x for x in It]
CM_git = np.corrcoef(vg,It_g[len(vg)*10:len(vg)*11])
Itd=[]
print(len(It_g))
print(len(vg))
print(len(vd))
print(len(lch))

for i in list(range(len(vd))):
    Itd.append(It_g[len(vg)-23+len(vg)*i])
print(Itd)
CM_dit = np.corrcoef(vd, Itd)

print(lch)
print()
Itl = []
print(It_g[len(vg)*len(vd)-4])
print(It_g[len(vg)*len(vd)*8-4])
print(list(range(len(lch))))
for i in list(range(len(lch))):
    Itl.append(It[len(vg)*len(vd)*(i+1)-20] )
print(Itl)
CM_lit = np.corrcoef(Lch, Itl)

print(CM_git)
print(CM_dit)
print(CM_lit)

# Define the neural network class
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(3, 25)
        self.fc2 = torch.nn.Linear(25, 12)
        # self.fc3 = torch.nn.Linear(20, 5)
        self.fc3 = torch.nn.Linear(12, 1)
        self.tanh = torch.nn.Tanh()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        # x = self.tanh(x)
        # x = self.fc4(x)
        return x

# Create an instance of the MLP class
model = MLP()

torch.nn.init.xavier_uniform(model.fc1.weight)
torch.nn.init.xavier_uniform(model.fc2.weight)
torch.nn.init.xavier_uniform(model.fc3.weight)
# torch.nn.init.xavier_uniform(model.fc4.weight)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# losses = []
# criterion = nn.MSELoss() # <== 파이토치에서 제공하는 평균 제곱 오차 함수\

nb_epochs = 3000
MLoss = [] 
for epoch in range(0, nb_epochs):
     
    current_loss = 0.0
    losses = []
    # Iterate over the dataloader for training data
    for i, data in enumerate(dataloader, 0):
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0],1))

        #zero the gradients
        optimizer.zero_grad()

        #perform forward pass
        outputs = model(inputs)
        L_weight = 3
        #compute loss
        batch_loss = []
        for j in range(inputs.size(0)):
            input_j = inputs[j].reshape((1, inputs.shape[1]))
            if input_j[0,0]>0.3:
                batch_loss.append(L_weight*loss_function(outputs[j], targets[j]))
            else:
                batch_loss.append(loss_function(outputs[j], targets[j]))
        
        loss = torch.stack(batch_loss).mean()

        losses.append(loss.item())

        #perform backward pass
        loss.backward()

        #perform optimization
        optimizer.step()
        # Print statistics
    
    mean_loss = sum(losses)/len(losses)
    scheduler.step(mean_loss)

    print('Loss (epoch: %4d): %.8f' %(epoch+1, mean_loss))
    current_loss = 0.0
    
    MLoss.append(mean_loss)
# Process is complete.
print('Training process has finished.')

torch.save(model, 'IWO_idvg.pt')
torch.save(model.state_dict(), 'IWO_idvg_state_dict.pt')

####### loss vs. epoch #######
xloss = list(range(0, nb_epochs))
plt.plot(xloss, np.log10(MLoss))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.plot(xloss[500:], MLoss[500:])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

with torch.no_grad():
    
    output = []
    # Iterate over the dataloader for training data
    for i, data in enumerate(testdataloader, 0):
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0],1))

        #zero the gradients
        optimizer.zero_grad()

        #perform forward pass
        outputs = model(inputs)
        output.append(outputs)
# Process is complete.
print('Training process has finished.')

output1 = np.power(10, output/normIt+MinIt)
ytest1 = np.power(10, y_test/normIt+MinIt)
plt.scatter(np.log10(output1), np.log10(ytest1))
a = [min(ytest1), max(ytest1)]
b = [min(ytest1), max(ytest1)]
plt.plot(np.log10(a), np.log10(b), 'k')
plt.plot(np.log10(a), np.log10(b), 'k')

plt.xlabel("TCAD [A/um]")
plt.ylabel("ML-Prediction [A/um]")
plt.show()

plt.plot(a,b,'k')
plt.scatter(ytest1,output1)
plt.xlabel("TCAD [A/um]")
plt.ylabel("ML-Prediction [A/um]")
plt.show()


answer_test = [i for sublist in ytest1.tolist() for i in sublist]
print(answer_test)
print(output1)

######################### IDVD Vg=0.8 ##############################
x_test = np.linspace(0.01, 3.4, num=30)
print(x_test)
# x_test = Logset(x_test.tolist())
X= []
# X =[[((x_test[i]/10)-MinV)*normV for i in list(range(len(x_test)))]]
for i in list(range(len(x_test))):
    temp=[(0.8-MinVg)*normVg, (x_test[i]-MinVd)*normVd, 0]
    X.append(temp)

Pred_y=[]
for i in list(range(len(x_test))):
    new_var =  torch.FloatTensor(X[i])
    pred_y=model(new_var).data.numpy()
    # Tolist=pred_y.tolist()
    Pred_y.append(pred_y)

I_pred =[np.power(10, i/normIt+MinIt) for i in Pred_y]

I_pred=np.array(I_pred)

I_final= []
for i in list(range(len(I_pred))):
    I_final.extend(I_pred[i])

######################### vg=1.7 IDVD ##############################

X1= []
# X =[[((x_test[i]/10)-MinV)*normV for i in list(range(len(x_test)))]]
for i in list(range(len(x_test))):
    temp=[(1.7-MinVg)*normVg, (x_test[i]-MinVd)*normVd, 0]
    X1.append(temp)

Pred_y1=[]
for i in list(range(len(x_test))):
    new_var =  torch.FloatTensor(X1[i])
    pred_y15=model(new_var).data.numpy()
    # Tolist=pred_y.tolist()
    Pred_y1.append(pred_y15)

I_pred1 =[np.power(10, i/normIt+MinIt) for i in Pred_y1]

I_pred1=np.array(I_pred1)

I_final1= []
for i in list(range(len(I_pred1))):
    I_final1.extend(I_pred1[i])

    ######################### vg=2.6 IDVD ##############################

X15= []
# X =[[((x_test[i]/10)-MinV)*normV for i in list(range(len(x_test)))]]
for i in list(range(len(x_test))):
    temp=[(2.6-MinVg)*normVg, (x_test[i]-MinVd)*normVd, 0]
    X15.append(temp)

Pred_y15=[]
for i in list(range(len(x_test))):
    new_var =  torch.FloatTensor(X15[i])
    pred_y15=model(new_var).data.numpy()
    # Tolist=pred_y.tolist()
    Pred_y15.append(pred_y15)

I_pred15 =[np.power(10, i/normIt+MinIt) for i in Pred_y15]

I_pred15=np.array(I_pred15)

I_final15= []
for i in list(range(len(I_pred15))):
    I_final15.extend(I_pred15[i])

############# Vg = 3.5 IDVD ######################

X25= []
# X =[[((x_test[i]/10)-MinV)*normV for i in list(range(len(x_test)))]]
for i in list(range(len(x_test))):
    temp=[(3.5-MinVg)*normVg, (x_test[i]-MinVd)*normVd, 0]
    X25.append(temp)

Pred_y25=[]
for i in list(range(len(x_test))):
    new_var =  torch.FloatTensor(X25[i])
    pred_y25=model(new_var).data.numpy()
    # Tolist=pred_y.tolist()
    Pred_y25.append(pred_y25)

I_pred25 =[np.power(10, i/normIt+MinIt) for i in Pred_y25]

I_pred25=np.array(I_pred25)

I_final25= []
for i in list(range(len(I_pred25))):
    I_final25.extend(I_pred25[i])

#
Vd_test = [0.01, 0.012, 0.015, 0.018, 0.022, 0.027, 0.034, 0.041, 0.050, 0.062, 0.075, 0.092, 0.113, 0.138, 0.169, 0.207, 0.253, 0.310, 0.379, 0.464, 0.568, 0.695, 0.851, 1.042, 1.275, 1.560, 1.9600, 2.337, 2.86, 3.4]

Id05_test =   [0.000002334528,	0.000002794075,	0.0000034787794,	0.0000041579335,	0.0000050547895,	0.0000061619935,	0.0000076860329,	0.0000091797256,	0.00001105585,	0.000013479929,	0.000016007828,	0.000019161787,	0.000022833255,	0.000026916777,	0.000031624447,	0.00003715309,	0.000043519255,	0.000051091182,	0.000059951515,	0.000070285433,	0.000081658741,	0.00009342872,	0.00010362595,	0.00011101203,	0.00011523482,	0.00011676641,	0.00011730475,	0.00011770665,	0.00011801129,	0.00011824027]
Id1_test = [0.0000068904883,	0.0000082636631,	0.000010320372,	0.000012373432,	0.000015105188,	0.000018510854,	0.000023262079,	0.000027993958,	0.000034049704,	0.000042075633,	0.000050709182,	0.000061905496,	0.000075594373,	0.000091696572,	0.0001113915,	0.0001351759,	0.00016354498,	0.00019827644,	0.00024007196,	0.00029160991,	0.00035507471,	0.00043267721,	0.00052608063,	0.00063305815,	0.0007463102,	0.0008524028,	0.00094690883,	0.0010169456,	0.0010548491,	0.0010714827]
Id15_test = [0.000010230849,	0.000012273384,	0.000015334923,	0.000018393846,	0.000022468346,	0.000027555048,	0.000034664693,	0.000041761002,	0.000050865865,	0.00006297397,	0.000076052026,	0.000093097636,	0.0001140765,	0.00013895091,	0.00016967963,	0.00020724046,	0.00025268947,	0.00030923446,	0.00037849644,	0.00046572552,	0.00057601516,	0.00071607664,	0.00089446014,	0.0011176712,	0.0013874073,	0.0016982907,	0.0020232064,	0.0023477774,	0.0026028589,	0.002758771]
Id_test =  [0.000012969341,	0.000015560565,	0.000019445809,	0.000023329203,	0.000028504135,	0.000034968453,	0.000044010611,	0.000053044062,	0.000064646594,	0.000080097706,	0.000096815121,	0.000118649,	0.00014559015,	0.00017764185,	0.00021740052,	0.0002662518,	0.00032573661,	0.0004003178,	0.00049245958,	0.00060963036,	0.00075939393,	0.00095209114,	0.0012021121,	0.0015230151,	0.0019258635,	0.0024179715,	0.0029806776,	0.003633391,	0.0042757708,	0.0047633642]

# x_test = np.power(10, x_test)
    
plt.scatter(Vd_test, Id05_test) ## TCAD
plt.scatter(Vd_test, Id1_test) ## TCAD
plt.scatter(Vd_test, Id15_test) ## TCAD
plt.scatter(Vd_test, Id_test) ## TCAD
plt.plot(x_test, I_final)
plt.plot(x_test, I_final1)
plt.plot(x_test, I_final15)
plt.plot(x_test, I_final25)
plt.xlabel("Drain Voltage [V]")
plt.ylabel("Current [A/um]")
plt.show()

plt.scatter(Vd_test, Id05_test) ## TCAD
plt.scatter(Vd_test, Id1_test) ## TCAD
plt.plot(x_test, I_final)
plt.plot(x_test, I_final1)
plt.xlabel("Drain Voltage [V]")
plt.ylabel("Current [A/um]")
plt.show()
print(np.round(x_test, 3).tolist)
print(I_final)
print(I_final1) 
print(I_final15)
print("")

#################### IDVG #######################

xv_test = list(range(-10, 36, 1))
xv_test = np.array(xv_test)/10

(normVgtest, xv_test_1, MinVtest)=normaliz(xv_test)
Xv= []
# X =[[((x_test[i]/10)-MinV)*normV for i in list(range(len(x_test)))]]
for i in list(range(len(xv_test_1))):
    temp=[(xv_test[i]-MinVg)*normVg, ((0.05)-MinVd)*normVd, 0]
    Xv.append(temp)

Predv_y=[]
for i in list(range(len(xv_test_1))):
    new_var =  torch.FloatTensor(Xv[i])
    pred_y=model(new_var).data.numpy()
    # Tolist=pred_y.tolist()
    Predv_y.append(pred_y)

Iv_pred =[np.power(10, i/normIt+MinIt) for i in Predv_y]

Iv_pred=np.array(Iv_pred)

Iv_final= []
for i in list(range(len(Iv_pred))):
    Iv_final.extend(Iv_pred[i])

#### VD =1.042V ####

Xv11=[]
for i in list(range(len(xv_test))):
    temp=[(xv_test[i]-MinVg)*normVg, ((1.042)-MinVd)*normVd, 0]
    Xv11.append(temp)

Predv_y11=[]
for i in list(range(len(xv_test))):
    new_var =  torch.FloatTensor(Xv11[i])
    pred_y11=model(new_var).data.numpy()
    Predv_y11.append(pred_y11)

Iv_pred11 =[np.power(10, i/normIt+MinIt) for i in Predv_y11]

Iv_pred11=np.array(Iv_pred11)

Iv_final11= []
for i in list(range(len(Iv_pred11))):
    Iv_final11.extend(Iv_pred11[i])

#### VD =3.4V #### 2.337

Xv25=[]
for i in list(range(len(xv_test))):
    temp=[(xv_test[i]-MinVg)*normVg, ((2.337)-MinVd)*normVd, 0]
    Xv25.append(temp)

Predv_y25=[]
for i in list(range(len(xv_test))):
    new_var =  torch.FloatTensor(Xv25[i])
    pred_y25=model(new_var).data.numpy()
    Predv_y25.append(pred_y25)

Iv_pred25 =[np.power(10, i/normIt+MinIt) for i in Predv_y25]

Iv_pred25=np.array(Iv_pred25)

Iv_final25= []
for i in list(range(len(Iv_pred25))):
    Iv_final25.extend(Iv_pred25[i])

##### below is from TCAD #####
Vg_test = [-1,	-0.87142857,	-0.74285714,	-0.61428571,	-0.48571429,	-0.35714286,	-0.22857143,	-0.1,	0.028571429,	0.15714286,	0.28571429,	0.41428571,	0.54285714,	0.67142857,	0.8,	0.92857143,	1.0571429,	1.1857143,	1.3142857,	1.4428571,	1.5714286,	1.7,	1.8285714,	1.9571429,	2.0857143,	2.2142857,	2.3428571,	2.4714286,	2.6,	2.7285714,	2.8571429,	2.9857143,	3.1142857,	3.2428571,	3.3714286,	3.5]

Id0_test = [6.0366509E-24,	8.4496776E-22,	1.1747828E-19,	1.6070645E-17,	2.1422178E-15,	2.4891875E-13,	0.000000000013962995,	0.00000000028716668,	0.0000000044495768,	0.000000052182154,	0.00000031576001,	0.0000012230809,	0.0000036792536,	0.0000071776321,	0.00001105585,	0.000015236741,	0.000018953721,	0.000022321276,	0.000025471855,	0.00002845546,	0.000031308864,	0.000034049704,	0.000036691166,	0.000039244928,	0.000041715704,	0.000044108729,	0.000046429037,	0.000048680267,	0.000050865865,	0.000052991948,	0.000055060637,	0.000057073072,	0.000059038778,	0.000060954363,	0.000062822385,	0.000064646594]
Id_test = [6.5726763E-23,	1.074733E-21,	1.5054616E-19,	2.089954E-17,	2.8457564E-15,	3.4630142E-13,	0.000000000021507479,	0.00000000047443979,	0.0000000076094844,	0.00000012160276,	0.0000013329527,	0.0000068796113,	0.000023222635,	0.000057753922,	0.00011101203,	0.00017662755,	0.0002477421,	0.00032092367,	0.00039619401,	0.00047418914,	0.00055484828,	0.00063305815,	0.00070885986,	0.00078224439,	0.00085324923,	0.00092217874,	0.0009890835,	0.0010542113,	0.0011176712,	0.0011795664,	0.0012400484,	0.0012991997,	0.0013570033,	0.0014135401,	0.00146889,	0.0015230151]
# Id2_test = [0.00000000028999272,	0.00000000024604411,	0.000000000208839,	0.00000000015892722,	0.00000000013323784,	0.00000000012752485,	0.00000000024163604,	0.0000000015000178,	0.00000001459907,	0.00000016488813,	0.0000014600513,	0.0000070961489,	0.000023578158,	0.000059393719,	0.00011824027,	0.00019892949,	0.00029982926,	0.0004191606,	0.00055617887,	0.00071067895,	0.00088251831,	0.0010714827,	0.0012768098,	0.0014977426,	0.0017292861,	0.0019734283,	0.0022272997,	0.0024898736,	0.002758771,	0.0030327668,	0.0033162884,	0.0036011479,	0.0038954135,	0.0041845065,	0.0044747038,	0.0047633642]
Id2_test = [0.000000000025494235,	0.000000000012954164,	5.0756078E-12,	1.3277786E-12,	1.9832705E-13,	3.8441706E-13,	0.000000000022061297,	0.00000000047642503,	0.0000000076135681,	0.00000012164447,	0.0000013351355,	0.0000069088198,	0.000023438983,	0.000059204515,	0.00011770665,	0.00019763123,	0.00029700373,	0.00041351776,	0.0005455993,	0.0006915413,	0.00084927227,	0.0010169456,	0.0011927032,	0.0013746856,	0.0015617533,	0.0017551156,	0.0019548531,	0.0021515877,	0.0023477774,	0.0025415313,	0.0027321298,	0.0029192402,	0.0031029262,	0.0032831873,	0.003459982,	0.003633391]

plt.scatter(Vg_test, np.log10(Id0_test)) ## TCAD
plt.scatter(Vg_test, np.log10(Id_test)) ## TCAD
plt.scatter(Vg_test, np.log10(Id2_test)) ## TCAD
plt.plot(xv_test, np.log10(Iv_final))
plt.plot(xv_test, np.log10(Iv_final11))
plt.plot(xv_test, np.log10(Iv_final25))
plt.xlabel("Gate Voltage [V]")
plt.ylabel("Current [A/um]")
plt.show()

plt.scatter(Vg_test, (Id0_test)) ## TCAD
plt.scatter(Vg_test, (Id_test)) ## TCAD
plt.scatter(Vg_test, (Id2_test)) ## TCAD
plt.plot(xv_test, (Iv_final))
plt.plot(xv_test, (Iv_final11))
plt.plot(xv_test, (Iv_final25))
plt.xlabel("Gate Voltage [V]")
plt.ylabel("Current [A/um]")
plt.show()

print(xv_test)
print(Iv_final)
print(Iv_final11)
print(Iv_final25)
print("")

# Instantiate the PyTorch model
model.load_state_dict(torch.load('IWO_idvg_state_dict.pt'))

# Extract the weights and biases from the model
weights_1 = model.fc1.weight.detach().numpy()
bias_1 = model.fc1.bias.detach().numpy()
weights_2 = model.fc2.weight.detach().numpy()
bias_2 = model.fc2.bias.detach().numpy()
weights_3 = model.fc3.weight.detach().numpy()
bias_3 = model.fc3.bias.detach().numpy()

verilog_code = ""

# Create the Verilog-A code for the 1st hidden layer
verilog_code += "real h1_0, h1_1, h1_2, h1_3, h1_4, h1_5, h1_6, h1_7, h1_8, h1_9, h1_10, h1_11, h1_12, h1_13, h1_14, h1_15, h1_16, h1_17, h1_18, h1_19, h1_20, h1_21, h1_22, h1_23, h1_24;\n"
for i in range(25):
    inputs = ["Vgs", "Vds", "Lg"]
    inputs = ["*".join([str(weights_1[i][j]), inp]) for j, inp in enumerate(inputs)]
    inputs = "+".join(inputs)
    inputs = "+".join([inputs, str(bias_1[i])])
    verilog_code += "h1_{} = tanh({});\n".format(i, inputs)

# Create the Verilog-A code for the 2nd hidden layer
verilog_code += "real h2_0, h2_1, h2_2, h2_3, h2_4, h2_5, h2_6, h2_7, h2_8, h2_9, h2_10, h2_11;\n"
for i in range(12):
    inputs = ["h1_{}".format(j) for j in range(25)]
    inputs = ["*".join([str(weights_2[i][j]), inp]) for j, inp in enumerate(inputs)]
    inputs = "+".join(inputs)
    inputs = "+".join([inputs, str(bias_2[i])])
    verilog_code += "h2_{} = tanh({});\n".format(i, inputs)

# Create the Verilog-A code for the output layer
inputs = ["h2_{}".format(i) for i in range(12)]
inputs = ["*".join([str(weights_3[0][i]), inp]) for i, inp in enumerate(inputs)]
inputs = "+".join(inputs)
inputs = "+".join([inputs, str(bias_3[0])])
verilog_code += "y = {};\n".format(inputs)

verilog_code = """
module IWO_verilogA (d, g, s);
inout d, g, s;
electrical d, g, s;

//****** Parameters L and W ********
parameter real W = 0.1;
parameter real L = 0.05;
parameter MinVg = {} ;
parameter normVg = {} ;
parameter MinVd = {} ;
parameter normVd = {} ;
parameter MinLg = {} ;
parameter normLg = {} ;
parameter MinI = {} ;
parameter normI ={};
real Vg, Vd, Vs, Vgs, Vds, Lg, Id, Cgg, Cgsd, Vgd;
real Vgsraw, Vgdraw, dir;

analog begin
	Vg = V(g);
	Vs = V(s);
	Vd = V(d);
    Vgsraw = Vg-Vs ;
    Vgdraw = Vg-Vd ;
if (Vgsraw>=Vgdraw) begin
	Vgs = ((Vg-Vs) - MinVg) * normVg ;
    dir = 1 ;
end
else begin
	Vgs = ((Vg-Vd) - MinVg) * normVg ;
    dir = -1 ;
end
	Vds = (abs(Vd-Vs) - MinVd) * normVd ;
	Lg = (L -MinLg)*normLg ;


{}

Id = pow(10, (y/normI + MinI))*W;
I(g, d) <+ Cgsd*ddt(Vg-Vd) ;
I(g, s) <+ Cgsd*ddt(Vg-Vs) ;

if (Vd >= Vs) begin
	I(d, s) <+ dir*Id;
end

else begin
	I(d, s) <+ dir*Id;
end

end
endmodule

""".format(MinVg, normVg, MinVd, normVd, MinLch, normLch, MinIt, normIt, verilog_code)

print(verilog_code)

with open("iwo_test.va", "w") as f:
    f.write(verilog_code)

