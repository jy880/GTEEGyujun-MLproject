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
cv_temp=pd.read_csv(r'./cv_iwo_0212.csv', encoding='utf8')
# idvg=idvg_temp.values


lch0 = [0.05, 0.055, 0.06, 0.65, 0.07, 0.75, 0.08, 0.09]
vd_temp=[0.01, 0.012, 0.015, 0.018, 0.022, 0.027, 0.034, 0.041, 0.050, 0.062, 0.075, 0.092, 0.113, 0.138, 0.169, 0.207, 0.253, 0.310, 0.379, 0.464, 0.568, 0.695, 0.851, 1.042, 1.275, 1.560, 1.9600, 2.337, 2.86, 3.4]
vd = np.array(vd_temp)
vg_temp=cv_temp.iloc[:, 0]
vg = np.array(vg_temp.values)
lch1 = np.array(lch0)

def Logset(target):
    temp = np.array(target)
    # temp[temp<0]=abs(temp)
    # temp = temp.tolist() not use
    temp = np.log10(temp)
    return temp

Ct = []
for l in list(range(len(lch1))):
    for i in list(range(len(vd))):
        temp = cv_temp.iloc[:, 2*i+1+2*len(vd)*l]
        temp = np.array(temp.values)
        Ct.extend(temp)

def normaliz(target): #Minmax normalization
    Min = min(target)
    Val = target-Min
    Max = max(Val)
    Norm = 1/Max
    Fin = Norm*Val
    return (Norm, Val, Min, Fin)

(normVg, Vg_1, MinVg, Vg)=normaliz(vg)
(normVd, Vd_1, MinVd, Vd)=normaliz(vd)
(normCt, Ct_1, MinCt, C0)=normaliz(Ct)
(normLch, Lch_1, MinLch, Lch) = normaliz(lch1)


datasets = []
for l in list(range(len(Lch))):
    for i in list(range(len(vd))):
        for j in list(range(len(vg))):
            temp=[Vg[j],Vd[i],Lch[l], C0[j+len(vg)*(i+len(vd)*l)]]
            datasets.append(temp)

print(np.array(datasets).shape)

V = []
for i in list(range(len(datasets))):
    temp = [datasets[i][0], datasets[i][1], datasets[i][2]]
    V.append(temp)

Cte = []
for i in list(range(len(datasets))):
    temp = [datasets[i][3]]
    Cte.append(temp)

V = torch.tensor(V)
C = torch.tensor(Cte)
print(C.shape)

# dataset = list(zip(V, I))
x_train, x_test, y_train, y_test = train_test_split(V, C, test_size=0.1, random_state=41)
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size = 32)
testdataloader = DataLoader(TensorDataset(x_test, y_test))

# Define the neural network class
class CVMLP(torch.nn.Module):
    def __init__(self):
        super(CVMLP, self).__init__()
        self.fc1 = torch.nn.Linear(3, 17)
        self.fc2 = torch.nn.Linear(17, 17)
        self.fc3 = torch.nn.Linear(17, 1)
        # self.fc3 = torch.nn.Linear(5, 1)
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

# Create an instance of the MLP class now adjusting size 3-30-25-1 --> 3-25-25-1 --> 3-20-20-1 --> 3-15-15-1
model = CVMLP()

torch.nn.init.xavier_uniform_(model.fc1.weight)
# torch.nn.init.xavier_uniform_(model.fc1.bias)
torch.nn.init.xavier_uniform_(model.fc2.weight)
# torch.nn.init.xavier_uniform_(model.fc2.bias)
torch.nn.init.xavier_uniform_(model.fc3.weight)
# torch.nn.init.xavier_uniform_(model.fc3.bias)
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
        L_weight = 1
        #compute loss
        batch_loss = []
        for j in range(inputs.size(0)):
            input_j = inputs[j].reshape((1, inputs.shape[1]))
            # print(input_j)
            if input_j[0,0]>0.4 and input_j[0,1]>0.4:
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

# torch.save(model, 'IWO_idvg.pt')
# torch.save(model.state_dict(), 'IWO_idvg_state_dict.pt')

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

A = [1e-14, 3.7e-14]
Y = [1e-14, 3.7e-14]
output1 = output/normCt+MinCt
y_test1 = y_test/normCt+MinCt
plt.scatter(y_test1, output1)
plt.plot(A,Y, 'k')
plt.xlabel("TCAD [F/um]")
plt.ylabel("ML-Prediction [F/um]")
plt.show()

answer_test = [i for sublist in y_test1.tolist() for i in sublist]
# new_output = [i for sublist in y_test1.tolist() for i in sublist]
print(answer_test)
output_1 = [output_temp.item() for output_temp in output1]
print(output_1)

test = [[0, 0, 0]]
test_var = torch.FloatTensor(test)
pred_test = model(test_var).data.numpy()
print(pred_test/normCt+MinCt)

######################### IDVD ##############################
x_test = list(range(1, 340, 5))
x_test = np.array(x_test)/100

X= []
# X =[[((x_test[i]/10)-MinV)*normV for i in list(range(len(x_test)))]]
for i in list(range(len(x_test))):
    temp=[(0.8-MinVg)*normVg, ((x_test[i])-MinVd)*normVd, 0]
    X.append(temp)

Pred_y=[]
for i in list(range(len(x_test))):
    new_var =  torch.FloatTensor(X[i])
    pred_y=model(new_var).data.numpy()
    # Tolist=pred_y.tolist()
    Pred_y.append(pred_y)

I_pred =[i/normCt+MinCt for i in Pred_y]

I_pred=np.array(I_pred)

I_final= []
for i in list(range(len(I_pred))):
    I_final.extend(I_pred[i])

######################### vg=1 IDVD ##############################

X1= []
# X =[[((x_test[i]/10)-MinV)*normV for i in list(range(len(x_test)))]]
for i in list(range(len(x_test))):
    temp=[(1.7-MinVg)*normVg, ((x_test[i])-MinVd)*normVd, 0]
    X1.append(temp)

Pred_y1=[]
for i in list(range(len(x_test))):
    new_var =  torch.FloatTensor(X1[i])
    pred_y15=model(new_var).data.numpy()
    # Tolist=pred_y.tolist()
    Pred_y1.append(pred_y15)

I_pred1 =[  (i)/normCt+MinCt for i in Pred_y1]

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

I_pred15 =[ i/normCt+MinCt for i in Pred_y15]

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

I_pred25 =[ i/normCt+MinCt for i in Pred_y25]

I_pred25=np.array(I_pred25)

I_final25= []
for i in list(range(len(I_pred25))):
    I_final25.extend(I_pred25[i])

Vd_test = [0.01, 0.012, 0.015, 0.018, 0.022, 0.027, 0.034, 0.041, 0.050, 0.062, 0.075, 0.092, 0.113, 0.138, 0.169, 0.207, 0.253, 0.310, 0.379, 0.464, 0.568, 0.695, 0.851, 1.042, 1.275, 1.560, 1.9600, 2.337, 2.86, 3.4]

Id05_test =   [0.0000000000000261,	0.0000000000000261,	0.0000000000000261,	0.000000000000026,	0.000000000000026,	0.0000000000000259,	0.0000000000000258,	0.0000000000000257,	0.0000000000000256,	0.0000000000000254,	0.0000000000000252,	0.0000000000000249,	0.0000000000000245,	0.0000000000000241,	0.0000000000000235,	0.0000000000000228,	0.0000000000000219,	0.0000000000000207,	0.0000000000000196,	0.0000000000000188,	0.0000000000000184,	0.0000000000000179,	0.0000000000000175,	0.000000000000017,	0.0000000000000163,	0.0000000000000158,	0.0000000000000156,	0.0000000000000155,	0.0000000000000155,	0.0000000000000154]
Id1_test = [0.0000000000000319,	0.0000000000000319,	0.0000000000000319,	0.0000000000000319,	0.0000000000000319,	0.0000000000000319,	0.0000000000000319,	0.0000000000000319,	0.0000000000000319,	0.0000000000000319,	0.0000000000000319,	0.0000000000000318,	0.0000000000000318,	0.0000000000000318,	0.0000000000000317,	0.0000000000000317,	0.0000000000000316,	0.0000000000000316,	0.0000000000000314,	0.0000000000000313,	0.000000000000031,	0.0000000000000306,	0.0000000000000298,	0.0000000000000279,	0.0000000000000241,	0.0000000000000218,	0.0000000000000205,	0.0000000000000194,	0.0000000000000187,	0.0000000000000182]
Id15_test = [0.0000000000000331,	0.0000000000000331,	0.0000000000000331,	0.0000000000000331,	0.0000000000000331,	0.0000000000000331,	0.0000000000000331,	0.0000000000000331,	0.0000000000000331,	0.000000000000033,	0.000000000000033,	0.000000000000033,	0.000000000000033,	0.000000000000033,	0.000000000000033,	0.000000000000033,	0.000000000000033,	0.0000000000000329,	0.0000000000000329,	0.0000000000000328,	0.0000000000000328,	0.0000000000000327,	0.0000000000000326,	0.0000000000000324,	0.0000000000000321,	0.0000000000000315,	0.0000000000000294,	0.0000000000000237,	0.0000000000000212,	0.0000000000000204]
Id_test =  [0.0000000000000337,	0.0000000000000337,	0.0000000000000337,	0.0000000000000337,	0.0000000000000337,	0.0000000000000337,	0.0000000000000337,	0.0000000000000337,	0.0000000000000337,	0.0000000000000337,	0.0000000000000337,	0.0000000000000337,	0.0000000000000337,	0.0000000000000337,	0.0000000000000337,	0.0000000000000336,	0.0000000000000336,	0.0000000000000336,	0.0000000000000336,	0.0000000000000336,	0.0000000000000335,	0.0000000000000335,	0.0000000000000334,	0.0000000000000334,	0.0000000000000332,	0.0000000000000331,	0.0000000000000328,	0.0000000000000322,	0.0000000000000292,	0.0000000000000233]

plt.scatter(Vd_test, Id05_test) ## TCAD
plt.scatter(Vd_test, Id1_test) ## TCAD
plt.scatter(Vd_test, Id15_test) ## TCAD
plt.scatter(Vd_test, Id_test) ## TCAD
plt.plot(x_test, I_final)
plt.plot(x_test, I_final1)
plt.plot(x_test, I_final15)
plt.plot(x_test, I_final25)
plt.xlabel("Drain Voltage [V]")
plt.ylabel("Capacitance [F/um]")
plt.show()

plt.scatter(Vd_test, Id05_test) ## TCAD
plt.scatter(Vd_test, Id1_test) ## TCAD
plt.plot(x_test, I_final)
plt.plot(x_test, I_final1)
plt.xlabel("Drain Voltage [V]")
plt.ylabel("Capacitance [F/um]")
plt.show()

#################### IDVG005 #######################

xv_test = list(range(-10, 36, 1))
xv_test = np.array(xv_test)/10

Xv= []
# X =[[((x_test[i]/10)-MinV)*normV for i in list(range(len(x_test)))]]
for i in list(range(len(xv_test))):
    temp=[(xv_test[i]-MinVg)*normVg, (0.05-MinVd)*normVd, 0]
    Xv.append(temp)

Predv_y=[]
for i in list(range(len(xv_test))):
    new_var =  torch.FloatTensor(Xv[i])
    pred_y=model(new_var).data.numpy()
    # Tolist=pred_y.tolist()
    Predv_y.append(pred_y)

Iv_pred =[(i/normCt+MinCt) for i in Predv_y]

Iv_pred=np.array(Iv_pred)

Iv_final= []
for i in list(range(len(Iv_pred))):
    Iv_final.extend(Iv_pred[i])

#### VD =1.9V ####

Xv11=[]
for i in list(range(len(xv_test))):
    temp=[(xv_test[i]-MinVg)*normVg, (1.042-MinVd)*normVd, 0]
    Xv11.append(temp)

Predv_y11=[]
for i in list(range(len(xv_test))):
    new_var =  torch.FloatTensor(Xv11[i])
    pred_y11=model(new_var).data.numpy()
    Predv_y11.append(pred_y11)

Iv_pred11 =[(i/normCt+MinCt) for i in Predv_y11]

Iv_pred11=np.array(Iv_pred11)

Iv_final11= []
for i in list(range(len(Iv_pred11))):
    Iv_final11.extend(Iv_pred11[i])

#### VD =2.337V ####

Xv34=[]
for i in list(range(len(xv_test))):
    temp=[(xv_test[i]-MinVg)*normVg, (2.337-MinVd)*normVd, 0]
    Xv34.append(temp)

Predv_y34=[]
for i in list(range(len(xv_test))):
    new_var =  torch.FloatTensor(Xv34[i])
    pred_y34=model(new_var).data.numpy()
    Predv_y34.append(pred_y34)

Iv_pred34 =[(i/normCt+MinCt) for i in Predv_y34]

Iv_pred34=np.array(Iv_pred34)

Iv_final34= []
for i in list(range(len(Iv_pred34))):
    Iv_final34.extend(Iv_pred34[i])

##### below is from TCAD #####
Vg_test = [-1,	-0.87142857,	-0.74285714,	-0.61428571,	-0.48571429,	-0.35714286,	-0.22857143,	-0.1,	0.028571429,	0.15714286,	0.28571429,	0.41428571,	0.54285714,	0.67142857,	0.8,	0.92857143,	1.0571429,	1.1857143,	1.3142857,	1.4428571,	1.5714286,	1.7,	1.8285714,	1.9571429,	2.0857143,	2.2142857,	2.3428571,	2.4714286,	2.6,	2.7285714,	2.8571429,	2.9857143,	3.1142857,	3.2428571,	3.3714286,	3.5]

Id0_test = [0.00000000000000816,	0.00000000000000816,	0.00000000000000816,	0.00000000000000816,	0.00000000000000816,	0.00000000000000816,	0.00000000000000816,	0.00000000000000816,	0.00000000000000845,	0.00000000000000973,	0.0000000000000104,	0.0000000000000115,	0.0000000000000157,	0.0000000000000211,	0.0000000000000256,	0.0000000000000282,	0.0000000000000295,	0.0000000000000303,	0.0000000000000309,	0.0000000000000313,	0.0000000000000316,	0.0000000000000319,	0.0000000000000321,	0.0000000000000323,	0.0000000000000325,	0.0000000000000327,	0.0000000000000328,	0.0000000000000329,	0.0000000000000331,	0.0000000000000332,	0.0000000000000333,	0.0000000000000334,	0.0000000000000335,	0.0000000000000335,	0.0000000000000336,	0.0000000000000337]
Id_test = [0.00000000000000816,	0.00000000000000816,	0.00000000000000816,	0.00000000000000816,	0.00000000000000816,	0.00000000000000816,	0.00000000000000816,	0.00000000000000816,	0.00000000000000816,	0.00000000000000819,	0.00000000000000841,	0.00000000000000917,	0.0000000000000118,	0.0000000000000147,	0.000000000000017,	0.0000000000000184,	0.0000000000000195,	0.0000000000000206,	0.0000000000000218,	0.0000000000000231,	0.0000000000000257,	0.0000000000000279,	0.0000000000000295,	0.0000000000000305,	0.0000000000000312,	0.0000000000000316,	0.0000000000000319,	0.0000000000000322,	0.0000000000000324,	0.0000000000000326,	0.0000000000000327,	0.0000000000000329,	0.000000000000033,	0.0000000000000331,	0.0000000000000333,	0.0000000000000334]
Id34_test = [0.00000000000000945,	0.00000000000000918,	0.00000000000000896,	0.00000000000000877,	0.00000000000000862,	0.00000000000000848,	0.00000000000000836,	0.00000000000000827,	0.00000000000000821,	0.00000000000000819,	0.00000000000000828,	0.00000000000000898,	0.0000000000000113,	0.0000000000000136,	0.0000000000000154,	0.0000000000000164,	0.0000000000000169,	0.0000000000000173,	0.0000000000000176,	0.0000000000000178,	0.000000000000018,	0.0000000000000182,	0.0000000000000185,	0.0000000000000188,	0.0000000000000191,	0.0000000000000194,	0.0000000000000197,	0.0000000000000201,	0.0000000000000204,	0.0000000000000207,	0.000000000000021,	0.0000000000000213,	0.0000000000000217,	0.0000000000000222,	0.0000000000000227,	0.0000000000000233]

#VD = 0.05, 1.042, 3.4V

plt.scatter(Vg_test, (Id0_test)) ## TCAD
plt.scatter(Vg_test, (Id_test)) ## TCAD
plt.scatter(Vg_test, Id34_test)
plt.plot(xv_test, (Iv_final))
plt.plot(xv_test, (Iv_final11))
plt.plot(xv_test, (Iv_final34))
plt.xlabel("Gate Voltage [V]")
plt.ylabel("Capacitance [F/um]")
plt.show()

print(xv_test)
print(Iv_final)
print(Iv_final11)
print(Iv_final34)

torch.save(model, 'IWO_CV.pt')
torch.save(model.state_dict(), 'IWO_CV_state_dict.pt')

# Instantiate the PyTorch model
model.load_state_dict(torch.load('IWO_CV_state_dict.pt'))

# Extract the weights and biases from the model
weights_1 = model.fc1.weight.detach().numpy()
bias_1 = model.fc1.bias.detach().numpy()
weights_2 = model.fc2.weight.detach().numpy()
bias_2 = model.fc2.bias.detach().numpy()
weights_3 = model.fc3.weight.detach().numpy()
bias_3 = model.fc3.bias.detach().numpy()

verilog_code = ""

# Create the Verilog-A code for the 1st hidden layer
verilog_code += "real hc1_0, hc1_1, hc1_2, hc1_3, hc1_4, hc1_5, hc1_6, hc1_7, hc1_8, hc1_9;\n"
verilog_code += "real hc1_10, hc1_11, hc1_12, hc1_13, hc1_14, hc1_15, hc1_16;\n"
for i in range(17):
    inputs = ["Vgs", "Vds", "Lg"]
    inputs = ["*".join([str(weights_1[i][j]), inp]) for j, inp in enumerate(inputs)]
    inputs = "+".join(inputs)
    inputs = "+".join([inputs, str(bias_1[i])])
    verilog_code += "hc1_{} = tanh({});\n".format(i, inputs)

# Create the Verilog-A code for the 2nd hidden layer
verilog_code += "real hc2_0, hc2_1, hc2_2, hc2_3, hc2_4, hc2_5, hc2_6, hc2_7, hc2_8, hc2_9;\n"
verilog_code += "real hc2_10, hc2_11, hc2_12, hc2_13, hc2_14, hc2_15, hc2_16;\n"
for i in range(17):
    inputs = ["hc1_{}".format(j) for j in range(10)]
    inputs = ["*".join([str(weights_2[i][j]), inp]) for j, inp in enumerate(inputs)]
    inputs = "+".join(inputs)
    inputs = "+".join([inputs, str(bias_2[i])])
    verilog_code += "hc2_{} = tanh({});\n".format(i, inputs)

# Create the Verilog-A code for the output layer
inputs = ["hc2_{}".format(i) for i in range(5)]
inputs = ["*".join([str(weights_3[0][i]), inp]) for i, inp in enumerate(inputs)]
inputs = "+".join(inputs)
inputs = "+".join([inputs, str(bias_3[0])])
verilog_code += "yc = {};\n".format(inputs)

verilog_code = """
module neural_network (d, g, s);
inout d, g, s;
electrical d, g, s;

//****** Parameters L and W ********
parameter MinVg = {} ;
parameter normVg = {} ;
parameter MinVd = {} ;
parameter normVd = {} ;
parameter MinLg = {} ;
parameter normLg = {} ;
parameter MinO = {} ;
parameter normO ={};

analog begin
	Vg = V(g);
	Vs = V(s);
	Vd = V(d);
if (Vd>=Vs) begin
	Vgs = ((Vg-Vs) - MinVg) * normVg ;
end
else begin
	Vgs = ((Vg-Vd) - MinVg) * normVg ;
end
	Vds = (abs(Vd-Vs) - MinVd) * normVd ;
	Lg = (L -MinLg)*normLg ;

{}

Cgg = (yc/normO + MinO)*W;
Cgsd = Cgg/2;

endmodule

""".format(MinVg, normVg, MinVd, normVd, MinLch, normLch, MinCt, normCt, verilog_code)

print(verilog_code)

with open("iwo_cvtest.va", "w") as f:
    f.write(verilog_code)
