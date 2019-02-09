# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import pdb
from pathlib import Path
from fastai.vision import *
from fastai.vision.gan import *

simDataX = np.linspace(0,2*np.pi,100)
amp,phase = np.random.rand(10000,1), np.random.rand(10000,1) * np.pi
simDataY = amp * np.sin(phase * simDataX.reshape(1,-1))

# +
yT = torch.tensor(simDataY,dtype=torch.float).unsqueeze(-1)
phaseT = torch.tensor(phase,dtype=torch.float).squeeze(1)

trainDS = torch.utils.data.TensorDataset(yT[:8000],yT[:8000].squeeze(-1))
valDS = torch.utils.data.TensorDataset(yT[8000:],yT[8000:].squeeze(-1))
trainDS.items = [1,2,3]
valDS.items = [1,2,3]
data = DataBunch.create(trainDS,valDS,num_workers=0)
# -

class SineEnc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(1,350,2,batch_first=True,bias=False)
        self.attn = torch.nn.Linear(350,1,bias=False)
        self.mean = torch.nn.Linear(350,20)
        self.var = torch.nn.Linear(350,20)
        
    def forward(self,ts):
        output,(h_n,c_n) = self.rnn(ts.float())
        #a = torch.softmax(self.attn(output),dim=1)
        #self.attns = a
        #output = a * output
        #output = output.sum(dim=1)
        output = output[:,-1]
        return self.mean(output), self.var(output)

class SineEnc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(1,350,batch_first=True)
        self.hiddenAttn = torch.nn.Linear(350,100,bias=False)
        self.lastAttn = torch.nn.Linear(350,100)
        self.attn = torch.nn.Linear(100,1,bias=False)
        self.mean = torch.nn.Linear(350,20)
        self.var = torch.nn.Linear(350,20)
        
    def forward(self,ts):
        output, (h,c) = self.rnn(ts)
        lastWeight = self.lastAttn(output[:,-1])
        hiddenWeight = self.hiddenAttn(output)
        u = torch.tanh(hiddenWeight + lastWeight.unsqueeze(1))
        a = torch.softmax(self.attn(u),0)
        output = (a*output).sum(dim=1)
        return self.mean(output), self.var(output)

class SineDec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(20,200,1,batch_first=True,bias=False)
        self.out = torch.nn.Linear(200,1)
    
    def forward(self, mean, var):
#         sample = mean
#         if self.training:
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        sample = eps.mul(std).add_(mean)
        
        sample = sample.unsqueeze(1).expand(-1,100,-1)
        output, (h_n,c_n) = self.rnn(sample)
        return self.out(output).squeeze(-1)
        #return output.squeeze(-1)

class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = SineEnc()
        self.dec = SineDec()
        
    def forward(self,ts):
        mean, var = self.enc(ts)
        out = self.dec(mean,var)
        return out,mean,var

def VAELoss(p,target):
    pred,mean,var = p
    mse = torch.nn.functional.mse_loss(pred,target,reduction="sum")
    kld = -0.5 * torch.sum(1+var-mean.pow(2)-var.exp())
    return mse + kld

model = VAE()
learn = Learner(data,model,loss_func=VAELoss)

x,y = next(iter(data.train_dl))
learn.model(x)[0].size()

learn.lr_find()
learn.recorder.plot()

learn.fit(10,1e-3)

# +
learn.model.eval()
out,_,_ = learn.model(x)
_,axes = plt.subplots(3,3,figsize=(20,16))

for o,r,ax in zip(x,out,axes.flatten()):
    ax.plot(o.flatten())
    ax.plot(r)
