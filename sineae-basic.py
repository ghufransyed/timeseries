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
        self.hidden = torch.nn.Linear(100,50)
        self.mean = torch.nn.Linear(50,10)
        self.var = torch.nn.Linear(50,10)
        
    def forward(self,ts):
        h = torch.relu(self.hidden(ts.squeeze(-1)))
        return self.mean(h), self.var(h)

class SineDec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = torch.nn.Linear(10,50)
        self.out = torch.nn.Linear(50,100)
        
    def forward(self,mean,var):
        sample = mean
        if self.training:
            std = torch.exp(0.5 * var)
            eps = torch.randn_like(std)
            sample = eps.mul(std).add_(mean)
        
        h = torch.relu(self.hidden(sample))
        return torch.tanh(self.out(h))

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
    return mse + 3*kld

model = VAE()
learn = Learner(data,model,loss_func=VAELoss)

x,y = next(iter(data.train_dl))
learn.model(x)[0].shape

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
