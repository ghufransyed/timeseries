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
amp,freq = np.random.rand(10000,1), np.random.rand(10000,1) * np.pi
simDataY = amp * np.sin(freq * simDataX.reshape(1,-1))
ls = np.concatenate([amp,freq],axis=1)

# +
yT = torch.tensor(simDataY,dtype=torch.float)
lsT = torch.tensor(ls,dtype=torch.float)

trainDS = torch.utils.data.TensorDataset(lsT[:8000],yT[:8000].squeeze(-1))
valDS = torch.utils.data.TensorDataset(lsT[8000:],yT[8000:].squeeze(-1))
trainDS.items = [1,2,3]
valDS.items = [1,2,3]
data = DataBunch.create(trainDS,valDS,num_workers=0)
# -

class SineDec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(2,200,1,batch_first=True,bias=False)
        self.out = torch.nn.Linear(200,1)
    
    def forward(self, ls):
        ls = ls.unsqueeze(1).expand(-1,100,-1)
        output, _ = self.rnn(ls)
        return self.out(output).squeeze(-1)

model = SineDec()
learn = Learner(data,model,loss_func=torch.nn.functional.mse_loss)

x,y = next(iter(data.train_dl))
model(x).size()

learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(10,1e-3)

# +
learn.model.eval()
out = learn.model(x)
_,axes = plt.subplots(3,3,figsize=(20,16))

for o,r,ax in zip(y,out,axes.flatten()):
    ax.plot(o.flatten())
    ax.plot(r)
