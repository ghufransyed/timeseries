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
from sklearn.decomposition import PCA

simDataX = np.linspace(0,2*np.pi,100)
amp,phase = np.random.rand(10000,1), np.random.rand(10000,1) * np.pi
simDataY = amp * np.sin(phase * simDataX.reshape(1,-1))

# +
yT = torch.tensor(simDataY,dtype=torch.float).unsqueeze(-1)
phaseT = torch.tensor(phase,dtype=torch.float).squeeze(1)

trainDS = torch.utils.data.TensorDataset(yT[:8000],phaseT[:8000])
valDS = torch.utils.data.TensorDataset(yT[8000:],phaseT[8000:])
trainDS.items = [1,2,3]
valDS.items = [1,2,3]
data = DataBunch.create(trainDS,valDS,num_workers=0)
# -

class PhaseFinder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(1,350,4,batch_first=True,dropout=0.5,bias=False)
        self.attn = torch.nn.Linear(350,1,bias=False)
        self.out = torch.nn.Linear(350,1)
        
    def forward(self,ts):
        output,(h_n,c_n) = self.rnn(ts.float())
        a = torch.softmax(self.attn(output),dim=1)
        self.attns = a
        output = a * output
        output = output.sum(dim=1)
        return self.out(output).squeeze(1)
        #return self.out(output).mean(dim=1).squeeze(1)
        #return self.out(h_n[-1]).squeeze(1)

model = PhaseFinder()
learn = Learner(data,model,loss_func=torch.nn.functional.mse_loss)
learn.split(split_model_idx(learn.model,idxs=[0,1,2]))

x,y = next(iter(data.train_dl))
learn.model(x).size()

learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(10,1e-4)#[1e-4,1e-3,1e-2])

def predict(a,p):
    ts = a * np.sin(p * simDataX)
    ts = torch.tensor(ts,dtype=torch.float)
    learn.model.eval()
    return ts,learn.model(ts[None,:,None].cuda())

def getActivations(a,p):
    ts = a * np.sin(p * simDataX)
    ts = torch.tensor(ts,dtype=torch.float)
    learn.model.eval()
    return ts,learn.model.rnn(ts[None,:,None].cuda())[0]

fig, axs = plt.subplots(3,3,figsize=(20,16))
for ax in axs.flatten():
    a = np.random.rand()
    p = np.random.rand() * np.pi
    ts,pred = predict(a,p)
    ax.plot(ts)
    at = learn.model.attns.flatten()
    at = at / at.max()
    ax.plot(a * at)
    ax.set_title(f"Freq: {p:.2f} Prediction: {pred.item():.2f}")

ts,output = getActivations(.5,2)
output = output.squeeze(0).cpu().detach().numpy()

pca = PCA(n_components=1)
pca.fit(output)
pX = pca.transform(output)

pX.shape

plt.plot(ts)
plt.plot(pX.flatten())
at = learn.model.attns.flatten()
at = at / at.max()
plt.plot(at)

pts = []
for i in range(output.shape[2]):
    pca = PCA(n_components=1)
    w = output[]
