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

class LSTMCritic(torch.nn.Module):
    def __init__(self,hiddenSize=350,isProb=False):
        super().__init__()
        self.isProb = isProb
        self.rnn = torch.nn.LSTM(1,hiddenSize,2,batch_first=True,bidirectional=True)
        self.out = torch.nn.Linear(hiddenSize*2,1)
        
    def forward(self,ts):
        output,(h_n,c_n) = self.rnn(ts.float())
        output = self.out(output)
        if self.isProb: output = torch.sigmoid(output)
        return output.squeeze(-1).mean(dim=1)

class LSTMGen(torch.nn.Module):
    def __init__(self,hiddenSize=350):
        super().__init__()
        self.rnn = torch.nn.LSTM(15,hiddenSize,2,batch_first=True)
        self.out = torch.nn.Linear(hiddenSize,1)
        
    def forward(self,noise):
        output,(h_n,c_n) = self.rnn(noise.float())
        output = self.out(output)
        return output

def conv_layer_1d(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias:bool=None,
               norm_type:Optional[NormType]=NormType.Batch,  use_activ:bool=True, leaky:float=None,
               transpose:bool=False, init:Callable=nn.init.kaiming_normal_, self_attention:bool=False):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None: padding = (ks-1)//2 if not transpose else 0
    bn = norm_type in (NormType.Batch, NormType.BatchZero)
    if bias is None: bias = not bn
    conv_func = nn.ConvTranspose1d if transpose else nn.Conv1d
    conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding), init)
    if   norm_type==NormType.Weight:   conv = weight_norm(conv)
    elif norm_type==NormType.Spectral: conv = spectral_norm(conv)
    layers = [conv]
    if use_activ: layers.append(relu(True, leaky=leaky))
    if bn: layers.append(nn.BatchNorm1d(nf))
    return nn.Sequential(*layers)

class CNNCritic(torch.nn.Module):
    def __init__(self,in_size,n_features=64):
        super().__init__()
        layers = [conv_layer(1,n_features,4,2,1,leaky=0.2,norm_type=None,is_1d=True)]
        cur_size, cur_ftrs = in_size//2, n_features
        while cur_size > 4:
            layers.append(conv_layer(cur_ftrs,cur_ftrs*2,4,2,1,leaky=0.2,is_1d=True))
            cur_ftrs *= 2
            cur_size //= 2
        layers.append(nn.Conv1d(cur_ftrs,1,cur_size,padding=0))
        self.conv = nn.Sequential(*layers)
        
    def forward(self,ts):
        ts = self.conv(ts)
        return ts.flatten()

d = CNNCritic(100)

d(torch.zeros(2,1,100).uniform_())

simDataX = np.linspace(0,2*np.pi,100)
amp,phase = np.random.rand(10000,1), np.random.rand(10000,1) * 2 * np.pi
phase = 1
amp = np.ones((10000,1))
simDataY = amp * np.sin(phase * simDataX.reshape(1,-1))

plt.plot(simDataX,simDataY[3])

class simDS(torch.utils.data.Dataset):
    def __init__(self, ts, noise_size):
        self.ts = ts
        self.noise_size = noise_size
        self.items = [1,2,3]
        
    def __len__(self): return len(self.ts)
    
    def __getitem__(self,idx):
        return np.random.randn(self.noise_size,15), np.expand_dims(self.ts[idx],-1)

# +
class TS():
    def __init__(self,ts):
        self.ts = ts
        
    def show(self,ax,title):
        ax.plot(simDataX,self.ts)
        ax.set_title(title)

class PlotCB(Callback):
    def __init__(self,gan):
        self.gan = gan
        self.noise = torch.zeros(1,100,15).normal_().cuda()
        self.images = []
        self.titles = []
        
    def on_epoch_end(self,pbar,epoch,**kwargs):
        a = self.gan.model.generator(self.noise).cpu().detach().numpy().flatten()
        self.images.append(TS(a))
        self.titles.append(f"Epoch {epoch}")
        pbar.show_imgs(self.images,self.titles)
# -

trainDS = simDS(simDataY[:8000],100)
valDS = simDS(simDataY[8000:],100)
data = DataBunch.create(trainDS,valDS,num_workers=0,bs=32)

gan = GANLearner.wgan(data,LSTMGen(),LSTMCritic(),switch_eval=False,show_img=False,
                     opt_func = partial(optim.Adam, betas = (0.,0.99)), wd=0.)

x,y = next(iter(data.train_dl))
gan.model.critic(gan.model.generator(x)).size()

gan.fit(100,2e-4,callbacks=[PlotCB(gan)])

def gen_loss(fake_pred,target,output):
    return torch.log(1-fake_pred).mean()

def crit_loss(real_pred,fake_pred): 
    loss = -torch.log(real_pred) - (torch.log(1-fake_pred))
    return loss.mean()

learn = GANLearner(data,LSTMGen(),LSTMCritic(isProb=True),gen_loss_func=gen_loss,crit_loss_func=crit_loss,
                  switch_eval=False,show_img=False,opt_func = partial(optim.Adam, betas = (0.,0.99)), wd=0.1)

learn.fit(30,1e-3,callbacks=[PlotCB(learn),GANDiscriminativeLR(learn,10)])

a = gan.model.generator(torch.zeros(1,100,1).uniform_().cuda()).cpu().detach().numpy()
#a = gan.model.generator(x).cpu().detach().numpy()

plt.plot(simDataX,a[0].flatten())



d = LSTMCritic()

d(torch.zeros(5,100,1).uniform_())
