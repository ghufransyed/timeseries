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
from pathlib import Path
path = Path("UCRArchive_2018/InsectWingbeatSound")

list(path.iterdir())

trainDF = pd.read_csv(path/"InsectWingbeatSound_TRAIN.tsv",sep="\t",header=None)
valDF = pd.read_csv(path/"InsectWingbeatSound_TEST.tsv",sep="\t",header=None)
print(trainDF.shape,valDF.shape)

trainDS = torch.utils.data.TensorDataset(torch.tensor(trainDF.iloc[:,1:].values,dtype=torch.float),torch.tensor(trainDF.iloc[:,0].values,dtype=torch.long))
valDS = torch.utils.data.TensorDataset(torch.tensor(valDF.iloc[:,1:].values,dtype=torch.float),torch.tensor(valDF.iloc[:,0].values,dtype=torch.long))
trainDS.items, valDS.items = [1,2,3],[1,2,3]
numClasses = len(trainDF.iloc[:,0].unique())
seqLen = trainDF.shape[1]-1
dataClassifier = DataBunch.create(trainDS,valDS,num_workers=0)

tT = torch.tensor(trainDF.iloc[:,1:].values,dtype=torch.float)
vT = torch.tensor(valDF.iloc[:,1:].values,dtype=torch.float)
combT = torch.cat([tT,vT])
trainDS = torch.utils.data.TensorDataset(combT,combT)
valDS = torch.utils.data.TensorDataset(vT,vT)
trainDS.items, valDS.items = [1,2,3],[1,2,3]
dataAE = DataBunch.create(trainDS,valDS,num_workers=0)

# +
def convBlock(nIn,nOut,ks,stride):
    return torch.nn.Sequential(torch.nn.Conv1d(nIn,nOut,ks,stride),
                               torch.nn.ReLU(),
                               torch.nn.BatchNorm1d(nOut))

def deconvBlock(nIn,nOut,ks,stride):
    return torch.nn.Sequential(torch.nn.ConvTranspose1d(nIn,nOut,ks,stride),
                               torch.nn.ReLU(),
                               torch.nn.BatchNorm1d(nOut))
# -

class TSAE(torch.nn.Module):
    def __init__(self,seqLen):
        super().__init__()
        self.conv1 = convBlock(1,10,4,2)
        self.conv2 = convBlock(10,20,3,2)
        self.conv3 = convBlock(20,40,3,2)
        self.deconv1 = deconvBlock(40,20,3,2)
        self.deconv2 = deconvBlock(20,10,3,2)
        self.deconv3 = deconvBlock(10,1,4,2)
        
    def forward(self,ts):
        ts = self.conv3(self.conv2(self.conv1(ts.unsqueeze(1))))
        ts = self.deconv3(self.deconv2(self.deconv1(ts)))
        return ts.squeeze(1)

class TSClassifier(torch.nn.Module):
    def __init__(self,seqLen,numClasses):
        super().__init__()
        self.conv1 = convBlock(1,10,4,2)
        self.conv2 = convBlock(10,20,3,2)
        self.conv3 = convBlock(20,1,4,2)
        self.conv4 = convBlock(40,80,4,2)
        self.conv5 = convBlock(80,160,4,2)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.out = torch.nn.Linear(30,numClasses)
        
    def forward(self,ts):
        ts = self.conv3(self.conv2(self.conv1(ts.unsqueeze(1))))
        #ts = self.conv5(self.conv4(ts))
        #ts = self.pool(ts)
        return self.out(ts.squeeze(1))

learnAE = Learner(dataAE,TSAE(seqLen),loss_func=torch.nn.functional.mse_loss)

x,y = next(iter(dataAE.train_dl))
learnAE.model(x).size()

learnAE.lr_find()
learnAE.recorder.plot()

learnAE.fit_one_cycle(20,1e-2)

learnClassifier = Learner(dataClassifier,TSClassifier(seqLen,numClasses),loss_func=torch.nn.functional.cross_entropy,metrics=[accuracy])
learnClassifier.split(split_model_idx(learnClassifier.model,idxs=[0,6]))

learnClassifier.model.conv1.load_state_dict(learnAE.model.conv1.state_dict())
learnClassifier.model.conv2.load_state_dict(learnAE.model.conv2.state_dict())

x,y = next(iter(dataClassifier.train_dl))
learnClassifier.model(x).size()

learnClassifier.lr_find()
learnClassifier.recorder.plot()

learnClassifier.freeze_to(1)

learnClassifier.fit_one_cycle(20,1e-2)

learnClassifier.unfreeze()

learnClassifier.fit_one_cycle(20,[1e-3,1e-2])

x,y = next(iter(dataAE.train_dl))
learnAE.model.eval()
preds = learnAE.model(x)
_,axes = plt.subplots(3,3,figsize=(20,16))
for ax, ts, p, in zip(axes.flatten(),x,preds):
    ax.plot(ts)
    ax.plot(p)
    
