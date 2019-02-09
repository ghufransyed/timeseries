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

class TSAE(torch.nn.Module):
    def __init__(self,seqLen):
        super().__init__()
        self.lin1 = torch.nn.Linear(seqLen,20)
        self.lin2 = torch.nn.Linear(20,300)
        self.out = torch.nn.Linear(300,seqLen)
        
    def forward(self,ts):
        ts = torch.relu(self.lin1(ts))
        ts = torch.relu(self.lin2(ts))
        return self.out(ts)

class TSClassifier(torch.nn.Module):
    def __init__(self,seqLen,numClasses):
        super().__init__()
        self.lin1 = torch.nn.Linear(seqLen,20)
        self.lin2 = torch.nn.Linear(20,300)
        self.out = torch.nn.Linear(300,numClasses)
        
    def forward(self,ts):
        ts = torch.relu(self.lin1(ts))
        ts = torch.relu(self.lin2(ts))
        return self.out(ts)

learnAE = Learner(dataAE,TSAE(seqLen),loss_func=torch.nn.functional.mse_loss)

learnAE.lr_find()
learnAE.recorder.plot()

learnAE.fit_one_cycle(20,1e-2)

learnClassifier = Learner(dataClassifier,TSClassifier(seqLen,numClasses),loss_func=torch.nn.functional.cross_entropy,metrics=[accuracy])
learnClassifier.split(split_model_idx(learnClassifier.model,idxs=[0,2]))

learnClassifier.layer_groups

learnClassifier.model.lin1.load_state_dict(learnAE.model.lin1.state_dict())
learnClassifier.model.lin2.load_state_dict(learnAE.model.lin2.state_dict())

x,y = next(iter(dataClassifier.train_dl))
learnClassifier.model(x).size()

learnClassifier.lr_find()
learnClassifier.recorder.plot()

#Without transfer
learnClassifier.fit_one_cycle(20,1e-2)

learnClassifier.freeze_to(1)

#With transfer
learnClassifier.fit_one_cycle(20,1e-2)

learnClassifier.unfreeze()

learnClassifier.fit_one_cycle(10,[1e-4,1e-3])

x,y = next(iter(dataAE.train_dl))
learnAE.model.eval()
preds = learnAE.model(x)
_,axes = plt.subplots(3,3,figsize=(20,16))
for ax, ts, p, in zip(axes.flatten(),x,preds):
    ax.plot(ts)
    ax.plot(p)
    
