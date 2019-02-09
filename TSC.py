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
from fastai.basics import *
path = Path("UCRArchive_2018")

list((path/"car").iterdir())

class TSCDS(torch.utils.data.Dataset):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        self.items = [1,2,3]
    
    def __len__(self): return len(self.X)
    
    def __getitem__(self,idx):
        return np.expand_dims(self.X[idx],axis=0), self.Y[idx] - 1

# +
class ConvBlock(torch.nn.Module):
    def __init__(self,inputC,outputC,kernelSize):
        super().__init__()
        self.conv = torch.nn.Conv1d(inputC,outputC,kernelSize)
        self.drop = torch.nn.Dropout()
        self.bn = torch.nn.BatchNorm1d(outputC)
        
    def forward(self,x):
        return self.drop(self.bn(torch.relu(self.conv(x))))

class TSCModel(torch.nn.Module):
    def __init__(self,numClasses):
        super().__init__()
        self.blocks = torch.nn.ModuleList([ConvBlock(i,o,k) for i,o,k in [(1,128,8),(128,256,5),(256,128,3)]])
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.out = torch.nn.Linear(128,numClasses)
        
    def forward(self,x):
        x = x.float()
        for b in self.blocks: x = b(x)
        x = self.avgpool(x)
        return self.out(x.squeeze(-1))
# -

def trainNetwork(d,weights=None,epochs=20,lr=5e-2):
    trainDF = pd.read_csv(path/f"{d}/{d}_TRAIN.tsv",sep="\t",header=None)
    testDF = pd.read_csv(path/f"{d}/{d}_TEST.tsv",sep="\t",header=None)
    cat = trainDF.iloc[:,0].astype("category")
    trainDF.iloc[:,0] = cat.cat.codes
    testDF.iloc[:,0] = pd.Categorical(testDF.iloc[:,0],categories=cat.cat.categories).codes
    nClasses = len(cat.cat.categories)
    
    trainDS = TSCDS(trainDF.iloc[:,1:].values,trainDF.iloc[:,0].values)
    testDS = TSCDS(testDF.iloc[:,1:].values,testDF.iloc[:,0].values)
    model = TSCModel(nClasses)
    learn = Learner(data,model,loss_func=torch.nn.functional.cross_entropy,metrics=[accuracy])
    if weights:
        learn.model.blocks.load_state_dict(weights)
    
    learn.fit_one_cycle(epochs,lr)
    return learn.model.blocks.state_dict()

weights = None
for p in path.iterdir():
    if p.is_dir():
        weights = trainNetwork(p.name,weights)

w = trainNetwork("Worms",lr=1e-2,epochs=100)

trainNetwork("Wine",w,lr=1e-2,epochs=100)
