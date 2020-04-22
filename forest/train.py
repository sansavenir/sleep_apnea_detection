import sys
sys.path.append('../')
import numpy as np
from dataset.parser import get_names
from dataset.dataset import Dataset
from dataset.sampler import BalancedBatchSampler
from sklearn.ensemble import RandomForestClassifier
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import wandb
wandb.init(project="osa")


names = get_names()

window_size = 80
signals=['Sound']
dataset = Dataset(names[:18], window_size=window_size)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset),
                                         shuffle=True, num_workers=0)
(data, target) = zip(*dataloader)

clf = RandomForestClassifier(max_depth=300, random_state=0)

clf.fit(data[0].data.numpy(),target[0].data.numpy())
names = get_names()

dataset = Dataset(names[:18], window_size=window_size, signals=signals)

train_sampler = BalancedBatchSampler(dataset, is_training=True, split=1.)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=len(train_sampler),
                                         shuffle=False, num_workers=0, sampler=train_sampler)

(data, target) = zip(*train_loader)
pred = clf.predict(data[0].data.numpy())
wandb.sklearn.plot_confusion_matrix(target[0].data.numpy(), pred)


print(confusion_matrix(target[0].data.numpy(), pred))
print(f1_score(target[0].data.numpy(),pred))
print(recall_score(target[0].data.numpy(),pred))


