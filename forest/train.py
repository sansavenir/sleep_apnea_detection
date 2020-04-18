import sys
sys.path.append('../')
import numpy as np
from dataset.parser import get_names
from dataset.dataset import Dataset
from sklearn.ensemble import RandomForestClassifier
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import wandb
wandb.init(project="osa")


names = get_names()

window_size = 80
dataset = Dataset(names[:18], window_size=window_size)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset),
                                         shuffle=True, num_workers=0)
(data, target) = zip(*dataloader)

clf = RandomForestClassifier(max_depth=300, random_state=0)

clf.fit(data[0].data.numpy(),target[0].data.numpy())
names = get_names()

dataset = Dataset(names[18:], is_training=False, window_size=window_size)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset),
                                         shuffle=True, num_workers=0,
                                         )
(data, target) = zip(*dataloader)
pred = clf.predict(data[0].data.numpy())
wandb.sklearn.plot_confusion_matrix(target[0].data.numpy(), pred)


# print(confusion_matrix(target[0].data.numpy(), pred))
# print(f1_score(target[0].data.numpy(),pred))
# print(recall_score(target[0].data.numpy(),pred))


