import sys
sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np
from dataset.parser import get_names
from dataset.dataset import Dataset
from dataset.sampler import BalancedBatchSampler
from model import model
from vars import window_size, signals
from tqdm import tqdm


batch_size = 64
epochs = 3000


model = model(n_in=window_size*len(signals), dropout=0.1)

criterium = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
names = get_names()

dataset = Dataset(names[:18], window_size=window_size, signals=signals)
sampler = BalancedBatchSampler(dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=0, sampler=sampler)

with tqdm(total=epochs) as pbar:
  for _ in range(epochs):
    loss_t = 0
    mk = 0
    for k,(data,target) in enumerate(dataloader):
      # print(target)
      if data.shape[0] == 1:
        continue
      optimizer.zero_grad()
      pred = model(data.view([data.shape[0], -1])).view(-1)
      loss = criterium(pred, target.view(-1))
      loss_t += float(loss.item())
      mk = max(mk,k)
      loss.backward()
      optimizer.step()
    pbar.set_description("Loss %f" % (loss_t / (mk+1)))
    pbar.update(1)

torch.save(model.state_dict(), 'model.pt')

