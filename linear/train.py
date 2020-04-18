import sys
sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np
from dataset.parser import get_names
from dataset.dataset import Dataset
from model import model
from vars import window_size


batch_size = 64
epochs = 100
model = model(n_in=window_size, dropout=0.1)

criterium = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
names = get_names()

dataset = Dataset(names[:18], window_size=window_size)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=0,
                                         # sampler=sampler
                                         )

for _ in range(epochs):
  for k,(data,target) in enumerate(dataloader):
    # print(target)
    # print(np.count_nonzero(target[0].data.numpy()))
    # Set gradient to 0.
    optimizer.zero_grad()
    # Feed forward.
    pred = model(data).view(-1)
    # Loss calculation.
    loss = criterium(pred, target)
    # Gradient calculation.
    loss.backward()

    # Print loss every 10 iterations.
    if k % 10 == 0:
      print('Loss {:.4f} at iter {:d}'.format(loss.item(), k))

    # Model weight modification based on the optimizer.
    optimizer.step()

torch.save(model.state_dict(), 'model.pt')

