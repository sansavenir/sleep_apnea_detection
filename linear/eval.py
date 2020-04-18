import sys
sys.path.append('../')
from model import model
import torch
from vars import window_size
from dataset.dataset import Dataset
from dataset.parser import get_names
from sklearn.metrics import average_precision_score
from torch.utils.tensorboard import SummaryWriter
import wandb
from sklearn.metrics import precision_recall_curve

wandb.init(project="osa", sync_tensorboard=True)



model = model(n_in=window_size)
wandb.watch(model)
model.load_state_dict(torch.load('model.pt'))

names = get_names()
dataset = Dataset(names[18:], is_training=False, window_size=window_size)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset),
                                         shuffle=False, num_workers=0)
(data, target) = zip(*dataloader)

pred = model(data[0])
print(average_precision_score(target[0].data.numpy(),pred.data.numpy()))

writer = SummaryWriter()
writer.add_pr_curve('pr_curve', target[0].data.numpy().reshape([-1]), pred.reshape([-1]), num_thresholds=100)
wandb.log({'average_precision_score': average_precision_score(target[0].data.numpy(),pred.data.numpy())})
writer.close()



