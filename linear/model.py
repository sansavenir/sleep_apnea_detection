import torch.nn as nn


class model(nn.Module):
  def __init__(self ,n_in, n_hidden=160, n_out=1, dropout=0.):
    super(model ,self).__init__()
    self.n_in  = n_in
    self.n_out = n_out
    self.n_hidden = n_hidden

    self.layer_1 = nn.Linear(self.n_in, self.n_hidden)
    self.layer_2 = nn.Linear(self.n_hidden, self.n_hidden)
    self.layer_3 = nn.Linear(self.n_hidden, self.n_hidden)
    self.layer_out = nn.Linear(self.n_hidden, self.n_out)

    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=dropout)
    self.batchnorm1 = nn.BatchNorm1d(n_hidden)
    self.batchnorm2 = nn.BatchNorm1d(n_hidden)
    self.batchnorm3 = nn.BatchNorm1d(n_hidden)
    self.sigmoid = nn.Sigmoid()

  def forward(self, inputs):
    x = self.relu(self.layer_1(inputs))
    x = self.batchnorm1(x)
    x = self.relu(self.layer_2(x))
    x = self.batchnorm2(x)
    x = self.relu(self.layer_3(x))
    x = self.batchnorm3(x)
    x = self.dropout(x)
    x = self.layer_out(x)
    x = self.sigmoid(x)


    return x