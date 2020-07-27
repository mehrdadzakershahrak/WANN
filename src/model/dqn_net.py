import torch.nn.functional as F
import torch.nn as nn
import torch as T


class DqnCnn(nn.Module):
    def __init__(self, w, h, n_acts, kwargs=None):
        super(DqnCnn, self).__init__()

        if kwargs is not None and 'dropout' in kwargs:
            self.dropout = nn.Dropout(p=kwargs['dropout'])
            self.use_dropout = True
        else:
            self.use_dropout = False

        self.n_acts = n_acts

        self.c1 = nn.Conv2d(3, 64, kernel_size=3, padding=(1, 1), stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.c2 = nn.Conv2d(64, 32, kernel_size=3, padding=(1, 1), stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 16, kernel_size=3, padding=(1, 1), stride=1)
        self.bn3 = nn.BatchNorm2d(16)

        self.n_f1_input = 16*w*h
        self.f1 = nn.Linear(self.n_f1_input, self.n_f1_input/float(2))
        self.head = nn.Linear(self.n_f1_input/float(2), n_acts)

        self.n_conv_layers = 3

    def forward(self, s):
        n = s
        for i in range(1, self.n_conv_layers):
            n = F.relu(getattr(self, f'bn{i}')(getattr(self, f'c{i}')(n)))

        n = n.view(self.n_f1_input, -1)

        n = F.relu(self.f1(n))
        if self.use_dropout:
            n = self.dropout(n)

        return self.head(n)


class DqnVnn(nn.Module):
    def __init__(self, n_in_feats, n_acts, kwargs=None):
        super(DqnVnn, self).__init__()

        if kwargs is not None and 'dropout' in kwargs:
            self.dropout = nn.Dropout(p=kwargs['dropout'])
            self.use_dropout = True
        else:
            self.use_dropout = False

        self.f1 = nn.Linear(n_in_feats, n_in_feats)
        self.f2 = nn.Linear(n_in_feats, n_in_feats)
        self.head = nn.Linear(n_in_feats, n_acts)

        self.n_hlayers = 2

    def forward(self, s):
        n = s
        for i in range(1, self.n_hlayers):
            n = F.relu(getattr(self, f'f{i}')(n))

            if self.use_dropout:
                n = self.dropout(n)

        return self.head(n)
