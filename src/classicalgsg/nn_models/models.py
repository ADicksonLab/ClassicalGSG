import torch.nn as nn


class GSGNN(nn.Module):
    def __init__(self, n_in, n_h=100, n_layers=1, dropout=0.0):
        """FIXME! briefly describe function

        :param n_in:
        :param n_h:
        :param n_layers:
        :param dropout:
        :returns:
        :rtype:

        """

        super(GSGNN, self).__init__()
        self.n_in = n_in

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_in, n_h))
        self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.ReLU())

        for _ in range(n_layers):
            self.layers.append(nn.Linear(n_h, n_h))
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(n_h, 1))

    def forward(self, x):
        """FIXME! briefly describe function

        :param x:
        :returns:
        :rtype:

        """

        y = x.view(-1, self.n_in)
        for _, layer in enumerate(self.layers):
            y = layer(y)

        return y


class OneLayerNN(nn.Module):
    def __init__(self, n_in, n_h=1, dropout=0.0):
        """FIXME! briefly describe function

        :param n_in:
        :param n_h:
        :param n_layers:
        :param dropout:
        :returns:
        :rtype:

        """

        super(OneLayerNN, self).__init__()
        self.n_in = n_in

        self.hidden = nn.Linear(n_in, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """FIXME! briefly describe function

        :param x:
        :returns:
        :rtype:

        """

        y = x.view(-1, self.n_in)
        y = self.hidden(y)
        y = self.dropout(y)
        y = self.relu(y)

        return y
