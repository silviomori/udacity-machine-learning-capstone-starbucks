import torch
import torch.nn as nn


class LinearNN(nn.Module):
    def __init__(self, input_size=18, hidden_layers=[128,128],
                 output_size=1, dropout=0.2):
        super(LinearNN, self).__init__()
        self.input_size = input_size

        nodes = []
        nodes.append(input_size)
        nodes.extend(hidden_layers)
        nodes.append(output_size)

        self.module_list = nn.ModuleList()
        for nodes_in, nodes_out in zip(nodes[:-1], nodes[1:]):
            layer = nn.Linear(nodes_in, nodes_out)
            self.module_list.append(layer)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        for layer in self.module_list[:-1]:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.module_list[-1](x)

        return x


class RecurrentNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, hidden_layers=2,
                 classifier_layers=[128], output_size=1, dropout=0.2):
        super(RecurrentNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        ## Build the Recurrent Network
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=hidden_layers,
                          batch_first=True,
                          dropout=dropout)

        self.hidden_tensor = torch.zeros(hidden_layers, 1, hidden_size) \
                             .uniform_(-3e-3, 3e-3)

        ## Build the classifier
        nodes = []
        nodes.append(hidden_size)
        nodes.extend(classifier_layers)
        nodes.append(output_size)
        self.classifier = nn.ModuleList()
        for nodes_in, nodes_out in zip(nodes[:-1], nodes[1:]):
            layer = nn.Linear(nodes_in, nodes_out)
            self.classifier.append(layer)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_length = x.shape[1]

        ## Feed the RNN
        h = self.init_hidden(batch_size)
        x, h = self.rnn(x, h)

        ## Feed the Classifier
        x = x.contiguous().view(-1, self.hidden_size)
        x = self.dropout(x)
        for layer in self.classifier[:-1]:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.classifier[-1](x)

        ## Stack the results
        x = x.view(batch_size, seq_length, self.output_size)

        return x

    def init_hidden(self, batch_size):
        self.hidden_tensor.repeat(1, batch_size, 1)

