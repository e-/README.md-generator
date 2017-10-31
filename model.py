import torch 
import torch.nn as nn
from torch.autograd import Variable

class LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers):
        super(LSTMModel, self).__init__()

        self.encoder = nn.Embedding(ntoken, ninp)
        self.lstm = nn.LSTM(ninp, nhid, nlayers)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, batch_size, self.nhid).zero_()), Variable(weight.new(self.nlayers, batch_size, self.nhid).zero_())

    def forward(self, input, hidden):
        emb = self.encoder(input)
        output, hidden = self.lstm(emb, hidden)
        decoded = self.decoder(output)
        
        return decoded, hidden

