# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchtext as tt
import argparse
from torch.autograd import Variable
from model import LSTMModel
from torch.optim.adam import Adam
import math

parser = argparse.ArgumentParser(description='Generate a language model using LSTM')

parser.add_argument('--data', type=str, default='data/crawl.txt',
                    help='path to train data')
parser.add_argument('--batch-size', type=int, default=15,
                    help='batch size')
parser.add_argument('--bptt', type=int, default=30,
                    help='sequence length')
parser.add_argument('--emsize', type=int, default=50,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=50,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=250,
                    help='upper epoch limit')
parser.add_argument('--print-every', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--vocab-max-size', type=int, default=240000, metavar='N',
                    help='vocab max size')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')

criterion = nn.CrossEntropyLoss()

args = parser.parse_args()

text_field = tt.data.Field()
train = tt.datasets.LanguageModelingDataset(args.data, text_field)
train_iter = tt.data.BPTTIterator(train, batch_size=args.batch_size, bptt_len=args.bptt, sort_key=lambda x: len(x.text), shuffle=True, repeat=False)

text_field.build_vocab(train, max_size=args.vocab_max_size)
ntokens = len(text_field.vocab)
model = LSTMModel(ntokens, args.emsize, args.nhid, args.nlayers)

optim = Adam(model.parameters(), lr=args.lr)
model.cuda()

def detach(hidden):
    if type(hidden) == Variable:
        return Variable(hidden.data)
    else:
        return tuple(detach(v) for v in hidden)

for epoch in range(1, args.epochs+1):
    hidden = model.init_hidden(args.batch_size)
    count = 0

    for batch in train_iter:
        text = batch.text.cuda()
        target = batch.target.cuda()
        hidden = detach(hidden)
        optim.zero_grad()

        output, hidden = model(text, hidden)
        loss = criterion(output.view(-1, ntokens), target.view(-1))

        loss.backward()

        optim.step()
        count += 1

        if count % args.print_every == 0:
            curr_loss = loss.data[0]
            print('| epoch {:3d} | batch {:5d} | loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, count, curr_loss, math.exp(curr_loss)))

    torch.save(model, args.save)
    