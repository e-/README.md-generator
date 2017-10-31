# -*- coding: utf-8 -*-

import argparse
import torchtext as tt
import torch
from torch.autograd import Variable
import data

parser = argparse.ArgumentParser(description='Generate README.md using the given language model')

# Model parameters.
parser.add_argument('--data', type=str, default='data/crawl.txt',
                    help='data path')
parser.add_argument('--model', type=str, default='model.pt',
                    help='input model')
parser.add_argument('--outf', type=str, default='generated.md',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--temperature', type=float, default=1.05,
                    help='temperature - higher will increase diversity')
parser.add_argument('--print-every', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--vocab-max-size', type=int, default=240000, metavar='N',
                    help='vocab max size, must be equal to the value used for model construction')
                                        
args = parser.parse_args()

model = torch.load(args.model)
model.eval()
model.cuda()

text_field = tt.data.Field()
train = tt.datasets.LanguageModelingDataset(args.data, text_field)
text_field.build_vocab(train, max_size=args.vocab_max_size)

ntokens = len(text_field.vocab)
hidden = model.init_hidden(1)
input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True).cuda()

with open(args.outf, 'w', encoding='utf8') as outf:
    for i in range(args.words):
        output, hidden = model(input, hidden)
        output = output.squeeze()

        word_weights = output.data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input.data.fill_(word_idx)
        word = text_field.vocab.itos[word_idx]

        outf.write(bytes(word, 'utf-8').decode('unicode_escape') + ' ')

        if i % args.print_every == 0:
            print('| Generated {}/{} words'.format(i, args.words))